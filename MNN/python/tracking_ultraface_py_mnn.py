#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
""" 
reference from:
@author:linzai 
@file: ultraface_py_mnn.py 
@time: 2019-11-25 
"""
from __future__ import print_function

# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import platform
import shutil
from pathlib import Path

import argparse
import sys
import time
from math import ceil

import MNN
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utils.datasets import LoadImages, xyxy2xywh, VID_FORMATS, increment_path
from utils.plots import Annotator, colors


sys.path.append('../../')
import vision.utils.box_utils_numpy as box_utils

# Config setting for MNN
image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
strides = [8, 16, 32, 64]


def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [ceil(size / stride) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def inference(args):
    out, source, model_path, deep_sort_model, save_vid, save_txt, project, exist_ok = \
        args.output, args.source, args.model_path, args.deep_sort_model, args.save_vid, \
        args.save_txt, args.project, args.exist_ok
    input_size = [int(v.strip()) for v in args.input_size.split(",")]
    priors = define_img_size(input_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Initialize 
    if os.path.exists(out):
        pass
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    exp_name = 'face_detect' + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Dataloader
    dataset = LoadImages(source, input_size=input_size, image_mean=image_mean, image_std=image_std)
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    # Run detection and tracking
    for frame_idx, (path, image, image_ori, vid_cap, s) in enumerate(dataset):
        # Create MNN session and prepare input tensor then run Session for predict
        interpreter = MNN.Interpreter(model_path)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)
        tmp_input = MNN.Tensor((1, 3, input_size[1], input_size[0]), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)
        time_time = time.time()
        interpreter.runSession(session)
        # Get scores, boxes from output Session
        scores = interpreter.getSessionOutput(session, "scores").getData()
        boxes = interpreter.getSessionOutput(session, "boxes").getData()
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        print("[FRAME {}] inference time: {} s".format(frame_idx, round(time.time() - time_time, 4)))

        # Convert output boxes and nms
        boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
        boxes = box_utils.center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(image_ori.shape[1], image_ori.shape[0], scores, boxes, args.threshold)
        if boxes is not None and len(boxes):
            xywhs = xyxy2xywh(boxes)
            confs = probs
            # print(confs)
            clss = labels
            seen += 1
            p, im0, _ = path, image_ori.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # video file
            if source.endswith(VID_FORMATS):
                txt_file_name = p.stem
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            annotator = Annotator(im0, line_width=2)
        # if boxes is not None and len(boxes):
            # pass detections to deepsort
            outputs[0] = deepsort_list[0].update(xywhs, confs, clss, im0)
            if len(outputs[0]) > 0:
                for j, (output, conf) in enumerate(zip(outputs[0], confs)):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    if save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                    if save_vid:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{id} class={c} {conf:.2f}'
                        # draw boxes for visualization
                        annotator.box_label(bboxes, label, color=colors(c, True))
        else:
            # increment age if does not get any prediction in this frame
            deepsort_list[0].increment_ages()

        # Stream results
        # im0 = annotator.result()
        # Save results (image with detections)
        if save_vid:
            if vid_path[0] != save_path:  # new video
                vid_path[0] = save_path
                if isinstance(vid_writer[0], cv2.VideoWriter):
                    vid_writer[0].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[0].write(im0)
    print(f"DONE! text file result was saved at {txt_path} \n Output video was saved at {save_path}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run ultraface with MNN in py')
    parser.add_argument('--model_path', default="../model/version-RFB/RFB-320.mnn", type=str, help='model path')
    parser.add_argument('--input_size', default="320,240", type=str, help='define network input size,format: width,height')
    parser.add_argument('--threshold', default=0.5, type=float, help='score threshold')
    parser.add_argument('--source', default="../videos", type=str, help='video dir')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument('--project', default= './runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    args = parser.parse_args()
    inference(args)