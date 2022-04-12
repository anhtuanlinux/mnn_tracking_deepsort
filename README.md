# mnn_tracking_deepsort
Python code run MNN face detection and tracking using DeepSort

Install and run this project

Step 1: Clone the repository: 

`git clone https://github.com/anhtuanlinux/mnn_tracking_deepsort.git`

Step 2: CD to the main folder

`cd mnn_tracking_deepsort/MNN/python`

Step 2: Install libraries and all requirements dependencies:

`sudo apt install graphviz`

`pip install -U pip`

`pip install -U MNN`

`pip install -U Cython`

`pip install -r requirements.txt`

Step 3: Run MNN face detection + DeepSORT tracking code

`bash run_tracking_ultraface_mnn.sh`

where --model_path: MNN model file path

      --source: input video file path
      
      --config_deepsort: yaml file that storing deepsort config
           
      --deep_sort_model: reid checkpoint model file path
      
      --save-vid (True) for saving output video
      
      --save-txt (True) for saving output text file (MOT output format)

Result:

Output video is saved at `mnn_tracking_deepsort/MNN/python/runs/track/face_detect_osnet_ibn_x1_0_MSMT17/short_video_test_02.mp4`

Output MOT format text file is saved at  `mnn_tracking_deepsort/MNN/python/runs/track/face_detect_osnet_ibn_x1_0_MSMT17/tracks/short_video_test_02.txt`
