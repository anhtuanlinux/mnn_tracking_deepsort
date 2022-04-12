# mnn_tracking_deepsort
Python code run tracking using deepsort and mnn face detection 

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
