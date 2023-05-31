# Object Tracking
## _Using YOLOv7 and DeepSORT_

This is a project that utilizes YOLOv7 for object detection and DeepSORT for object tracking. The project aims to automate the process of tracking objects in videos or images, providing accurate and efficient tracking results.

## ✨Features✨

- Simple and user-friendly command line interface for object tracking.
- Fast and precise tracking capabilities.
- Easy integration into existing projects.
- Real-time tracking of objects.
- Shows FPS as well as number of objects in frame.
- Shows tracks followed by each object in source.
- Source can be a video file or live feed.
- Assigns ID's to each object in frame.
- Shows Live feed and has capabilities to save the output in video file.

## Tech

Object Tracking employs several open-source technologies to function optimally:

- [Pytorch] - Machine learning framework.
- [YOLOv7] - Awesome computer vision model. Used here for object detection.
- [DeepSORT] - A deep learning-based object tracking algorithm that combines the Kalman filter and Hungarian algorithm.
- [OpenCV] - Open source computer vision and machine learning software library.

## Usage
Object tracking requires [Python](https://www.python.org/) 3.6+ to run.
- Clone repository
    ```sh
    git clone https://github.com/yashjain-99/object_tracking.git 
    ```
- Move to cloned repository
    ```sh
    cd object_tracking
    ```
- Install the dependencies:
    ```py
    pip install -r requirements.txt
    ```
- Download trained models from [here] and store it inside models_weights folder.
- Clone DeepSORT repository
    ```sh
    git clone https://github.com/nwojke/deep_sort.git
    ```
- Copy and replace following files from main directory to .\deep_sort\deep_sort 
    ```sh
    cp .\linear_assignment.py .\deep_sort\deep_sort\
    cp .\detection.py .\deep_sort\deep_sort\ 
    ```
- Now you are good to go to make predictions
    - To use the command line interface for making predictions:
        ```sh
        python main.py --source input_video.mp4 --output pred.mp4 --show_live --count_obj --show_tracks --show_fps
        ```
        > Replace input_video.mp4 with path of your orignal video and pred.mp4 with path where you want output to be saved.

    - You could use a python file to make prediction:
        ```py
        from detection_helpers import *
        from tracking_helpers import *
        from  wrapper import *

        detector = Detector(classes = [0]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
        detector.load_model('model_weights/yolov7-tiny.pt',)
        tracker = YOLOv7_DeepSORT(reID_model_path="model_weights/mars-small128.pb", detector=detector)
        tracker.track_video("video.mp4", output="out.avi", show_live = True, skip_frames = 0, count_objects = True, show_fps=True, verbose=1, show_tracks=True)
        ```
**Please feel free to reach out to me at yashj133.yj@gmail.com in case of any queries or suggestions!**

   [YOLOv7]: <https://github.com/WongKinYiu/yolov7>
   [Pytorch]: <https://pytorch.org/>
   [DeepSORT]: <https://github.com/nwojke/deep_sort>
   [OpenCV]: <https://opencv.org/>
   [here]: <https://drive.google.com/drive/folders/1TCY6wLHxzXhlw5KCMVIcaF3pKQkJJWZI?usp=sharing>