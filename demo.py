from detection_helpers import *
from tracking_helpers import *
from  wrapper import *
from PIL import Image

detector = Detector(classes = [0]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('model_weights/yolov7x.pt',)
tracker = YOLOv7_DeepSORT(reID_model_path="model_weights/mars-small128.pb", detector=detector)
tracker.track_video("video.mp4", output="out.avi", show_live = True, skip_frames = 0, count_objects = True, show_fps=True, verbose=1)
