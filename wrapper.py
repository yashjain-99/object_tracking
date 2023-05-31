'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

from detection_helpers import *
from tracking_helpers import read_class_names, create_box_encoder
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort import nn_matching
from deep_sort.application_util import preprocessing
# DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import time
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# deep sort imports

# import from helpers

# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True


class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''

    def __init__(self, reID_model_path: str, detector, max_cosine_distance: float = 0.4, nn_budget: float = None, nms_max_overlap: float = 1.0,
                 coco_names_path: str = "./io_data/input/classes/coco.names",):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)  # calculate cosine distance metric
        self.tracker = Tracker(metric)  # initialize tracker

    def track_video(self, video: str, output: str, skip_frames: int = 0, show_live: bool = False, count_objects: bool = False, show_fps: bool = False, verbose: int = 0, show_tracks: bool = False):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try:  # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        if output:  # get video ready to save locally if flag is set
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0
        d_tracks = {}
        while True:  # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num += 1

            if skip_frames and not frame_num % skip_frames:
                continue  # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:
                start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            yolo_dets = self.detector.detect(
                frame.copy(), plot_bb=False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0

            else:
                bboxes = yolo_dets[:, :4]
                # convert from xyxy to xywh
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
                bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

                scores = yolo_dets[:, 4]
                classes = yolo_dets[:, -1]
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------

            names = []
            # loop through objects and use class index to get class name
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            count = len(names)

            if count_objects:
                # cv2.rectangle(frame, (0,0), (500, 85), (0,0,0), -1)
                cv2.putText(frame, "Objects being tracked: {}".format(
                    count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            features = self.encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(
                bboxes, scores, names, features)]  # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b')  # initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression below
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(
                boxs, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections)  # updtate using Kalman Gain

            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                # track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                bbox = track.to_tlbr()
                CX = int((bbox[0]+bbox[2])//2)
                CY = int((bbox[1]+bbox[3])//2)

                class_name = 'person'
                color = colors[int(track.track_id) %
                               len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(
                    bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
                    len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id), (int(
                    bbox[0]), int(bbox[1]-11)), 0, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                
                #storing tracks
                if str(track.track_id) in d_tracks:
                    d_tracks[str(track.track_id)].append([CX, CY])
                    if len(d_tracks[str(track.track_id)]) > 50:
                        d_tracks[str(track.track_id)].pop(0)
                else:
                    d_tracks[str(track.track_id)] = [[CX, CY]]
                if show_tracks:
                    [cv2.line(frame, (int(d_tracks[str(track.track_id)][i][0]),
                                      int(d_tracks[str(track.track_id)][i][1])),
                              (int(d_tracks[str(track.track_id)][i+1][0]),
                               int(d_tracks[str(track.track_id)][i+1][1])),
                              color, thickness=2)
                     for i, _ in enumerate(d_tracks[str(track.track_id)])
                        if i < len(d_tracks[str(track.track_id)])-1]

                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                        str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if verbose >= 1:
                # calculate frames per second of running detections
                fps = 1.0 / (time.time() - start_time)
                if not count_objects:
                    print(
                        f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else:
                    print(
                        f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")

            if show_fps:
                # 300
                cv2.putText(frame, "FPS: {}".format({round(fps, 2)}), (5, 65),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 0), 2)

            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if output:
                out.write(result)  # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
