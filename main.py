import typer
from detection_helpers import *
from tracking_helpers import *
from  wrapper import *
app = typer.Typer(help="CLI for object tracking using yolov7 and deepSORT")


@app.command()
def object_tracker(
        source:str                = typer.Option(..., "--source", "-s"),
        output:str                = typer.Option(..., "--output", "-o"),
        show_live:bool            = True,
        count_obj:bool            = True,
        show_tracks:bool          = True,
        show_fps:bool             = True,
        verbose:int               = typer.Option(1, "--verbose")):
    detector = Detector(classes = [0]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
    detector.load_model('model_weights/yolov7-tiny.pt',)
    tracker = YOLOv7_DeepSORT(reID_model_path="model_weights/mars-small128.pb", detector=detector)
    tracker.track_video(source, output=output, show_live = show_live, skip_frames = 0, count_objects = count_obj, show_fps=show_fps, verbose=verbose, show_tracks=show_tracks)
    
if __name__ == "__main__":

    app()