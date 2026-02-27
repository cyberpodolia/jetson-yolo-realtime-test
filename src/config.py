"""Centralized runtime defaults."""


class RuntimeDefaults(object):
    engine = "artifacts/joystick_fp16.engine"
    camera = 0
    frames = 300
    imgsz = 320
    conf = 0.60
    iou = 0.70
    width = 640
    height = 480
    fps = 30
    warmup = 10
    max_det = 10
    det_interval = 1
    track_interval = 1
    tracker = "off"


DEFAULTS = RuntimeDefaults()
