import threading
import time

import cv2


def build_v4l2_pipeline(
    camera=0,
    width=640,
    height=480,
    fps=30,
    pixel_format="YUY2",
):
    """Return a basic GStreamer pipeline string for USB camera capture."""
    return (
        "v4l2src device=/dev/video{camera} ! "
        "video/x-raw,format={fmt},width={w},height={h},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    ).format(camera=camera, fmt=pixel_format, w=width, h=height, fps=fps)


def open_camera(
    camera=0,
    width=640,
    height=480,
    fps=30,
    use_gstreamer=False,
    gst_pipeline=None,
):
    """Open camera with GStreamer or V4L2 fallback."""
    cap = None

    if gst_pipeline:
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    elif use_gstreamer:
        cap = cv2.VideoCapture(
            build_v4l2_pipeline(camera=camera, width=width, height=height, fps=fps),
            cv2.CAP_GSTREAMER,
        )

    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


class FrameGrabber(object):
    """Background capture thread that always keeps the latest frame."""

    def __init__(self, cap):
        self._cap = cap
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._frame_id = 0

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, name="frame-grabber", daemon=True
        )
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = bool(ok)
                if ok:
                    self._frame = frame
                    self._frame_id += 1
            if not ok:
                time.sleep(0.002)

    def read(self, last_frame_id, timeout_sec=1.0):
        """Return (ok, frame, frame_id), waiting for a newer frame."""
        t0 = time.time()
        while True:
            with self._lock:
                ok = self._ok
                frame = None if self._frame is None else self._frame.copy()
                frame_id = self._frame_id
            if ok and frame is not None and frame_id > last_frame_id:
                return True, frame, frame_id
            if time.time() - t0 > timeout_sec:
                return False, None, last_frame_id
            time.sleep(0.001)

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
