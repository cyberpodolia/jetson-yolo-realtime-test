import cv2

from .postprocess import Detection


def _create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    raise RuntimeError("OpenCV build does not provide CSRT tracker API")


def csrt_available():
    """Return True if OpenCV has CSRT tracker constructors."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return True
    legacy = getattr(cv2, "legacy", None)
    return legacy is not None and hasattr(legacy, "TrackerCSRT_create")


class TrackerUpdate(object):
    def __init__(self, detection=None, ok=False):
        self.detection = detection
        self.ok = bool(ok)


class CsrtTracker(object):
    """Single-object CSRT tracker wrapper for tracking-by-detection."""

    def __init__(self):
        self._tracker = None
        self._active = False
        self._last_score = 0.0

    @property
    def is_active(self):
        return self._active and self._tracker is not None

    def reset(self):
        self._tracker = None
        self._active = False
        self._last_score = 0.0

    def init_from_detection(self, frame, det):
        """Initialize tracker from detector output."""
        tracker = _create_csrt_tracker()
        w = max(1, det.x2 - det.x1)
        h = max(1, det.y2 - det.y1)
        ok = tracker.init(frame, (det.x1, det.y1, w, h))
        if not ok:
            self.reset()
            return False
        self._tracker = tracker
        self._active = True
        self._last_score = float(det.score)
        return True

    def update(self, frame):
        """Update tracker and return tracked detection or failure."""
        if not self.is_active:
            return TrackerUpdate(detection=None, ok=False)

        ok, bbox = self._tracker.update(frame)
        if not ok:
            self.reset()
            return TrackerUpdate(detection=None, ok=False)

        x, y, w, h = bbox
        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(max(x1 + 1, x + w))
        y2 = int(max(y1 + 1, y + h))
        det = Detection(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=self._last_score,
            cls_id=0,
        )
        return TrackerUpdate(detection=det, ok=True)
