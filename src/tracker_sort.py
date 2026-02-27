import numpy as np

from .postprocess import Detection


def sort_available():
    """Return True if required runtime deps for SORT tracker are present."""
    return True


class TrackerUpdate(object):
    def __init__(self, detection=None, ok=False):
        self.detection = detection
        self.ok = bool(ok)


class SortTracker(object):
    """Single-object lightweight SORT tracker with a constant-velocity Kalman model."""

    def __init__(self, max_age=30):
        self._max_age = int(max(1, max_age))
        self._state = None  # [cx, cy, w, h, vx, vy, vw, vh]^T
        self._cov = None
        self._misses = 0
        self._last_score = 0.0
        self._active = False

        self._F = np.eye(8, dtype=np.float32)
        self._F[0, 4] = 1.0
        self._F[1, 5] = 1.0
        self._F[2, 6] = 1.0
        self._F[3, 7] = 1.0

        self._H = np.zeros((4, 8), dtype=np.float32)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0

        # Modest process noise for stable bbox smoothing.
        self._Q = np.diag([1.0, 1.0, 1.5, 1.5, 12.0, 12.0, 18.0, 18.0]).astype(
            np.float32
        )
        # Measurement noise tuned for detector jitter.
        self._R = np.diag([16.0, 16.0, 25.0, 25.0]).astype(np.float32)

    @property
    def is_active(self):
        return self._active and self._state is not None and self._cov is not None

    def reset(self):
        self._state = None
        self._cov = None
        self._misses = 0
        self._last_score = 0.0
        self._active = False

    def _det_to_measurement(self, det):
        x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return np.array([[cx], [cy], [w], [h]], dtype=np.float32)

    def _to_detection(self, score, frame_shape=None):
        cx = float(self._state[0, 0])
        cy = float(self._state[1, 0])
        w = max(2.0, float(self._state[2, 0]))
        h = max(2.0, float(self._state[3, 0]))

        x1 = int(round(cx - 0.5 * w))
        y1 = int(round(cy - 0.5 * h))
        x2 = int(round(cx + 0.5 * w))
        y2 = int(round(cy + 0.5 * h))

        if frame_shape is not None:
            fh, fw = frame_shape[:2]
            x1 = max(0, min(fw - 1, x1))
            y1 = max(0, min(fh - 1, y1))
            x2 = max(x1 + 1, min(fw, x2))
            y2 = max(y1 + 1, min(fh, y2))

        return Detection(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=float(score),
            cls_id=0,
        )

    def _predict(self):
        self._state = np.matmul(self._F, self._state)
        self._cov = np.matmul(np.matmul(self._F, self._cov), self._F.T) + self._Q

    def _correct(self, z):
        innovation = z - np.matmul(self._H, self._state)
        s = np.matmul(np.matmul(self._H, self._cov), self._H.T) + self._R
        k = np.matmul(np.matmul(self._cov, self._H.T), np.linalg.inv(s))
        self._state = self._state + np.matmul(k, innovation)
        i = np.eye(self._cov.shape[0], dtype=np.float32)
        self._cov = np.matmul(i - np.matmul(k, self._H), self._cov)
        # Keep size terms positive.
        self._state[2, 0] = max(2.0, float(self._state[2, 0]))
        self._state[3, 0] = max(2.0, float(self._state[3, 0]))

    def init_from_detection(self, frame, det):
        """Initialize or correct tracker from detector output."""
        z = self._det_to_measurement(det)

        if not self.is_active:
            self._state = np.zeros((8, 1), dtype=np.float32)
            self._state[:4, :] = z
            self._cov = np.diag(
                [25.0, 25.0, 36.0, 36.0, 250.0, 250.0, 300.0, 300.0]
            ).astype(np.float32)
        else:
            # Move state to current frame, then fuse detector measurement.
            self._predict()
            self._correct(z)

        self._misses = 0
        self._active = True
        self._last_score = float(det.score)
        return True

    def update(self, frame):
        """Predict next bbox when detector is skipped."""
        if not self.is_active:
            return TrackerUpdate(detection=None, ok=False)

        self._predict()
        self._misses += 1

        if self._misses > self._max_age:
            self.reset()
            return TrackerUpdate(detection=None, ok=False)

        # Decay confidence slightly while running open-loop prediction.
        self._last_score = max(0.05, self._last_score * 0.995)
        det = self._to_detection(self._last_score, frame_shape=frame.shape if frame is not None else None)
        return TrackerUpdate(detection=det, ok=True)
