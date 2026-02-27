import statistics
import time


def _percentile(values, p):
    """Return percentile from a list using simple index selection."""
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(len(values) * p) - 1))
    return sorted(values)[idx]


class RuntimeMetrics(object):
    """Collect frame-level latency and detection counters."""

    def __init__(self):
        self._infer_latencies_ms = []  # type: List[float]
        self.frames = 0
        self.det_frames = 0
        self._t0 = None  # type: Optional[float]

    def start(self):
        """Mark benchmark start time."""
        self._t0 = time.time()

    def add_latency_ms(self, value_ms):
        """Append one inference latency sample."""
        self._infer_latencies_ms.append(float(value_ms))

    def add_frame(self, infer_ms, has_detection):
        """Record one processed frame and detection presence."""
        if infer_ms is not None:
            self.add_latency_ms(infer_ms)
        self.frames += 1
        if has_detection:
            self.det_frames += 1

    def elapsed_sec(self):
        """Return elapsed runtime in seconds."""
        if self._t0 is None:
            return 0.0
        return max(time.time() - self._t0, 1e-6)

    def fps_now(self):
        """Return current average FPS."""
        if self.frames <= 0:
            return 0.0
        return self.frames / self.elapsed_sec()

    def snapshot(self):
        """Return summary metrics snapshot."""
        elapsed = self.elapsed_sec()
        if self.frames <= 0:
            return {
                "frames": 0.0,
                "total_sec": round(elapsed, 3),
                "fps": 0.0,
                "infer_samples": 0.0,
                "infer_ms_mean": 0.0,
                "infer_ms_p50": 0.0,
                "infer_ms_p95": 0.0,
                "det_frames": 0.0,
                "det_ratio": 0.0,
            }

        if not self._infer_latencies_ms:
            infer_mean = 0.0
            infer_p50 = 0.0
            infer_p95 = 0.0
        else:
            infer_mean = round(
                sum(self._infer_latencies_ms) / len(self._infer_latencies_ms), 3
            )
            infer_p50 = round(statistics.median(self._infer_latencies_ms), 3)
            infer_p95 = round(_percentile(self._infer_latencies_ms, 0.95), 3)

        return {
            "frames": float(self.frames),
            "total_sec": round(elapsed, 3),
            "fps": round(self.frames / elapsed, 3),
            "infer_samples": float(len(self._infer_latencies_ms)),
            "infer_ms_mean": infer_mean,
            "infer_ms_p50": infer_p50,
            "infer_ms_p95": infer_p95,
            "det_frames": float(self.det_frames),
            "det_ratio": round(self.det_frames / self.frames, 4),
        }
