import cv2


def _blend_box_fill(frame, x1, y1, x2, y2, color, alpha):
    """Blend a solid color into ROI with given alpha."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _box_style(source):
    if source in ("track", "track_hold"):
        return (0, 220, 255), "track"
    return (40, 220, 40), "joystick"


def draw_detections(frame, detections, source="detect", alpha=0.2):
    color, prefix = _box_style(source)
    h, w = frame.shape[:2]
    for det in detections:
        x1 = max(0, min(w - 1, int(det.x1)))
        y1 = max(0, min(h - 1, int(det.y1)))
        x2 = max(x1 + 1, min(w, int(det.x2)))
        y2 = max(y1 + 1, min(h, int(det.y2)))

        _blend_box_fill(frame, x1, y1, x2, y2, color, alpha)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, 2)

        label = "{} {:.2f}".format(prefix, det.score)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        "FPS {:.1f}".format(fps),
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
    )


def draw_status(frame, text, y=54, color=(255, 220, 80)):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
