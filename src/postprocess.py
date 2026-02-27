import cv2
import numpy as np


class Detection(object):
    """Single postprocessed detection in image pixel coordinates."""

    def __init__(self, x1, y1, x2, y2, score, cls_id=0):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.score = float(score)
        self.cls_id = int(cls_id)


def decode_and_filter(
    raw_output,
    in_w,
    in_h,
    frame_w,
    frame_h,
    conf_thres=0.6,
    iou_thres=0.7,
    max_det=10,
):
    """Decode YOLO-like output and run NMS for one class."""
    pred = raw_output
    if pred is None:
        return []
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        return []
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes_xywh = []
    scores = []

    sx = frame_w / float(in_w)
    sy = frame_h / float(in_h)

    for row in pred:
        if row.shape[0] < 5:
            continue

        cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        obj = float(row[4])

        if row.shape[0] > 5:
            cls_prob = float(np.max(row[5:]))
            score = obj * cls_prob
        else:
            score = obj

        if score < conf_thres:
            continue

        x = (cx - w / 2.0) * sx
        y = (cy - h / 2.0) * sy
        bw = w * sx
        bh = h * sy

        x = max(0.0, min(x, frame_w - 1.0))
        y = max(0.0, min(y, frame_h - 1.0))
        bw = max(1.0, min(bw, frame_w - x))
        bh = max(1.0, min(bh, frame_h - y))

        boxes_xywh.append([int(x), int(y), int(bw), int(bh)])
        scores.append(score)

    if not boxes_xywh:
        return []

    nms = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres)
    if nms is None or len(nms) == 0:
        return []

    detections = []
    for idx in np.array(nms).reshape(-1)[:max_det]:
        x, y, bw, bh = boxes_xywh[int(idx)]
        detections.append(
            Detection(
                x1=x,
                y1=y,
                x2=x + bw,
                y2=y + bh,
                score=float(scores[int(idx)]),
                cls_id=0,
            )
        )
    return detections
