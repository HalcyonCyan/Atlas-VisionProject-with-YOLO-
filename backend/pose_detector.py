"""
PoseDetector - YOLO Human Pose Estimation (yolo11n-pose / yolov8n-pose)

Output layout (YOLOv8/v11-pose):
  output0: [1, 4 + 1 + 17*3, N]
             │   │   └─ 17 keypoints × (x, y, visibility)
             │   └───── objectness / box confidence
             └───────── cx, cy, w, h

COCO 17 Keypoints (index → name):
  0  nose          1  left_eye      2  right_eye
  3  left_ear      4  right_ear     5  left_shoulder
  6  right_shoulder 7 left_elbow   8  right_elbow
  9  left_wrist   10  right_wrist  11  left_hip
  12 right_hip    13 left_knee     14 right_knee
  15 left_ankle   16 right_ankle

Required files in models/:
  yolo11n-pose.onnx     (exported from Ultralytics with task=pose)
  (no label file needed — poses are person-only)

Export command:
  yolo export model=yolo11n-pose.pt format=onnx imgsz=640 opset=17

To convert to .om:
  atc --model=yolo11n-pose.onnx --framework=5 \
      --output=yolo11n-pose --soc_version=Ascend310B4 \
      --input_format=NCHW --input_shape="images:1,3,640,640"
"""

import logging
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

logger = logging.getLogger(__name__)

# COCO skeleton connections (keypoint index pairs)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head → ears
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12),                        # torso sides
    (11, 12),                                # hips
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Color per body part group
_KP_COLORS = [
    (255, 128,   0),  # head group (0-4)
    (255, 128,   0),
    (255, 128,   0),
    (255, 128,   0),
    (255, 128,   0),
    ( 51, 153, 255),  # shoulders + arms (5-10)
    ( 51, 153, 255),
    ( 51, 153, 255),
    ( 51, 153, 255),
    ( 51, 153, 255),
    ( 51, 153, 255),
    (128, 255,   0),  # hips + legs (11-16)
    (128, 255,   0),
    (128, 255,   0),
    (128, 255,   0),
    (128, 255,   0),
    (128, 255,   0),
]

_LIMB_COLORS = [
    (255, 128,   0),  # head
    (255, 128,   0),
    (255, 128,   0),
    (255, 128,   0),
    ( 51, 153, 255),
    ( 51, 153, 255),
    ( 51, 153, 255),
    ( 51, 153, 255),
    ( 51, 153, 255),
    (128, 255,   0),
    (128, 255,   0),
    (128, 255,   0),
    (128, 255,   0),
    (128, 255,   0),
    (128, 255,   0),
    (128, 255,   0),
]


class PoseDetector:
    """
    YOLO-pose ONNX pose-estimation detector.

    detect_pose(frame) → List[Dict] where each dict has:
      class_id, class_name, confidence, bbox, bbox_xywh,
      keypoints: List[Dict{name, x, y, visibility}]
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        kp_conf_threshold: float = 0.5,     # min visibility to draw a keypoint
        input_size: Optional[Tuple[int, int]] = None,
        providers: Optional[List[str]] = None,
    ):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.kp_conf_threshold = kp_conf_threshold
        self.class_names = ["person"]
        self.num_classes = 1

        if providers is None:
            avail = ort.get_available_providers()
            providers = []
            if "CANNExecutionProvider" in avail:
                providers.append(("CANNExecutionProvider", {}))
                logger.info("PoseDetector: using CANN NPU")
            providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(model_path, opts, providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if input_size:
            self.input_w, self.input_h = input_size
        else:
            self.input_w = shape[3] if isinstance(shape[3], int) else 640
            self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_shape = (self.input_w, self.input_h)

        out_shapes = [o.shape for o in self.session.get_outputs()]
        logger.info(f"PoseDetector input={self.input_shape} outputs={out_shapes}")

        # Verify output has 4+1+51=56 or 4+51=55 channels
        d = out_shapes[0]
        if len(d) >= 2:
            ch = d[1] if isinstance(d[1], int) else 56
            self._n_kp = (ch - 5) // 3 if ch >= 56 else 17
        else:
            self._n_kp = 17
        logger.info(f"PoseDetector: {self._n_kp} keypoints per person")

    # ── Pre-processing ────────────────────────────────────────────────────────

    @staticmethod
    def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(new_shape[0] / w, new_shape[1] / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw = (new_shape[0] - nw) / 2
        dh = (new_shape[1] - nh) / 2
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def _preprocess(self, frame):
        img, ratio, (dw, dh) = self._letterbox(frame, (self.input_w, self.input_h))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[np.newaxis], ratio, dw, dh

    # ── Post-processing ───────────────────────────────────────────────────────

    def _decode_kps(self, raw_kps, ratio, dw, dh, orig_w, orig_h):
        """
        raw_kps: [n_kp * 3]  (x, y, vis) × n_kp — in network-input coordinates
        Returns list of {name, x, y, visibility}
        """
        kps = []
        for i in range(self._n_kp):
            kx_net = raw_kps[i * 3 + 0]
            ky_net = raw_kps[i * 3 + 1]
            vis    = float(raw_kps[i * 3 + 2])
            # Map back to original image coordinates
            kx = max(0, min((kx_net - dw) / ratio, orig_w))
            ky = max(0, min((ky_net - dh) / ratio, orig_h))
            name = KP_NAMES[i] if i < len(KP_NAMES) else f"kp_{i}"
            kps.append({"name": name, "x": round(kx, 1),
                        "y": round(ky, 1), "visibility": round(vis, 3)})
        return kps

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_pose(self, frame: np.ndarray) -> List[Dict]:
        """
        Run pose estimation.

        Returns list of dicts with keys:
          class_id (always 0), class_name ('person'), confidence,
          bbox [x1,y1,x2,y2], bbox_xywh,
          keypoints: list of {name, x, y, visibility}
        """
        orig_h, orig_w = frame.shape[:2]
        blob, ratio, dw, dh = self._preprocess(frame)

        outputs = self.session.run(None, {self.input_name: blob})
        pred = outputs[0][0].T   # [N, 4+1+kp*3]  (YOLOv8/v11 layout)

        boxes, scores, kp_raws = [], [], []
        for row in pred:
            conf = float(row[4])
            if conf < self.conf_threshold:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = max(0.0, (cx - bw / 2 - dw) / ratio)
            y1 = max(0.0, (cy - bh / 2 - dh) / ratio)
            x2 = min(float(orig_w), (cx + bw / 2 - dw) / ratio)
            y2 = min(float(orig_h), (cy + bh / 2 - dh) / ratio)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            kp_raws.append(row[5:5 + self._n_kp * 3])

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores,
                                    self.conf_threshold, self.iou_threshold)
        if indices is None or len(indices) == 0:
            return []

        flat = indices.flatten() if hasattr(indices, "flatten") else indices
        detections = []
        for i in flat:
            x, y, bw, bh = boxes[i]
            kps = self._decode_kps(kp_raws[i], ratio, dw, dh, orig_w, orig_h)
            detections.append({
                "class_id": 0,
                "class_name": "person",
                "confidence": round(float(scores[i]), 3),
                "bbox": [round(x), round(y), round(x + bw), round(y + bh)],
                "bbox_xywh": [round(x), round(y), round(bw), round(bh)],
                "keypoints": kps,
            })
        return detections

    # Alias for model factory compatibility
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """detect() strips keypoints to stay compatible with draw_detections()."""
        results = self.detect_pose(frame)
        for r in results:
            r.pop("keypoints", None)
        return results


# ── Drawing helper ─────────────────────────────────────────────────────────────

def draw_pose(frame: np.ndarray, detections: List[Dict],
              kp_conf_threshold: float = 0.5) -> np.ndarray:
    """
    Draw keypoints and skeleton limbs on frame.
    detections must include 'keypoints' key (output of PoseDetector.detect_pose).
    """
    for det in detections:
        kps = det.get("keypoints", [])
        if not kps:
            continue

        # Draw bounding box
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'person {det["confidence"]:.2f}'
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        # Build coordinate array for quick lookup
        pts = [(int(kp["x"]), int(kp["y"]), kp["visibility"]) for kp in kps]

        # Draw skeleton limbs
        for idx, (a, b) in enumerate(SKELETON):
            if a >= len(pts) or b >= len(pts):
                continue
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va < kp_conf_threshold or vb < kp_conf_threshold:
                continue
            color = _LIMB_COLORS[idx % len(_LIMB_COLORS)]
            cv2.line(frame, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)

        # Draw keypoint dots
        for idx, (px, py, vis) in enumerate(pts):
            if vis < kp_conf_threshold:
                continue
            color = _KP_COLORS[idx % len(_KP_COLORS)]
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.circle(frame, (px, py), 4, (255, 255, 255), 1)

    return frame
