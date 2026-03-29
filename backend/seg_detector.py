"""
SegDetector - YOLO Instance Segmentation (yolo11n-seg / yolov8n-seg)
Supports ONNX and .om via the unified model factory.

Output layout (YOLOv8/v11-seg):
  output0: [1, 4 + num_classes + 32, N]   ← box + class scores + mask coefs
  output1: [1, 32, mask_H, mask_W]         ← prototype masks

Pipeline:
  1. Parse detections from output0 (same as regular YOLO v8)
  2. For each kept detection, grab its 32-dim mask coef vector
  3. mask = sigmoid(coefs @ proto.reshape(32, -1)).reshape(mask_H, mask_W)
  4. Crop mask to predicted box, resize to original image size

Required files in models/:
  yolo11n-seg.onnx      (exported from Ultralytics with task=segment)
  yolo11n-seg.txt       (class labels, one per line)

Export command (on a machine with Ultralytics):
  yolo export model=yolo11n-seg.pt format=onnx imgsz=640 opset=17

To convert to .om for NPU:
  atc --model=yolo11n-seg.onnx --framework=5 \
      --output=yolo11n-seg --soc_version=Ascend310B4 \
      --input_format=NCHW --input_shape="images:1,3,640,640" \
      --out_nodes="output0;output1"
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

# Sigmoid helper
_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

# COCO 17-keypoint skeleton (used in pose, kept here for reuse)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (11, 12), (5, 11), (6, 12),                    # torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # legs
]


class SegDetector:
    """
    YOLO-seg ONNX instance-segmentation detector.

    Returns detections with an extra 'mask' key (H×W uint8 binary mask,
    already cropped and resized to original frame resolution).

    Usage:
        det = SegDetector('models/yolo11n-seg.onnx', 'models/yolo11n-seg.txt')
        results = det.detect_seg(frame)
        for r in results:
            cv2.imshow('mask', r['mask'] * 255)
    """

    DEFAULT_CLASSES = [f"class_{i}" for i in range(80)]

    def __init__(
        self,
        model_path: str,
        label_path: Optional[str] = None,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        input_size: Optional[Tuple[int, int]] = None,
        providers: Optional[List[str]] = None,
    ):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self._load_labels(label_path)
        self.num_classes = len(self.class_names)

        # Provider selection (NPU → CPU)
        if providers is None:
            avail = ort.get_available_providers()
            providers = []
            if "CANNExecutionProvider" in avail:
                providers.append(("CANNExecutionProvider", {}))
                logger.info("SegDetector: using CANN NPU")
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

        # Validate two outputs
        out_names = [o.name for o in self.session.get_outputs()]
        out_shapes = [o.shape for o in self.session.get_outputs()]
        logger.info(f"SegDetector outputs: {list(zip(out_names, out_shapes))}")
        if len(out_names) < 2:
            raise ValueError(
                "Expected 2 outputs (detections + proto masks). "
                "Make sure you exported with task=segment."
            )
        self.output_names = out_names[:2]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_labels(path):
        if path and __import__("os").path.exists(path):
            with open(path, "r") as f:
                return [l.strip() for l in f if l.strip()]
        return SegDetector.DEFAULT_CLASSES

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

    # ── Core segmentation post-process ───────────────────────────────────────

    def _process_masks(
        self,
        mask_coefs: np.ndarray,   # [N_det, 32]
        proto: np.ndarray,         # [32, mask_H, mask_W]
        boxes_xyxy: np.ndarray,    # [N_det, 4]  in original image coords
        orig_h: int,
        orig_w: int,
    ) -> List[np.ndarray]:
        """
        Compute per-instance binary masks at original resolution.
        Returns list of uint8 arrays [orig_H, orig_W], values 0/1.
        """
        mask_h, mask_w = proto.shape[1], proto.shape[2]

        # [N, 32] @ [32, mask_H*mask_W] → [N, mask_H*mask_W]
        masks_flat = mask_coefs @ proto.reshape(32, -1)
        masks_flat = _sigmoid(masks_flat)                   # [N, mask_H*mask_W]
        masks = masks_flat.reshape(-1, mask_h, mask_w)     # [N, mask_H, mask_W]

        result_masks = []
        scale_x = orig_w / mask_w
        scale_y = orig_h / mask_h

        for i, (mask, box) in enumerate(zip(masks, boxes_xyxy)):
            # Resize mask to original resolution
            full_mask = cv2.resize(mask, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LINEAR)

            # Crop to bounding box
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            crop_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            crop_mask[y1:y2, x1:x2] = (full_mask[y1:y2, x1:x2] > 0.5).astype(np.uint8)

            result_masks.append(crop_mask)

        return result_masks

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_seg(self, frame: np.ndarray) -> List[Dict]:
        """
        Run segmentation inference.

        Returns list of dicts with keys:
          class_id, class_name, confidence, bbox, bbox_xywh, mask
        where mask is a uint8 np.ndarray [H, W] (0=background, 1=object).
        """
        orig_h, orig_w = frame.shape[:2]
        blob, ratio, dw, dh = self._preprocess(frame)

        outputs = self.session.run(
            self.output_names, {self.input_name: blob}
        )
        det_out = outputs[0]   # [1, 4+nc+32, N]
        proto   = outputs[1]   # [1, 32, mH, mW]

        preds = det_out[0].T   # [N, 4+nc+32]
        proto = proto[0]       # [32, mH, mW]

        nc = self.num_classes
        boxes, scores, class_ids, mask_coefs_list = [], [], [], []

        for pred in preds:
            cls_scores = pred[4:4 + nc]
            cls_id = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_id])
            if score < self.conf_threshold:
                continue
            cx, cy, bw, bh = pred[:4]
            x1 = max(0, min((cx - bw / 2 - dw) / ratio, orig_w))
            y1 = max(0, min((cy - bh / 2 - dh) / ratio, orig_h))
            x2 = max(0, min((cx + bw / 2 - dw) / ratio, orig_w))
            y2 = max(0, min((cy + bh / 2 - dh) / ratio, orig_h))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(cls_id)
            mask_coefs_list.append(pred[4 + nc:4 + nc + 32])   # 32 coefs

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores,
                                    self.conf_threshold, self.iou_threshold)
        if indices is None or len(indices) == 0:
            return []

        flat = indices.flatten() if hasattr(indices, "flatten") else indices

        kept_boxes_xyxy = np.array(
            [[boxes[i][0], boxes[i][1],
              boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]]
             for i in flat], dtype=np.float32
        )
        kept_coefs = np.array([mask_coefs_list[i] for i in flat], dtype=np.float32)

        masks = self._process_masks(kept_coefs, proto, kept_boxes_xyxy,
                                     orig_h, orig_w)

        detections = []
        for idx, i in enumerate(flat):
            x, y, bw, bh = boxes[i]
            cls_id = class_ids[i]
            name = (self.class_names[cls_id]
                    if cls_id < len(self.class_names) else f"cls_{cls_id}")
            detections.append({
                "class_id": int(cls_id),
                "class_name": name,
                "confidence": round(float(scores[i]), 3),
                "bbox": [round(x), round(y), round(x + bw), round(y + bh)],
                "bbox_xywh": [round(x), round(y), round(bw), round(bh)],
                "mask": masks[idx],
            })

        return detections

    # Alias for compatibility with the model factory
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """detect() strips the mask to keep API compatible with app.py."""
        results = self.detect_seg(frame)
        for r in results:
            r.pop("mask", None)
        return results


def draw_seg(frame: np.ndarray, detections: List[Dict],
             palette=None, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay instance-segmentation masks + bounding boxes on frame.
    Call this instead of (or after) draw_detections() when mode=='segmentation'.

    detections must include a 'mask' key (output of SegDetector.detect_seg).
    """
    if palette is None:
        palette = [
            (52, 211, 153), (251, 113, 133), (96, 165, 250), (250, 204, 21),
            (167, 139, 250), (34, 211, 238), (249, 115, 22), (132, 204, 22),
        ]
    overlay = frame.copy()

    for det in detections:
        color = palette[det.get("class_id", 0) % len(palette)]
        mask = det.get("mask")

        if mask is not None:
            overlay[mask == 1] = color

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
