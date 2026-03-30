"""
YOLODetector - ONNX Runtime based YOLO detector
Supports: YOLOv5, YOLOv6, YOLOv7, YOLOv8 / YOLO11 output formats
"""
##

import os
import time
import logging
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logging.warning('onnxruntime not found, detector disabled')

logger = logging.getLogger(__name__)


class YOLODetector:
    """Universal YOLO ONNX detector.

    Auto-detects output format:
      - YOLOv5/v6/v7: [B, N, 5+C]  (cx,cy,w,h,obj,cls...)
      - YOLOv8/v11:   [B, 4+C, N]  (cx,cy,w,h,cls...) — no obj score
    """

    DEFAULT_CLASSES = [f'class_{i}' for i in range(80)]

    def __init__(
        self,
        model_path: str,
        label_path: Optional[str] = None,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        input_size: Optional[Tuple[int, int]] = None,   # (w, h)
        providers: Optional[List[str]] = None,
    ):
        if not ORT_AVAILABLE:
            raise RuntimeError('onnxruntime is not installed')

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load class names
        self.class_names = self._load_labels(label_path)
        self.num_classes = len(self.class_names)

        # Build provider list (NPU -> CUDA -> CPU)
        if providers is None:
            available = ort.get_available_providers()
            providers = []
            # Ascend NPU via CANN EP (if available)
            if 'CANNExecutionProvider' in available:
                providers.append(('CANNExecutionProvider', {}))
                logger.info('Using CANNExecutionProvider (Ascend NPU)')
            if 'CUDAExecutionProvider' in available:
                providers.append(('CUDAExecutionProvider', {}))
            providers.append('CPUExecutionProvider')

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        logger.info(f'Loading model: {model_path}')
        self.session = ort.InferenceSession(model_path, sess_opts,
                                             providers=providers)

        # Inspect model I/O
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape          # e.g. [1, 3, 640, 640] or [batch, 3, H, W]
        if input_size:
            self.input_w, self.input_h = input_size
        else:
            self.input_w = shape[3] if (len(shape) == 4 and isinstance(shape[3], int)) else 640
            self.input_h = shape[2] if (len(shape) == 4 and isinstance(shape[2], int)) else 640
        self.input_shape = (self.input_w, self.input_h)

        out_shapes = [o.shape for o in self.session.get_outputs()]
        logger.info(f'Input: {inp.name} {shape}  Outputs: {out_shapes}')
        self._detect_format(out_shapes)

    # ── Label loading ─────────────────────────────────────────────────────────
    @staticmethod
    def _load_labels(path: Optional[str]) -> List[str]:
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                names = [l.strip() for l in f if l.strip()]
            logger.info(f'Loaded {len(names)} class labels from {path}')
            return names
        logger.warning('No label file found, using generic class names')
        return YOLODetector.DEFAULT_CLASSES

    # ── Format detection ──────────────────────────────────────────────────────
    def _detect_format(self, out_shapes):
        """Heuristically decide YOLOv5 vs YOLOv8 output layout."""
        s = out_shapes[0]           # first output tensor shape
        if len(s) == 3:
            # [B, N, D] → v5/v6/v7 if D > N, else probably transposed
            if isinstance(s[2], int) and isinstance(s[1], int):
                if s[2] > s[1]:     # D >> N → v8/v11 transpose [B, D, N]
                    self.fmt = 'v8'
                else:
                    self.fmt = 'v5'
            else:
                self.fmt = 'v5'     # dynamic dims fallback
        else:
            self.fmt = 'v5'
        logger.info(f'Detected YOLO format: {self.fmt}')

    # ── Pre/post processing ───────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray):
        """Letterbox resize → float32 NCHW."""
        img, ratio, (dw, dh) = self._letterbox(frame, (self.input_w, self.input_h))
        img = img[:, :, ::-1].transpose(2, 0, 1)   # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[np.newaxis]                        # add batch dim
        return img, ratio, dw, dh

    @staticmethod
    def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        h, w = img.shape[:2]
        new_w, new_h = new_shape
        r = min(new_w / w, new_h / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw = (new_w - nw) / 2
        dh = (new_h - nh) / 2
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def _postprocess_v5(self, output, orig_h, orig_w, ratio, dw, dh):
        """Parse YOLOv5/v6/v7 output [1, N, 5+C]."""
        preds = output[0][0]        # [N, 5+C]
        boxes, scores, class_ids = [], [], []
        for pred in preds:
            obj_conf = float(pred[4])
            if obj_conf < self.conf_threshold:
                continue
            cls_scores = pred[5:]
            cls_id = int(np.argmax(cls_scores))
            score = obj_conf * float(cls_scores[cls_id])
            if score < self.conf_threshold:
                continue
            cx, cy, bw, bh = pred[:4]
            x1 = (cx - bw / 2 - dw) / ratio
            y1 = (cy - bh / 2 - dh) / ratio
            x2 = (cx + bw / 2 - dw) / ratio
            y2 = (cy + bh / 2 - dh) / ratio
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(cls_id)
        return boxes, scores, class_ids

    def _postprocess_v8(self, output, orig_h, orig_w, ratio, dw, dh):
        """Parse YOLOv8/v11 output [1, 4+C, N]."""
        preds = output[0][0].T      # [N, 4+C]
        boxes, scores, class_ids = [], [], []
        for pred in preds:
            cls_scores = pred[4:]
            cls_id = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_id])
            if score < self.conf_threshold:
                continue
            cx, cy, bw, bh = pred[:4]
            x1 = (cx - bw / 2 - dw) / ratio
            y1 = (cy - bh / 2 - dh) / ratio
            x2 = (cx + bw / 2 - dw) / ratio
            y2 = (cy + bh / 2 - dh) / ratio
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(cls_id)
        return boxes, scores, class_ids

    # ── Public detect API ─────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run inference on a BGR frame, return list of detections."""
        orig_h, orig_w = frame.shape[:2]
        blob, ratio, dw, dh = self._preprocess(frame)

        t0 = time.time()
        outputs = self.session.run(None, {self.input_name: blob})
        _ = time.time() - t0

        if self.fmt == 'v8':
            boxes, scores, class_ids = self._postprocess_v8(
                outputs, orig_h, orig_w, ratio, dw, dh)
        else:
            boxes, scores, class_ids = self._postprocess_v5(
                outputs, orig_h, orig_w, ratio, dw, dh)

        if not boxes:
            return []

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold)
        if indices is None or len(indices) == 0:
            return []

        detections = []
        for i in (indices.flatten() if hasattr(indices, 'flatten') else indices):
            x, y, bw, bh = boxes[i]
            cls_id = class_ids[i]
            name = (self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else f'cls_{cls_id}')
            detections.append({
                'class_id': int(cls_id),
                'class_name': name,
                'confidence': round(float(scores[i]), 3),
                'bbox': [round(x), round(y), round(x + bw), round(y + bh)],
                'bbox_xywh': [round(x), round(y), round(bw), round(bh)],
            })
        return detections

    def classify_top_n(self, frame: np.ndarray, top_n: int = 5) -> List[Dict]:
        """Return top-N class confidences (uses detection aggregate)."""
        detections = self.detect(frame)
        # Aggregate by class
        class_conf: Dict[str, float] = {}
        for d in detections:
            n = d['class_name']
            class_conf[n] = max(class_conf.get(n, 0.0), d['confidence'])
        sorted_c = sorted(class_conf.items(), key=lambda x: -x[1])
        return [{'class_name': k, 'confidence': round(v, 3)}
                for k, v in sorted_c[:top_n]]
