"""
OMDetector - Ascend NPU .om model inference via ACL Python API
Compatible with Atlas 200I DK A2 + CANN 7.x

Requires CANN toolkit installed and environment sourced:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Host (ARM CPU)          │  Device (Ascend NPU)      │
  │                          │                           │
  │  preprocess (OpenCV)     │                           │
  │  host_buf → copy ──────►│  device_in_buf            │
  │                          │  acl.mdl.execute()        │
  │  host_out ◄──── copy ───│  device_out_buf           │
  │  postprocess (numpy)     │                           │
  └─────────────────────────────────────────────────────┘
"""

import os
import logging
import time
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── ACL availability guard ────────────────────────────────────────────────────
try:
    import acl  # provided by CANN toolkit
    ACL_AVAILABLE = True
    logger.info("ACL Python binding loaded (Ascend CANN)")
except ImportError:
    ACL_AVAILABLE = False
    logger.warning("acl module not found – OMDetector requires CANN toolkit")

# ACL return-code constant
ACL_SUCCESS = 0
# Memory flags
ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _check(ret: int, msg: str):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"ACL error [{ret}] in {msg}")


class OMDetector:
    """
    YOLO .om model detector running on Ascend NPU via ACL.

    Supports the same public interface as YOLODetector:
      - detect(frame) → List[Dict]
      - classify_top_n(frame, top_n) → List[Dict]
      - .num_classes, .input_shape, .class_names

    Output format auto-detection:
      The YOLO output dimension is inferred from the model description.
      - v8/v11: [4+C, N] transposed  → fmt = 'v8'
      - v5/v6/v7: [N, 5+C]          → fmt = 'v5'
    """

    DEFAULT_CLASSES = [f"class_{i}" for i in range(80)]

    def __init__(
        self,
        model_path: str,
        label_path: Optional[str] = None,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device_id: int = 0,
        input_size: Optional[Tuple[int, int]] = None,  # (w, h)
    ):
        if not ACL_AVAILABLE:
            raise RuntimeError(
                "acl module not available. "
                "Source CANN env: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
            )

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device_id = device_id

        # Class labels
        self.class_names = self._load_labels(label_path)
        self.num_classes = len(self.class_names)

        # ACL context / stream
        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None

        # Buffer bookkeeping
        self._in_bufs: List[Tuple] = []   # list of (device_ptr, size)
        self._out_bufs: List[Tuple] = []  # list of (device_ptr, size)
        self._in_dataset = None
        self._out_dataset = None

        # Init ACL and load model
        self._init_acl()
        self._load_model()
        self._setup_io_buffers()
        self._detect_io_shapes(input_size)

        logger.info(
            f"OMDetector ready: {os.path.basename(model_path)} "
            f"input={self.input_shape} classes={self.num_classes} fmt={self.fmt}"
        )

    # ── Init / teardown ───────────────────────────────────────────────────────

    def _init_acl(self):
        ret = acl.init()
        _check(ret, "acl.init")

        ret = acl.rt.set_device(self.device_id)
        _check(ret, "acl.rt.set_device")

        self.context, ret = acl.rt.create_context(self.device_id)
        _check(ret, "acl.rt.create_context")

        self.stream, ret = acl.rt.create_stream()
        _check(ret, "acl.rt.create_stream")

        logger.info(f"ACL initialised on device {self.device_id}")

    def _load_model(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        _check(ret, f"acl.mdl.load_from_file({self.model_path})")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        _check(ret, "acl.mdl.get_desc")

        logger.info(f"OM model loaded: {self.model_path}")

    def _setup_io_buffers(self):
        """Allocate device memory for every input / output tensor."""
        num_inputs = acl.mdl.get_num_inputs(self.model_desc)
        num_outputs = acl.mdl.get_num_outputs(self.model_desc)

        self._in_dataset = acl.mdl.create_dataset()
        for i in range(num_inputs):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            _check(ret, f"malloc input[{i}]")
            data_buf = acl.create_data_buffer(dev_ptr, size)
            acl.mdl.add_dataset_buffer(self._in_dataset, data_buf)
            self._in_bufs.append((dev_ptr, size))

        self._out_dataset = acl.mdl.create_dataset()
        for i in range(num_outputs):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            _check(ret, f"malloc output[{i}]")
            data_buf = acl.create_data_buffer(dev_ptr, size)
            acl.mdl.add_dataset_buffer(self._out_dataset, data_buf)
            self._out_bufs.append((dev_ptr, size))

    def _detect_io_shapes(self, input_size_override):
        """Read shapes from model desc and decide YOLO format."""
        # ── input shape ──────────────────────────────────────────────────────
        dims_in, ret = acl.mdl.get_input_dims(self.model_desc, 0)
        # dims_in: {'dimCount': 4, 'dims': [1, 3, H, W]}
        dims = dims_in.get("dims", [1, 3, 640, 640])
        if input_size_override:
            self.input_w, self.input_h = input_size_override
        else:
            self.input_h = dims[2] if len(dims) > 2 and dims[2] > 0 else 640
            self.input_w = dims[3] if len(dims) > 3 and dims[3] > 0 else 640
        self.input_shape = (self.input_w, self.input_h)

        # ── output shape → format detection ──────────────────────────────────
        num_outputs = acl.mdl.get_num_outputs(self.model_desc)
        self._out_dims = []
        for i in range(num_outputs):
            dims_out, _ = acl.mdl.get_output_dims(self.model_desc, i)
            self._out_dims.append(dims_out.get("dims", []))

        # YOLO v8/v11: first output [1, 4+C, N]   → dim[1] > dim[2]
        # YOLO v5/v6/v7: first output [1, N, 5+C] → dim[2] > dim[1]
        d = self._out_dims[0] if self._out_dims else []
        if len(d) >= 3 and isinstance(d[1], int) and isinstance(d[2], int):
            self.fmt = "v8" if d[1] > d[2] else "v5"
        else:
            self.fmt = "v8"  # safe default for modern YOLOs

        logger.info(f"Output dims: {self._out_dims}  →  format={self.fmt}")

    def __del__(self):
        """Release ACL resources."""
        try:
            self._release()
        except Exception:
            pass

    def _release(self):
        for dev_ptr, _ in self._in_bufs + self._out_bufs:
            acl.rt.free(dev_ptr)
        if self._in_dataset:
            acl.mdl.destroy_dataset(self._in_dataset)
        if self._out_dataset:
            acl.mdl.destroy_dataset(self._out_dataset)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

    # ── Label loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_labels(path: Optional[str]) -> List[str]:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                names = [l.strip() for l in f if l.strip()]
            logger.info(f"Loaded {len(names)} labels from {path}")
            return names
        logger.warning("No label file – using generic class names")
        return OMDetector.DEFAULT_CLASSES

    # ── Pre / post processing ─────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray):
        img, ratio, (dw, dh) = _letterbox(frame, (self.input_w, self.input_h))
        img = img[:, :, ::-1].transpose(2, 0, 1)            # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[np.newaxis]                                # [1,3,H,W]
        return img, ratio, dw, dh

    def _run_inference(self, blob: np.ndarray) -> List[np.ndarray]:
        """Copy blob → device, execute model, copy outputs → host."""
        blob_bytes = blob.tobytes()
        dev_ptr, size = self._in_bufs[0]
        ret = acl.rt.memcpy(dev_ptr, size, blob_bytes, len(blob_bytes),
                             ACL_MEMCPY_HOST_TO_DEVICE)
        _check(ret, "memcpy H→D")

        ret = acl.mdl.execute(self.model_id, self._in_dataset, self._out_dataset)
        _check(ret, "acl.mdl.execute")

        results = []
        for i, (dev_ptr, size) in enumerate(self._out_bufs):
            host_buf = bytearray(size)
            ret = acl.rt.memcpy(host_buf, size, dev_ptr, size,
                                 ACL_MEMCPY_DEVICE_TO_HOST)
            _check(ret, f"memcpy D→H output[{i}]")

            # Reconstruct numpy array from dims
            dims = self._out_dims[i] if i < len(self._out_dims) else []
            dtype = np.float32
            arr = np.frombuffer(bytes(host_buf), dtype=dtype)
            if dims and np.prod([d for d in dims if d > 0]) == arr.size:
                arr = arr.reshape([d if d > 0 else 1 for d in dims])
            results.append(arr)

        return results

    def _postprocess(self, outputs, orig_h, orig_w, ratio, dw, dh):
        out = outputs[0]  # [1, 4+C, N] or [1, N, 5+C]
        if self.fmt == "v8":
            preds = out[0].T   # [N, 4+C]
            boxes, scores, class_ids = [], [], []
            for pred in preds:
                cls_scores = pred[4:]
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
        else:
            preds = out[0]     # [N, 5+C]
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
                x1 = max(0, min((cx - bw / 2 - dw) / ratio, orig_w))
                y1 = max(0, min((cy - bh / 2 - dh) / ratio, orig_h))
                x2 = max(0, min((cx + bw / 2 - dw) / ratio, orig_w))
                y2 = max(0, min((cy + bh / 2 - dh) / ratio, orig_h))
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(score)
                class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold,
                                    self.iou_threshold)
        if indices is None or len(indices) == 0:
            return []

        detections = []
        for i in (indices.flatten() if hasattr(indices, "flatten") else indices):
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
                "backend": "ascend_npu",
            })
        return detections

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict]:
        orig_h, orig_w = frame.shape[:2]
        blob, ratio, dw, dh = self._preprocess(frame)
        outputs = self._run_inference(blob)
        return self._postprocess(outputs, orig_h, orig_w, ratio, dw, dh)

    def classify_top_n(self, frame: np.ndarray, top_n: int = 5) -> List[Dict]:
        detections = self.detect(frame)
        class_conf: Dict[str, float] = {}
        for d in detections:
            n = d["class_name"]
            class_conf[n] = max(class_conf.get(n, 0.0), d["confidence"])
        sorted_c = sorted(class_conf.items(), key=lambda x: -x[1])
        return [{"class_name": k, "confidence": round(v, 3)}
                for k, v in sorted_c[:top_n]]


# ── Shared helper ─────────────────────────────────────────────────────────────

def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw = (new_w - nw) / 2
    dh = (new_h - nh) / 2
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)
