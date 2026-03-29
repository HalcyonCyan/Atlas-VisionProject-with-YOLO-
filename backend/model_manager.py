"""
ModelManager - Scans models/ directory and instantiates the right detector.

Supported file types:
  .onnx   → YOLODetector (CPU / CANN NPU via ORT CANNExecutionProvider)
  .om     → OMDetector   (Ascend NPU via native ACL)

Model type auto-detection by filename keywords:
  *seg*   → SegDetector   (instance segmentation)
  *pose*  → PoseDetector  (keypoint estimation)
  default → YOLODetector  (object detection)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Lazy imports so the server still starts if some deps are missing
def _import_detectors():
    from detector import YOLODetector
    from om_detector import OMDetector, ACL_AVAILABLE
    from seg_detector import SegDetector
    from pose_detector import PoseDetector
    return YOLODetector, OMDetector, ACL_AVAILABLE, SegDetector, PoseDetector


class ModelInfo:
    """Lightweight descriptor for a discovered model file."""

    SUPPORTED_EXTENSIONS = {".onnx", ".om"}

    def __init__(self, path: Path, models_dir: Path):
        self.path = path
        self.name = path.name
        self.stem = path.stem
        self.ext = path.suffix.lower()
        self.size_mb = round(path.stat().st_size / 1024 / 1024, 1)

        # Detect task from filename
        stem_lower = self.stem.lower()
        if "seg" in stem_lower:
            self.task = "segmentation"
        elif "pose" in stem_lower:
            self.task = "pose"
        elif "cls" in stem_lower or "class" in stem_lower:
            self.task = "classification"
        else:
            self.task = "detection"

        self.backend = "onnx" if self.ext == ".onnx" else "ascend_om"

        # Label file lookup
        label_candidates = [
            models_dir / (self.stem + ".txt"),
            models_dir / (self.stem + "_labels.txt"),
            models_dir / "labels.txt",
            models_dir / "label.txt",
        ]
        found = next((p for p in label_candidates if p.exists()), None)
        self.label_path: Optional[str] = str(found) if found else None
        self.has_labels = found is not None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "task": self.task,
            "backend": self.backend,
            "size_mb": self.size_mb,
            "has_labels": self.has_labels,
            "label_file": Path(self.label_path).name if self.label_path else None,
        }

    def __repr__(self):
        return (f"<ModelInfo {self.name} task={self.task} "
                f"backend={self.backend} {self.size_mb}MB>")


class ModelManager:
    """
    Scans models/ directory on startup, exposes available models,
    and instantiates the correct detector on demand.

    Usage:
        mgr = ModelManager(Path("../models"))
        mgr.scan()                          # call once at startup
        print(mgr.list_models())            # human-readable table
        detector = mgr.load("yolo11n.onnx", conf=0.45, iou=0.45)
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._catalog: List[ModelInfo] = []

    # ── Discovery ─────────────────────────────────────────────────────────────

    def scan(self) -> List[ModelInfo]:
        """
        Scan models_dir for supported model files.
        Called once at server startup.  Safe to call again to refresh.
        """
        self._catalog = []
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return self._catalog

        for ext in ModelInfo.SUPPORTED_EXTENSIONS:
            for p in sorted(self.models_dir.glob(f"*{ext}")):
                try:
                    info = ModelInfo(p, self.models_dir)
                    self._catalog.append(info)
                    logger.info(f"  Found model: {info}")
                except Exception as e:
                    logger.warning(f"  Skipping {p.name}: {e}")

        logger.info(f"Model scan complete: {len(self._catalog)} model(s) found")
        return self._catalog

    def list_models(self) -> List[Dict]:
        """Return serialisable list for /api/models endpoint."""
        return [m.to_dict() for m in self._catalog]

    def get_info(self, name: str) -> Optional[ModelInfo]:
        for m in self._catalog:
            if m.name == name:
                return m
        return None

    def print_table(self):
        """Pretty-print model table to stdout at startup."""
        if not self._catalog:
            print("  [ModelManager] No models found in", self.models_dir)
            return
        print(f"\n{'─'*60}")
        print(f"  {'#':<3} {'Name':<30} {'Task':<14} {'Backend':<12} {'MB':>5}")
        print(f"{'─'*60}")
        for i, m in enumerate(self._catalog):
            print(f"  {i:<3} {m.name:<30} {m.task:<14} {m.backend:<12} {m.size_mb:>5}")
        print(f"{'─'*60}\n")

    # ── Instantiation ─────────────────────────────────────────────────────────

    def load(
        self,
        name: str,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
    ) -> Any:
        """
        Instantiate and return the appropriate detector for the given model name.

        Returns one of: YOLODetector | OMDetector | SegDetector | PoseDetector
        All share the .detect(frame) → List[Dict] interface.
        """
        info = self.get_info(name)
        if info is None:
            # Try direct path
            p = self.models_dir / name
            if p.exists():
                info = ModelInfo(p, self.models_dir)
            else:
                raise FileNotFoundError(f"Model not found: {name}")

        YOLODetector, OMDetector, ACL_AVAILABLE, SegDetector, PoseDetector = (
            _import_detectors()
        )

        model_path = str(info.path)
        label_path = info.label_path

        logger.info(f"Loading {info.name}  task={info.task}  backend={info.backend}")

        # ── .om → Ascend NPU ──────────────────────────────────────────────────
        if info.ext == ".om":
            if not ACL_AVAILABLE:
                raise RuntimeError(
                    ".om model requires CANN ACL. "
                    "Source: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
                )
            det = OMDetector(
                model_path=model_path,
                label_path=label_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            det._task = info.task
            return det

        # ── .onnx → task-specific detector ────────────────────────────────────
        if info.task == "segmentation":
            det = SegDetector(
                model_path=model_path,
                label_path=label_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        elif info.task == "pose":
            det = PoseDetector(
                model_path=model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        else:
            det = YOLODetector(
                model_path=model_path,
                label_path=label_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

        det._task = info.task
        return det

    def auto_load_single(
        self,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
    ) -> Optional[Any]:
        """
        If exactly one model exists, load and return it automatically.
        Returns None if 0 or multiple models found.
        """
        if len(self._catalog) == 1:
            logger.info(f"Auto-loading sole model: {self._catalog[0].name}")
            return self.load(self._catalog[0].name, conf_threshold, iou_threshold)
        return None
