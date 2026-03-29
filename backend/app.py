"""
Atlas 200I DK A2 Vision Server
Supports: YOLO ONNX / .om NPU inference, distance estimation,
          instance segmentation, pose estimation, real-time video
"""

import os
import sys
import time
import threading
import base64
import json
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import YOLODetector
from distance_estimator import DistanceEstimator
from model_manager import ModelManager
from seg_detector import SegDetector, draw_seg
from pose_detector import PoseDetector, draw_pose

# ─── Logging ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Directories ──────────────────────────────────────────────────────────────
MODELS_DIR   = BASE_DIR / "models"
UPLOADS_DIR  = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

for d in (MODELS_DIR, UPLOADS_DIR):
    d.mkdir(exist_ok=True)

# ─── App Init ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app, origins="*")
socketio = SocketIO(
    app, cors_allowed_origins="*", async_mode="threading",
    max_http_buffer_size=10 * 1024 * 1024,
)

# ─── Model Manager (startup scan) ─────────────────────────────────────────────
model_manager = ModelManager(MODELS_DIR)

# ─── Global State ─────────────────────────────────────────────────────────────
state = {
    "detector": None,
    "model_name": None,
    "model_task": "detection",   # detection | segmentation | pose | classification | distance | all
    "camera": None,
    "is_streaming": False,
    "stream_thread": None,
    "mode": "detection",
    "conf_threshold": 0.45,
    "iou_threshold": 0.45,
    "target_class": None,
    "camera_index": 0,
    "record": False,
    "video_writer": None,
}

perf = {
    "fps": 0.0,
    "inference_ms": 0.0,
    "frame_count": 0,
    "detection_count": 0,
    "start_time": time.time(),
    "_fps_t": time.time(),
    "_fps_frames": 0,
}

# ─── FPS ──────────────────────────────────────────────────────────────────────
def tick_fps():
    perf["_fps_frames"] += 1
    perf["frame_count"] += 1
    now = time.time()
    elapsed = now - perf["_fps_t"]
    if elapsed >= 1.0:
        perf["fps"] = perf["_fps_frames"] / elapsed
        perf["_fps_frames"] = 0
        perf["_fps_t"] = now


# ─── Startup model scan ────────────────────────────────────────────────────────
def startup_scan():
    """
    Called once at server start.
    Prints a table of available models and auto-loads if exactly one is found.
    """
    logger.info("=" * 60)
    logger.info("Atlas Vision Server – scanning models/")
    models = model_manager.scan()
    model_manager.print_table()

    if not models:
        logger.warning("No models found in models/. "
                       "Add .onnx or .om files and call /api/models/reload.")
        return

    # Auto-load if only one model present
    det = model_manager.auto_load_single(
        state["conf_threshold"], state["iou_threshold"]
    )
    if det is not None:
        info = model_manager.get_info(model_manager.list_models()[0]["name"])
        state["detector"] = det
        state["model_name"] = info.name
        state["model_task"] = info.task
        state["mode"] = info.task
        logger.info(f"Auto-loaded: {info.name}")
    else:
        logger.info(f"{len(models)} models available – "
                    "select one via /api/load_model or the UI.")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR) + "/static", filename)


@app.route("/api/status", methods=["GET"])
def status():
    uptime = int(time.time() - perf["start_time"])
    return jsonify({
        "status": "running",
        "model_loaded": state["detector"] is not None,
        "model_name": state["model_name"],
        "model_task": state["model_task"],
        "streaming": state["is_streaming"],
        "mode": state["mode"],
        "fps": round(perf["fps"], 1),
        "inference_ms": round(perf["inference_ms"], 1),
        "frame_count": perf["frame_count"],
        "uptime_s": uptime,
        "board": "Atlas 200I DK A2",
    })


# ── /api/models  (extended: shows .onnx AND .om, with task info) ───────────────
@app.route("/api/models", methods=["GET"])
def get_models():
    """Return all scanned models with task and backend info."""
    return jsonify({
        "models": model_manager.list_models(),
        "count": len(model_manager.list_models()),
        "current": state["model_name"],
    })


@app.route("/api/models/reload", methods=["POST"])
def reload_models():
    """Re-scan the models directory (useful after uploading new files)."""
    model_manager.scan()
    return jsonify({"models": model_manager.list_models(),
                    "count": len(model_manager.list_models())})


@app.route("/api/upload/model", methods=["POST"])
def upload_model():
    results = {}
    for key, f in request.files.items():
        if f.filename:
            save_path = MODELS_DIR / f.filename
            f.save(str(save_path))
            results[f.filename] = "saved"
            logger.info(f"Uploaded: {f.filename}")
    model_manager.scan()   # refresh catalog
    return jsonify({"success": True, "saved": results,
                    "models": model_manager.list_models()})


@app.route("/api/load_model", methods=["POST"])
def load_model():
    data = request.json or {}
    model_name = data.get("model_name", "")
    conf = float(data.get("conf_threshold", state["conf_threshold"]))
    iou  = float(data.get("iou_threshold",  state["iou_threshold"]))

    try:
        det = model_manager.load(model_name, conf, iou)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)})
    except Exception as e:
        logger.error(f"load_model error: {e}")
        return jsonify({"success": False, "error": str(e)})

    state["detector"]      = det
    state["model_name"]    = model_name
    state["conf_threshold"] = conf
    state["iou_threshold"]  = iou

    info = model_manager.get_info(model_name)
    task = info.task if info else getattr(det, "_task", "detection")
    state["model_task"] = task
    state["mode"]       = task   # switch mode automatically

    response = {
        "success": True,
        "message": f"Loaded {model_name}",
        "task": task,
        "backend": getattr(det, "_task", "onnx"),
        "input_shape": list(det.input_shape),
        "num_classes": det.num_classes,
    }
    if hasattr(det, "class_names"):
        response["classes"] = det.class_names[:20]
    logger.info(f"Loaded model: {model_name}  task={task}")
    return jsonify(response)


@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.json or {}
    if "conf_threshold" in data:
        state["conf_threshold"] = float(data["conf_threshold"])
        if state["detector"]:
            state["detector"].conf_threshold = state["conf_threshold"]
    if "iou_threshold" in data:
        state["iou_threshold"] = float(data["iou_threshold"])
        if state["detector"]:
            state["detector"].iou_threshold = state["iou_threshold"]
    if "mode" in data:
        state["mode"] = data["mode"]
    if "target_class" in data:
        state["target_class"] = data["target_class"] or None
    return jsonify({"success": True, "config": {
        "conf_threshold": state["conf_threshold"],
        "iou_threshold":  state["iou_threshold"],
        "mode": state["mode"],
        "target_class": state["target_class"],
    }})


# ─── Single-image inference ────────────────────────────────────────────────────
@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    if state["detector"] is None:
        return jsonify({"success": False, "error": "No model loaded"})

    f = request.files.get("image")
    if not f:
        return jsonify({"success": False, "error": "No image uploaded"})

    img_bytes = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"success": False, "error": "Cannot decode image"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(UPLOADS_DIR / f"{timestamp}_orig.jpg"), frame)

    t0 = time.time()
    result_frame, detections = _run_and_draw(frame.copy())
    inference_ms = (time.time() - t0) * 1000
    perf["inference_ms"] = inference_ms

    # Serialise (masks are numpy arrays – strip before JSON)
    serial_dets = _serialise_detections(detections)

    _, buf = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64 = base64.b64encode(buf).decode()

    distances = []
    if state["mode"] in ("distance", "all"):
        estimator = DistanceEstimator()
        for d in serial_dets:
            dist = estimator.estimate(d["bbox"], d["class_name"])
            d["distance_cm"] = dist
            distances.append({"class": d["class_name"], "distance_cm": dist})

    cv2.imwrite(str(UPLOADS_DIR / f"{timestamp}_result.jpg"), result_frame)
    return jsonify({
        "success": True,
        "detections": serial_dets,
        "distances": distances,
        "inference_ms": round(inference_ms, 1),
        "image_b64": img_b64,
        "detection_count": len(serial_dets),
        "mode": state["mode"],
    })


# ─── Core inference + drawing dispatcher ──────────────────────────────────────
def _run_and_draw(frame):
    """
    Run inference with the currently loaded detector and draw results.
    Dispatches to the right detector method and draw function based on task/mode.

    Returns (annotated_frame, raw_detections_list).
    """
    det = state["detector"]
    mode = state["mode"]

    if det is None:
        return frame, []

    # ── Segmentation ──────────────────────────────────────────────────────────
    if isinstance(det, SegDetector) or mode == "segmentation":
        if hasattr(det, "detect_seg"):
            detections = det.detect_seg(frame)
        else:
            detections = det.detect(frame)
        frame = draw_seg(frame, detections)
        return frame, detections

    # ── Pose ──────────────────────────────────────────────────────────────────
    if isinstance(det, PoseDetector) or mode == "pose":
        if hasattr(det, "detect_pose"):
            detections = det.detect_pose(frame)
        else:
            detections = det.detect(frame)
        frame = draw_pose(frame, detections)
        return frame, detections

    # ── Standard detection / distance / all ───────────────────────────────────
    detections = det.detect(frame)
    if state["target_class"]:
        detections = [d for d in detections if d["class_name"] == state["target_class"]]
    if mode in ("distance", "all"):
        estimator = DistanceEstimator()
        for d in detections:
            d["distance_cm"] = estimator.estimate(d["bbox"], d["class_name"])
    frame = draw_detections(frame, detections, mode)
    return frame, detections


def _serialise_detections(dets):
    """Remove non-JSON-serialisable fields (e.g. numpy mask arrays)."""
    out = []
    for d in dets:
        row = {k: v for k, v in d.items() if k != "mask"}
        # keypoints are already plain dicts – safe to include
        out.append(row)
    return out


# ─── Video streaming ──────────────────────────────────────────────────────────
def _open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        for dev in [f"/dev/video{index}", "/dev/video0"]:
            cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                break
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _stream_loop():
    logger.info("Stream thread started")
    cap = _open_camera(state["camera_index"])
    if not cap.isOpened():
        socketio.emit("stream_error", {"error": "Cannot open camera"})
        state["is_streaming"] = False
        return

    state["camera"] = cap

    while state["is_streaming"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        tick_fps()
        inf_ms = 0.0

        if state["detector"] is not None:
            t0 = time.time()
            result_frame, detections = _run_and_draw(frame.copy())
            inf_ms = (time.time() - t0) * 1000
            perf["inference_ms"] = inf_ms
            perf["detection_count"] = len(detections)
        else:
            result_frame = frame.copy()
            detections = []

        result_frame = draw_overlay(result_frame)

        _, buf = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode()

        serial_dets = _serialise_detections(detections)
        socketio.emit("frame", {
            "image": b64,
            "fps": round(perf["fps"], 1),
            "inference_ms": round(inf_ms, 1),
            "detections": serial_dets,
            "detection_count": len(serial_dets),
            "frame_count": perf["frame_count"],
            "mode": state["mode"],
        })
        time.sleep(0.01)

    cap.release()
    state["camera"] = None
    logger.info("Stream thread stopped")


@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    if state["is_streaming"]:
        return jsonify({"success": False, "error": "Already streaming"})
    data = request.json or {}
    state["camera_index"] = int(data.get("camera_index", 0))
    state["is_streaming"] = True
    t = threading.Thread(target=_stream_loop, daemon=True)
    t.start()
    state["stream_thread"] = t
    return jsonify({"success": True})


@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    state["is_streaming"] = False
    if state["camera"]:
        state["camera"].release()
        state["camera"] = None
    return jsonify({"success": True})


# ─── MJPEG fallback ───────────────────────────────────────────────────────────
def _mjpeg_generator():
    cap = _open_camera(state["camera_index"])
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tick_fps()
            if state["detector"]:
                result_frame, _ = _run_and_draw(frame)
            else:
                result_frame = frame
            result_frame = draw_overlay(result_frame)
            _, buf = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.route("/api/stream/mjpeg")
def mjpeg_stream():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ─── Drawing helpers ──────────────────────────────────────────────────────────
PALETTE = [
    (52, 211, 153), (251, 113, 133), (96, 165, 250), (250, 204, 21),
    (167, 139, 250), (34, 211, 238), (249, 115, 22), (132, 204, 22),
    (232, 121, 249), (20, 184, 166),
]


def _color(cls_id):
    return PALETTE[int(cls_id) % len(PALETTE)]


def draw_detections(frame, detections, mode="detection"):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = det.get("class_id", 0)
        conf   = det.get("confidence", 0)
        name   = det.get("class_name", "unknown")
        color  = _color(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        if "distance_cm" in det and mode in ("distance", "all"):
            label += f" {det['distance_cm']:.0f}cm"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if mode == "classification":
            cv2.putText(frame, f"#{cls_id}: {name}",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_overlay(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (240, 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    lines = [
        f"FPS: {perf['fps']:.1f}",
        f"Inf: {perf['inference_ms']:.1f}ms",
        f"Det:{perf['detection_count']}  Frm:{perf['frame_count']}",
        f"Mode: {state['mode']}  Mdl:{state['model_name'] or 'none'}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 1, cv2.LINE_AA)
    return frame


# ─── SocketIO ─────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    # Push current model list to newly connected client
    emit("connected", {
        "status": "ok",
        "board": "Atlas 200I DK A2",
        "models": model_manager.list_models(),
        "current_model": state["model_name"],
        "mode": state["mode"],
    })


@socketio.on("disconnect")
def on_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on("ping_stats")
def on_ping_stats(_):
    emit("stats", {
        "fps": round(perf["fps"], 1),
        "inference_ms": round(perf["inference_ms"], 1),
        "frame_count": perf["frame_count"],
        "detection_count": perf["detection_count"],
        "model": state["model_name"],
        "task": state["model_task"],
        "mode": state["mode"],
    })


@socketio.on("load_model")
def on_load_model(data):
    """Allow loading a model directly via WebSocket."""
    name = data.get("model_name", "")
    conf = float(data.get("conf_threshold", state["conf_threshold"]))
    iou  = float(data.get("iou_threshold",  state["iou_threshold"]))
    try:
        det = model_manager.load(name, conf, iou)
        state["detector"]      = det
        state["model_name"]    = name
        state["conf_threshold"] = conf
        state["iou_threshold"]  = iou
        info = model_manager.get_info(name)
        task = info.task if info else getattr(det, "_task", "detection")
        state["model_task"] = task
        state["mode"] = task
        emit("model_loaded", {"success": True, "model": name, "task": task})
        logger.info(f"[WS] Model loaded: {name}")
    except Exception as e:
        emit("model_loaded", {"success": False, "error": str(e)})


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    startup_scan()
    logger.info(f"Frontend:  {FRONTEND_DIR}")
    logger.info(f"Models:    {MODELS_DIR}")
    logger.info("Starting server on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
