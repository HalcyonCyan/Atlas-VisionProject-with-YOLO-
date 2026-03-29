#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Atlas Vision Project — Install Script
#  Target: Atlas 200I DK A2 (Ubuntu 22.04 / OpenEuler, Python 3.9+)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
LOG_DIR="$SCRIPT_DIR/logs"
MODELS_DIR="$SCRIPT_DIR/models"
UPLOADS_DIR="$SCRIPT_DIR/uploads"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]  ${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERR] ${NC} $1"; exit 1; }

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Atlas Vision — Installation Script           ║${NC}"
echo -e "${GREEN}║     Atlas 200I DK A2 · YOLO ONNX Platform        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=$(command -v python3.9 || command -v python3.10 || command -v python3.11 || command -v python3 || echo "")
[ -z "$PYTHON" ] && error "Python 3.9+ not found. Install: sudo apt install python3.9"
PY_VER=$($PYTHON --version 2>&1)
ok "Found: $PY_VER at $PYTHON"

# ── Create virtualenv ─────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  info "Creating virtual environment..."
  $PYTHON -m venv "$VENV_DIR"
  ok "Virtual environment created: $VENV_DIR"
else
  ok "Virtual environment already exists"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON_VENV="$VENV_DIR/bin/python"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
info "Upgrading pip..."
$PIP install --upgrade pip -q

# ── Install packages ──────────────────────────────────────────────────────────
info "Installing Python packages..."
$PIP install -r "$SCRIPT_DIR/requirements.txt" -q
ok "Python packages installed"

# ── Check for CANN / onnxruntime-cann ─────────────────────────────────────────
info "Checking for Ascend CANN environment..."
if [ -d "/usr/local/Ascend" ] || [ -d "/home/HwHiAiUser/Ascend" ]; then
  ok "Ascend CANN directory detected"
  warn "Consider installing onnxruntime-cann for NPU acceleration:"
  echo "       pip install onnxruntime-cann (if available for your CANN version)"
  echo "       Or use Ascend ACL API for om model inference"
else
  warn "CANN not detected, using CPU inference (onnxruntime)"
fi

# ── Check OpenCV ──────────────────────────────────────────────────────────────
info "Checking OpenCV installation..."
$PYTHON_VENV -c "import cv2; print('  OpenCV:', cv2.__version__)" && ok "OpenCV OK" || {
  warn "OpenCV failed, trying to install headless version..."
  $PIP install opencv-python-headless -q
}

# ── Check camera ──────────────────────────────────────────────────────────────
info "Checking camera devices..."
for dev in /dev/video0 /dev/video1 /dev/video2; do
  [ -e "$dev" ] && ok "Camera device found: $dev" || warn "Not found: $dev"
done

# ── Create directories ────────────────────────────────────────────────────────
info "Creating project directories..."
mkdir -p "$LOG_DIR" "$MODELS_DIR" "$UPLOADS_DIR"
ok "Directories ready"

# ── Create systemd service (optional) ────────────────────────────────────────
SYSTEMD_PATH="/etc/systemd/system/atlas-vision.service"
if command -v systemctl &>/dev/null && [ "$(id -u)" -eq 0 ]; then
  info "Creating systemd service..."
  cat > "$SYSTEMD_PATH" <<EOF
[Unit]
Description=Atlas Vision Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR/backend
ExecStart=$VENV_DIR/bin/python app.py
Restart=on-failure
RestartSec=5
StandardOutput=append:$LOG_DIR/service.log
StandardError=append:$LOG_DIR/service.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  ok "Systemd service installed: atlas-vision.service"
  echo ""
  echo "  To enable auto-start: sudo systemctl enable atlas-vision"
  echo "  To start now:         sudo systemctl start atlas-vision"
else
  warn "Not root — skipping systemd service installation"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Installation Complete!                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  📁 Place your .onnx model file in:  $MODELS_DIR/"
echo "  📄 Place label.txt in the same dir"
echo ""
echo "  🚀 Start the server:"
echo "       ./run.sh"
echo ""
echo "  🌐 Then open browser:"
echo "       http://localhost:5000"
echo "       http://$(hostname -I | awk '{print $1}'):5000  (LAN access)"
echo ""
