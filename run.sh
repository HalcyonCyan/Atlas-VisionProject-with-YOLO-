#!/bin/bash
# Atlas Vision — Run Script (conda python, 无需 venv)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
PYTHON="/usr/local/miniconda3/bin/python"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# 检查模型
MODEL_COUNT=$(ls "$SCRIPT_DIR/models/"*.onnx 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
  echo -e "${YELLOW}[WARN] models/ 目录下没有 .onnx 模型文件${NC}"
  echo "       请先把模型文件复制到 models/ 目录"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Atlas Vision Server Starting...      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "  🌐 Web UI: http://localhost:5000"
echo "  🌐 LAN:    http://$(hostname -I | awk '{print $1}'):5000"
echo "  📹 MJPEG:  http://localhost:5000/api/stream/mjpeg"
echo ""
echo "  按 Ctrl+C 停止"
echo ""

export PYTHONUNBUFFERED=1
cd "$BACKEND_DIR"
exec "$PYTHON" app.py
