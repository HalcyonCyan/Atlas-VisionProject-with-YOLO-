# Atlas Vision — YOLO 视觉感知平台
### Atlas 200I DK A2 · ONNX Runtime · Flask + WebSocket

---

## 项目结构

```
vision_project/
├── backend/
│   ├── app.py                 # Flask 主服务 + WebSocket
│   ├── detector.py            # YOLO ONNX 检测器（v5/v6/v7/v8/v11）
│   └── distance_estimator.py  # 单目测距模块
├── frontend/
│   └── index.html             # 完整前端 SPA（无额外依赖）
├── models/                    # 放置 .onnx 和 label.txt
├── uploads/                   # 图片检测结果保存
├── logs/                      # 运行日志
├── requirements.txt
├── install.sh                 # 一键安装
└── run.sh                     # 启动服务
```

---

## 快速开始

### 1. 安装

```bash
chmod +x install.sh run.sh
./install.sh
```

### 2. 放置模型

```bash
# 将你的 ONNX 模型和标签文件放入 models/ 目录
cp /path/to/your_model.onnx    models/
cp /path/to/label.txt          models/your_model.txt  # 与 onnx 同名
```

**标签文件格式**（每行一个类名）：
```
person
bicycle
car
...
```

### 3. 启动

```bash
./run.sh
```

### 4. 打开浏览器

```
http://localhost:5000
```
或局域网访问：
```
http://192.168.x.x:5000
```

---

## 功能说明

| 功能 | 描述 |
|------|------|
| 🎯 **目标检测** | YOLO 检测 + 彩色 BBox 标注 |
| 🏷 **分类识别** | 按置信度聚合，显示 Top-10 类别 |
| 📏 **距离估计** | 基于针孔相机模型的单目测距 |
| ⚡ **全部模式** | 检测 + 分类 + 距离同时运行 |
| 📹 **实时视频** | WebSocket 帧推送，低延迟 |
| 🖼 **图片检测** | 上传图片离线推理 |
| 📊 **实时统计** | FPS / 推理时间 / 检测数量 |
| 🔧 **参数调节** | 置信度阈值 / IOU / 类别过滤 |

---

## YOLO 模型兼容性

| 模型版本 | 输出格式 | 兼容 |
|---------|---------|------|
| YOLOv5  | [B, N, 5+C] | ✅ |
| YOLOv6  | [B, N, 5+C] | ✅ |
| YOLOv7  | [B, N, 5+C] | ✅ |
| YOLOv8  | [B, 4+C, N] | ✅ |
| YOLOv11 | [B, 4+C, N] | ✅ |

---

## Atlas 200I DK A2 Ascend NPU 加速

默认使用 CPU 推理。若要使用 Ascend NPU 加速：

### 方式一：onnxruntime-cann（推荐）

```bash
# 查看 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 安装对应 onnxruntime-cann（版本需匹配）
pip install onnxruntime-cann==x.x.x
```

安装后 `detector.py` 会自动检测并优先使用 `CANNExecutionProvider`。

### 方式二：转换为 .om 模型

```bash
# 使用 ATC 工具转换
atc --model=your_model.onnx \
    --framework=5 \
    --output=your_model \
    --input_shape="images:1,3,640,640" \
    --soc_version=Ascend310B1
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/status` | 服务状态 |
| GET  | `/api/models` | 可用模型列表 |
| POST | `/api/load_model` | 加载模型 |
| POST | `/api/config` | 更新参数 |
| POST | `/api/detect/image` | 图片检测 |
| POST | `/api/stream/start` | 启动视频流 |
| POST | `/api/stream/stop` | 停止视频流 |
| GET  | `/api/stream/mjpeg` | MJPEG 备用流 |
| POST | `/api/upload/model` | 上传模型文件 |

---

## 测距算法说明

使用针孔相机模型：

```
距离(cm) = 物体真实高度(cm) × 焦距(px) / BBox高度(px)
```

内置常见 COCO 类别真实尺寸（person=170cm, car=145cm 等）。
可通过 `DistanceEstimator.calibrate()` 方法用实测数据标定焦距。

---

## 依赖版本

```
Python        3.9+
Flask         2.3+
Flask-SocketIO 5.3+
onnxruntime   1.16+
OpenCV        4.8+
numpy         1.24+
```

---

## 常见问题

**Q: 摄像头打不开？**
```bash
# 检查设备
ls /dev/video*
# 添加当前用户到 video 组
sudo usermod -aG video $USER
```

**Q: 推理很慢？**
- 确认模型输入尺寸（默认 640×640）
- 安装 onnxruntime-cann 使用 NPU
- 降低置信度阈值减少后处理量

**Q: 无法访问 Web UI？**
```bash
# 检查防火墙
sudo ufw allow 5000
# 或临时关闭
sudo ufw disable
```
