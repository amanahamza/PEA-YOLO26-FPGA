#!/bin/bash
cd ~/ultralytics

echo "======================================"
echo "Continuing baselines from B2"
echo "$(date)"
echo "======================================"

# ── B2: YOLOv8s ──────────────────────────
echo "[2/5] Training YOLOv8s 300ep..."
yolo detect train \
  model=yolov8s.yaml \
  data=pku_pcb.yaml \
  epochs=300 imgsz=640 batch=16 \
  optimizer=AdamW \
  pretrained=/home/m250647ec/PCB_dataset/yolov8s.pt \
  lr0=0.01 lrf=0.01 cos_lr=True \
  warmup_epochs=3 patience=50 \
  name=B2_yolov8s_300ep \
  project=runs/detect/baseline_300ep \
  device=0 || true
echo "[2/5] YOLOv8s DONE — $(date)"

# ── B3: YOLOv8n+CBAM ─────────────────────
echo "[3/5] Training YOLOv8n+CBAM 300ep..."
yolo detect train \
  model=ultralytics/cfg/models/v8/yolov8n_cbam.yaml \
  data=pku_pcb.yaml \
  epochs=300 imgsz=640 batch=16 \
  optimizer=AdamW \
  pretrained=/home/m250647ec/PCB_dataset/yolov8n.pt \
  lr0=0.01 lrf=0.01 cos_lr=True \
  warmup_epochs=3 patience=50 \
  name=B3_yolov8n_cbam_300ep \
  project=runs/detect/baseline_300ep \
  device=0 || true
echo "[3/5] YOLOv8n+CBAM DONE — $(date)"

# ── B4: YOLO11n ───────────────────────────
echo "[4/5] Training YOLO11n 300ep..."
yolo detect train \
  model=yolo11n.yaml \
  data=pku_pcb.yaml \
  epochs=300 imgsz=640 batch=16 \
  optimizer=AdamW \
  pretrained=/home/m250647ec/PCB_dataset/yolo11n.pt \
  lr0=0.01 lrf=0.01 cos_lr=True \
  warmup_epochs=3 patience=50 \
  name=B4_yolo11n_300ep \
  project=runs/detect/baseline_300ep \
  device=0 || true
echo "[4/5] YOLO11n DONE — $(date)"

# ── B5: YOLO26n ───────────────────────────
echo "[5/5] Training YOLO26n 300ep..."
yolo detect train \
  model=ultralytics/cfg/models/26/yolo26.yaml \
  data=pku_pcb.yaml \
  epochs=300 imgsz=640 batch=16 \
  optimizer=AdamW \
  pretrained=/home/m250647ec/PCB_dataset/yolo26n.pt \
  lr0=0.01 lrf=0.01 cos_lr=True \
  warmup_epochs=3 patience=50 \
  name=B5_yolo26n_300ep \
  project=runs/detect/baseline_300ep \
  device=0 || true
echo "[5/5] YOLO26n DONE — $(date)"

echo "======================================"
echo "All baselines complete — $(date)"
echo "======================================"
