#!/bin/bash
# Complete baseline training script — all 5 models
# Run inside tmux: tmux new -s all_baselines# Then: bash run_all_baselines.sh

  # stop if any command fails
cd ~/ultralytics
source $(conda info --base)/etc/profile.d/conda.sh
conda activate yolo26_ema

echo "======================================"
echo "Starting all baseline training"
echo "$(date)"
echo "======================================"

# ── B1: YOLOv8n ──────────────────────────
echo "[1/5] Training YOLOv8n 300ep..."
yolo detect train \
  model=yolov8n.yaml \
  data=pku_pcb.yaml \
  epochs=300 imgsz=640 batch=16 \
  optimizer=AdamW \
  pretrained=/home/m250647ec/PCB_dataset/yolov8n.pt \
  lr0=0.01 lrf=0.01 cos_lr=True \
  warmup_epochs=3 patience=50 \
  name=B1_yolov8n_300ep \
  project=runs/detect/baseline_300ep \
  device=0
echo "[1/5] YOLOv8n DONE — $(date)"

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
  device=0
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
  device=0
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
  device=0
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
  device=0
echo "[5/5] YOLO26n DONE — $(date)"

# ── Evaluate all ──────────────────────────
echo ""
echo "======================================"
echo "All training complete. Evaluating..."
echo "======================================"

python3 - << 'PYEOF'
from ultralytics import YOLO

experiments = {
    "YOLOv8n":      "runs/detect/baseline_300ep/B1_yolov8n_300ep/weights/best.pt",
    "YOLOv8s":      "runs/detect/baseline_300ep/B2_yolov8s_300ep/weights/best.pt",
    "YOLOv8n+CBAM": "runs/detect/baseline_300ep/B3_yolov8n_cbam_300ep/weights/best.pt",
    "YOLO11n":      "runs/detect/baseline_300ep/B4_yolo11n_300ep/weights/best.pt",
    "YOLO26n":      "runs/detect/baseline_300ep/B5_yolo26n_300ep/weights/best.pt",
}

names = ['missing_hole','mouse_bite','open_circuit',
         'short','spur','spurious_copper']

print(f"\n{'='*80}")
print(f"{'Model':<15} {'mAP50':>7} {'mAP95':>7} {'MB_AP':>7} {'Spur':>7} {'P':>7} {'R':>7}")
print(f"{'='*80}")

for model_name, path in experiments.items():
    try:
        model = YOLO(path)
        m = model.val(data="pku_pcb.yaml",
                      imgsz=640, device=0, verbose=False)
        print(f"{model_name:<15} "
              f"{m.box.map50:>7.4f} "
              f"{m.box.map:>7.4f} "
              f"{m.box.ap[1]:>7.4f} "
              f"{m.box.ap[4]:>7.4f} "
              f"{m.box.mp:>7.4f} "
              f"{m.box.mr:>7.4f}")
    except Exception as e:
        print(f"{model_name:<15} ERROR: {e}")

print(f"{'='*80}")
print("Results saved. Copy table into thesis.")
PYEOF

echo ""
echo "======================================"
echo "Everything complete — $(date)"
echo "======================================"
