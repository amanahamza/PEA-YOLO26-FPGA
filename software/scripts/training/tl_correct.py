from ultralytics import YOLO

# Load FULL checkpoint — not yaml
model = YOLO('runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best.pt')

model.train(
    data='/home/m250647ec/DeepPCB_dataset/data.yaml',
    epochs=150,
    lr0=0.01,         # same as scratch
    lrf=0.01,
    optimizer='AdamW',
    freeze=10,        # freeze backbone, train head only
    cos_lr=False,
    warmup_epochs=2,
    patience=30,
    imgsz=640,
    batch=16,
    device=0,
    project='pcb_ablation',
    name='TL_correct_freeze10'
)
