from ultralytics import YOLO

model = YOLO('runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best.pt')

model.train(
    data='/home/m250647ec/DeepPCB_dataset/data.yaml',
    epochs=300,
    lr0=0.005,        # moderate — not too high, not too low
    lrf=0.01,
    optimizer='AdamW',
    freeze=0,         # unfreeze everything
    cos_lr=False,
    warmup_epochs=5,  # longer warmup to stabilise
    patience=50,
    imgsz=640,
    batch=16,
    device=0,
    project='pcb_ablation',
    name='TL_final_unfreeze_lr005'
)
