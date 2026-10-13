# transfer_hripcb.py
from ultralytics import YOLO

model = YOLO('/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best.pt')

results = model.train(
    data='/home/m250647ec/HRIPCB_UPDATE/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    project='/home/m250647ec/HRIPCB_UPDATE/results',
    name='hripcb_pea_transfer',
    patience=30,
    optimizer='auto',
    lr0=0.001,
    warmup_epochs=3,
    exist_ok=True,
    verbose=True,
)
