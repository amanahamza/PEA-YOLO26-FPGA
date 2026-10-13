from ultralytics import YOLO

print("=== SCRATCH on TEST set ===")
model_scratch = YOLO('/home/m250647ec/HRIPCB_UPDATE/results/hripcb_pea_scratch/weights/best.pt')
metrics_scratch = model_scratch.val(data='/home/m250647ec/HRIPCB_UPDATE/data.yaml', split='test')

print("\n=== TRANSFER on TEST set ===")
model_transfer = YOLO('/home/m250647ec/HRIPCB_UPDATE/results/hripcb_pea_transfer/weights/best.pt')
metrics_transfer = model_transfer.val(data='/home/m250647ec/HRIPCB_UPDATE/data.yaml', split='test')

