from ultralytics import YOLO

model = YOLO('/home/m250647ec/HRIPCB_UPDATE/results/hripcb_pea_scratch/weights/best.pt')
metrics = model.val(data='/home/m250647ec/HRIPCB_UPDATE/data.yaml')

print(f"\nOverall mAP50: {metrics.box.map50:.4f}")
print(f"Overall mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

print("\nPer-class mAP50:")
names = model.names
for i, ap50 in enumerate(metrics.box.ap50):
    print(f"{names[i]:20s} mAP50: {ap50:.4f}")

