from ultralytics import YOLO
model = YOLO('/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best.pt')

for i, module in enumerate(model.model.model):
    print(i, module.__class__.__name__)
