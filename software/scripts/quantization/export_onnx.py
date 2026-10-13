from ultralytics import YOLO

model = YOLO("/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best.pt")
model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
