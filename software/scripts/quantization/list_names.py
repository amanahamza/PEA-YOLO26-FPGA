import onnx
from onnx import numpy_helper
import numpy as np

model_path = "/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best.onnx"
model = onnx.load(model_path)

# build a lookup: initializer name -> actual array
init_lookup = {init.name: init for init in model.graph.initializer}

print("PEA (model.17) Conv layers and their real weight ranges:\n")

for node in model.graph.node:
    if node.op_type == "Conv" and "/model.17/" in node.name:
        weight_name = node.input[1]  # second input is always the weight tensor
        if weight_name in init_lookup:
            w = numpy_helper.to_array(init_lookup[weight_name])
            print(f"{node.name}")
            print(f"  weight tensor: {weight_name}")
            print(f"  shape: {w.shape}")
            print(f"  min: {w.min():.4f}, max: {w.max():.4f}, abs max: {np.abs(w).max():.4f}\n")
