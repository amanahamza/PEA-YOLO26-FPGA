import onnx
from onnx import numpy_helper
import numpy as np

model_path = "/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best_int8.onnx"
model = onnx.load(model_path)

init_lookup = {init.name: init for init in model.graph.initializer}

# Find Branch H's Conv node and trace its inputs
for node in model.graph.node:
    if node.op_type == "Conv" and "branch_h" in node.name:
        print(f"Node: {node.name}")
        print(f"Inputs: {list(node.input)}\n")

        # Input 1 is typically the weight (possibly via a DequantizeLinear node)
        weight_input = node.input[1]
        print(f"Weight input name: {weight_input}")

        if weight_input in init_lookup:
            w = numpy_helper.to_array(init_lookup[weight_input])
            print(f"Weight tensor shape: {w.shape}, dtype: {w.dtype}")
            print(f"Channel 0 raw values: {w[0].flatten()}")

# Also search for any DequantizeLinear nodes feeding into branch_h's conv,
# to find the actual scale/zero-point constants
print("\n--- Searching for quantization params near branch_h ---")
for node in model.graph.node:
    if node.op_type in ("DequantizeLinear", "QuantizeLinear") and "branch_h" in node.name:
        print(f"\n{node.op_type}: {node.name}")
        for inp in node.input:
            if inp in init_lookup:
                val = numpy_helper.to_array(init_lookup[inp])
                print(f"  {inp} = {val}")

