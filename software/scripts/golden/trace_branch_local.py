import onnx
from onnx import numpy_helper
import numpy as np

model_path = "/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best_int8.onnx"
model = onnx.load(model_path)

init_lookup = {init.name: init for init in model.graph.initializer}
node_by_output = {}
for node in model.graph.node:
    for out in node.output:
        node_by_output[out] = node

def trace_and_print(tensor_name, label):
    print(f"\n--- {label}: {tensor_name} ---")
    if tensor_name in node_by_output:
        producer = node_by_output[tensor_name]
        print(f"Produced by: {producer.op_type} ({producer.name})")
        for inp in producer.input:
            if inp in init_lookup:
                val = numpy_helper.to_array(init_lookup[inp])
                print(f"  {inp}: shape={val.shape}, value(s)={val.flatten()[:5]}")

for node in model.graph.node:
    if node.op_type == "Conv" and "branch_local" in node.name:
        print(f"Node: {node.name}")
        print(f"Inputs: {list(node.input)}")
        print(f"Attrs: {[(a.name, a.i if a.ints==[] else list(a.ints)) for a in node.attribute]}")
        trace_and_print(node.input[0], "INPUT ACTIVATION")
        trace_and_print(node.input[1], "WEIGHT")
        trace_and_print(node.input[2], "BIAS")
