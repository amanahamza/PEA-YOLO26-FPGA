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
        print(f"Producer inputs: {list(producer.input)}")
        for inp in producer.input:
            if inp in init_lookup:
                val = numpy_helper.to_array(init_lookup[inp])
                print(f"  {inp}: shape={val.shape}, value(s)={val.flatten()[:5]}")
    elif tensor_name in init_lookup:
        val = numpy_helper.to_array(init_lookup[tensor_name])
        print(f"Direct initializer: shape={val.shape}, value(s)={val.flatten()[:5]}")
    else:
        print("Not found as node output or initializer")

# The three real inputs to Branch H's conv node
trace_and_print("/model.17/ReduceMean_output_0_DequantizeLinear_Output", "INPUT ACTIVATION")
trace_and_print("onnx::Conv_876_DequantizeLinear_Output", "WEIGHT")
trace_and_print("onnx::Conv_877", "BIAS")
