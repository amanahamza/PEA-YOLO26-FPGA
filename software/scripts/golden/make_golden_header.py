import numpy as np

pooled_q = np.load("golden_reference/01_missing_hole_20_1__model.17_ReduceMean_output_0_QuantizeLinear_Output.npy")
expected = np.load("golden_reference/01_missing_hole_20_1__model.17_branch_h_branch_h.0_Conv_output_0.npy")

ch = 0
q_vals = pooled_q[0, ch, :, 0].astype(int)
exp_vals = expected[0, ch, :, 0]

input_scale = 0.0372996
input_zero = -121
w_quant = [-105, 127, 3]
w_scale = 0.07008958
bias_quant = -480
bias_scale = 0.0026143133

with open("golden_data.h", "w") as f:
    f.write("#ifndef GOLDEN_DATA_H\n#define GOLDEN_DATA_H\n\n#define NUM_ROWS 80\n\n")
    f.write("static const int input_q[NUM_ROWS] = {" + ", ".join(str(v) for v in q_vals) + "};\n\n")
    f.write(f"static const int input_zero_point = {input_zero};\n")
    f.write(f"static const float input_scale = {input_scale}f;\n\n")
    f.write("static const int weight_q[3] = {" + ", ".join(str(v) for v in w_quant) + "};\n")
    f.write(f"static const float weight_scale = {w_scale}f;\n\n")
    f.write(f"static const int bias_q = {bias_quant};\n")
    f.write(f"static const float bias_scale = {bias_scale}f;\n\n")
    f.write("static const float expected_output[NUM_ROWS] = {" + ", ".join(f"{v:.6f}f" for v in exp_vals) + "};\n\n#endif\n")

print("golden_data.h written")
print("First 5 quantized inputs:", q_vals[:5])
