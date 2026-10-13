import numpy as np

# FIXED: use the actual DequantizeLinear_Output — this is what really feeds the conv
pooled_input = np.load("golden_reference/01_missing_hole_20_1__model.17_ReduceMean_output_0_DequantizeLinear_Output.npy")
expected_output = np.load("golden_reference/01_missing_hole_20_1__model.17_branch_h_branch_h.0_Conv_output_0.npy")

ch = 0
input_vals = pooled_input[0, ch, :, 0]
expected_vals = expected_output[0, ch, :, 0]

w_quant = np.array([-105, 127, 3])
w_scale = 0.07008958
w_real = w_quant * w_scale

bias_quant = -480
bias_scale = 0.0026143133
bias_real = bias_quant * bias_scale

padded = np.pad(input_vals, (1, 1), mode='constant')
computed = np.zeros(80)
for row in range(80):
    window = padded[row:row+3]
    computed[row] = np.sum(window * w_real) + bias_real

print("Row | Computed | Expected | Diff")
for i in range(10):
    print(f"{i:3d} | {computed[i]:.6f} | {expected_vals[i]:.6f} | {abs(computed[i]-expected_vals[i]):.6f}")

print(f"\nMax abs diff across all 80 rows: {np.max(np.abs(computed - expected_vals)):.6f}")
