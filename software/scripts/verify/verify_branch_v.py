import numpy as np

pooled_input = np.load("golden_reference/01_missing_hole_20_1__model.17_ReduceMean_1_output_0_DequantizeLinear_Output.npy")
expected_output = np.load("golden_reference/01_missing_hole_20_1__model.17_branch_v_branch_v.0_Conv_output_0.npy")

ch = 0
# Branch V pools height -> shape (1,64,1,80), so index differently than Branch H
input_vals = pooled_input[0, ch, 0, :]
expected_vals = expected_output[0, ch, 0, :]

w_quant = np.array([-5, 127, -22])
w_scale = 0.12130069
w_real = w_quant * w_scale

bias_quant = 163
bias_scale = 0.00380655
bias_real = bias_quant * bias_scale

padded = np.pad(input_vals, (1, 1), mode='constant')
computed = np.zeros(80)
for col in range(80):
    window = padded[col:col+3]
    computed[col] = np.sum(window * w_real) + bias_real

print("Col | Computed | Expected | Diff")
for i in range(10):
    print(f"{i:3d} | {computed[i]:.6f} | {expected_vals[i]:.6f} | {abs(computed[i]-expected_vals[i]):.6f}")

print(f"\nMax abs diff across all 80 cols: {np.max(np.abs(computed - expected_vals)):.6f}")
