import numpy as np

input_data = np.load("golden_reference/01_missing_hole_20_1__model.16_cv2_act_Mul_output_0_QuantizeLinear_Output.npy")
expected_output = np.load("golden_reference/01_missing_hole_20_1__model.17_branch_local_branch_local.0_Conv_output_0.npy")

print("Input shape:", input_data.shape)
print("Expected shape:", expected_output.shape)

input_scale = 0.05033763
input_zero = -122

w_ch0 = np.array([[[-92, -51, -7], [61, 12, -15], [-13, -22, -53]],
                   [[20, -37, 37], [11, 8, 33], [14, -14, 31]],
                   [[-23, 43, -13], [32, -14, -4], [-14, -20, -78]],
                   [[22, 34, 4], [48, 35, -70], [37, -46, 38]],
                   [[-37, -19, 30], [7, 56, 127], [-65, -28, -8]],
                   [[-25, 58, -24], [-20, 40, 41], [4, -43, -17]],
                   [[-32, 5, -8], [43, -14, 18], [24, 31, 21]],
                   [[-35, -45, -66], [-1, -96, 1], [-22, 15, 27]],
                   [[-18, -32, -27], [54, 67, 40], [-44, -61, -58]],
                   [[3, 9, 42], [2, -83, -1], [-54, -45, 27]],
                   [[-62, 0, 14], [-51, -17, 3], [-34, -27, 13]],
                   [[20, -24, -11], [48, -22, 47], [18, -4, 2]],
                   [[-68, -30, -42], [-21, -2, -105], [6, -45, -3]],
                   [[37, 19, 70], [-52, -47, 74], [-90, -20, -46]],
                   [[-28, -2, -12], [-73, -86, 23], [-76, -3, 47]],
                   [[7, -36, -56], [-3, -5, -41], [56, 9, 16]]])
w_scale = 0.00619843
w_real = w_ch0 * w_scale

bias_quant = -2392
bias_scale = 0.00031201
bias_real = bias_quant * bias_scale

# Group 0 = input channels 0-15
group0_input = input_data[0, 0:16, :, :].astype(np.float64)  # (16, 80, 80)
# Dequantize: real = scale * (q - zero_point). Note: this is the QUANTIZED tensor, needs zero-point applied.
group0_real = input_scale * (group0_input - input_zero)

padded = np.pad(group0_real, ((0,0),(1,1),(1,1)), mode='constant')

computed = np.zeros((80,80))
for row in range(80):
    for col in range(80):
        patch = padded[:, row:row+3, col:col+3]  # (16,3,3)
        computed[row, col] = np.sum(patch * w_real) + bias_real

expected_ch0 = expected_output[0, 0, :, :]
diff = np.abs(computed - expected_ch0)
print("\nMax abs diff:", diff.max())
print("Sample [0,0]:", computed[0,0], "vs", expected_ch0[0,0])
print("Sample [5,5]:", computed[5,5], "vs", expected_ch0[5,5])
