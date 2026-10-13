import numpy as np

input_data = np.load("golden_reference/01_missing_hole_20_1__model.16_cv2_act_Mul_output_0_QuantizeLinear_Output.npy")
expected_output = np.load("golden_reference/01_missing_hole_20_1__model.17_branch_local_branch_local.0_Conv_output_0.npy")

group0_input = input_data[0, 0:16, :, :].astype(int)  # (16, 80, 80)
expected_ch0 = expected_output[0, 0, :, :]

input_scale = 0.05033763
input_zero = -122
w_ch0 = [[[-92, -51, -7], [61, 12, -15], [-13, -22, -53]],
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
         [[7, -36, -56], [-3, -5, -41], [56, 9, 16]]]
w_scale = 0.00619843
bias_quant = -2392
bias_scale = 0.00031201

with open("golden_data_local.h", "w") as f:
    f.write("#ifndef GOLDEN_DATA_LOCAL_H\n#define GOLDEN_DATA_LOCAL_H\n\n")
    f.write("#define IMG_H 80\n#define IMG_W 80\n#define IN_CH 16\n#define K 3\n\n")

    f.write("static const int input_q[IN_CH][IMG_H][IMG_W] = {\n")
    for c in range(16):
        f.write("{" + ",".join("{" + ",".join(str(v) for v in row) + "}" for row in group0_input[c]) + "},\n")
    f.write("};\n\n")

    f.write(f"static const int input_zero_point = {input_zero};\n")
    f.write(f"static const float input_scale = {input_scale}f;\n\n")

    f.write("static const int weight_q[IN_CH][K][K] = {\n")
    for c in range(16):
        f.write("{" + ",".join("{" + ",".join(str(v) for v in row) + "}" for row in w_ch0[c]) + "},\n")
    f.write("};\n")
    f.write(f"static const float weight_scale = {w_scale}f;\n\n")

    f.write(f"static const int bias_q = {bias_quant};\n")
    f.write(f"static const float bias_scale = {bias_scale}f;\n\n")

    f.write("static const float expected_output[IMG_H][IMG_W] = {\n")
    for row in expected_ch0:
        f.write("{" + ",".join(f"{v:.6f}f" for v in row) + "},\n")
    f.write("};\n\n#endif\n")

print("golden_data_local.h written")
