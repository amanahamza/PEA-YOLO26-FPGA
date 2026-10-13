import numpy as np

arr = np.load("golden_reference/01_missing_hole_20_1__model.17_Mul_2_output_0.npy")
print("Shape:", arr.shape, "Dtype:", arr.dtype)
print("Any NaN:", np.isnan(arr).any())
print("Min/Max:", arr.min(), arr.max())
