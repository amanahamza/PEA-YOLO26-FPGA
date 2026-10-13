import numpy as np
import os

golden_dir = "golden_reference"

# Layers we care about — the actual output of each branch's conv (post BN-fold, pre-activation)
target_layers = [
    "model.17_branch_h_branch_h.0_Conv_output_0",
    "model.17_branch_v_branch_v.0_Conv_output_0",
    "model.17_branch_local_branch_local.0_Conv_output_0",
    "model.17_fuse_Conv_output_0",
]

for layer in target_layers:
    all_values = []
    for fname in os.listdir(golden_dir):
        if layer in fname and "Quantize" not in fname and "Dequantize" not in fname:
            arr = np.load(os.path.join(golden_dir, fname))
            all_values.append(arr.flatten())

    if not all_values:
        print(f"{layer}: no files found, skipping\n")
        continue

    combined = np.concatenate(all_values)
    print(f"{layer}")
    print(f"  across {len(all_values)} images, {combined.size} total values")
    print(f"  min: {combined.min():.4f}, max: {combined.max():.4f}, abs max: {np.abs(combined).max():.4f}\n")
