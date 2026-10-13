import os
import cv2
import numpy as np
import onnx
import onnxruntime as ort

# --- Config ---
model_path = "/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best_int8.onnx"
img_dir = "/home/m250647ec/PCB_dataset/val/images"
output_dir = "golden_reference"
os.makedirs(output_dir, exist_ok=True)

# Pick a small, diverse set of test images (one per class is a good start)
test_images = [
    "01_missing_hole_20_1.jpg",
    "01_mouse_bite_05_1.jpg",
    "01_open_circuit_02_1.jpg",
    "06_short_02_1.jpg",
    "04_spur_08_1.jpg",
    "04_spurious_copper_07_1.jpg",
]  # adjust filenames to match real files in your val/images folder

# --- Step 2: expose every PEA module tensor as a graph output ---
model = onnx.load(model_path)

pea_output_names = []
for node in model.graph.node:
    if "/model.17/" in node.name:
        for out_name in node.output:
            pea_output_names.append(out_name)

# Add each of these as a declared output of the graph (dtype/shape left generic)
existing_outputs = {o.name for o in model.graph.output}
for name in pea_output_names:
    if name not in existing_outputs:
        model.graph.output.append(onnx.ValueInfoProto(name=name))

modified_model_path = "best_int8_with_pea_outputs.onnx"
onnx.save(model, modified_model_path)
print(f"Modified graph saved with {len(pea_output_names)} extra PEA outputs")

# --- Preprocessing (same letterbox pipeline as before) ---
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = letterbox(img, 640)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

# --- Step 3 + 4: run inference, save every PEA tensor per image ---
sess = ort.InferenceSession(modified_model_path)
output_names = [o.name for o in sess.get_outputs()]

index_lines = []

for img_file in test_images:
    img_path = os.path.join(img_dir, img_file)
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping")
        continue

    input_tensor = preprocess(img_path)
    outputs = sess.run(output_names, {"images": input_tensor})

    img_stem = os.path.splitext(img_file)[0]

    for name, value in zip(output_names, outputs):
        if "/model.17/" not in name:
            continue  # only save PEA-related tensors for now

        safe_name = name.replace("/", "_").strip("_")
        save_path = os.path.join(output_dir, f"{img_stem}__{safe_name}.npy")
        np.save(save_path, value)
        index_lines.append(f"{img_stem}\t{name}\t{save_path}")

    print(f"Saved PEA reference tensors for {img_file}")

# --- Step 6: save the lookup index ---
with open(os.path.join(output_dir, "index.tsv"), "w") as f:
    f.write("image\tlayer_name\tfile_path\n")
    f.write("\n".join(index_lines))

print(f"\nDone. Index saved to {output_dir}/index.tsv")
