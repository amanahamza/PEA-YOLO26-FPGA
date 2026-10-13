import os
import random
from collections import defaultdict

# --- Config ---
train_img_dir = "/home/m250647ec/PCB_dataset/train/images"
train_label_dir = "/home/m250647ec/PCB_dataset/train/labels"
num_classes = 6
per_class_sample = 35   # images sampled per class before dedup
seed = 42                # fixed seed so results are reproducible

random.seed(seed)

# --- Build class -> image mapping ---
class_to_images = defaultdict(set)

for label_file in os.listdir(train_label_dir):
    if not label_file.endswith(".txt"):
        continue

    img_name = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(train_img_dir, img_name)

    # skip if the matching image doesn't actually exist
    if not os.path.exists(img_path):
        print(f"Warning: no matching image for {label_file}, skipping")
        continue

    label_path = os.path.join(train_label_dir, label_file)
    with open(label_path) as f:
        classes_in_image = set(
            int(line.split()[0]) for line in f if line.strip()
        )

    for c in classes_in_image:
        class_to_images[c].add(img_name)

# --- Report how many images exist per class ---
print("Images available per class:")
for c in range(num_classes):
    print(f"  class {c}: {len(class_to_images[c])} images")

# --- Sample per class, merge into one set (auto-dedupes) ---
calibration_set = set()
for c in range(num_classes):
    imgs = list(class_to_images[c])
    sample_size = min(per_class_sample, len(imgs))
    sampled = random.sample(imgs, sample_size)
    calibration_set.update(sampled)

print(f"\nTotal unique calibration images: {len(calibration_set)}")

# --- Save the list ---
output_file = "calibration_images.txt"
with open(output_file, "w") as f:
    for img in sorted(calibration_set):
        f.write(img + "\n")

print(f"Saved list to {output_file}")
