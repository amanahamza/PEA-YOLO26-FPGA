import os
import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

class PCBCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_list_path, image_dir, input_name="images", img_size=640):
        with open(image_list_path) as f:
            self.image_files = [line.strip() for line in f if line.strip()]
        self.image_dir = image_dir
        self.input_name = input_name
        self.img_size = img_size
        self.index = 0

    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

    def _preprocess(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._letterbox(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def get_next(self):
        if self.index >= len(self.image_files):
            return None

        img_path = os.path.join(self.image_dir, self.image_files[self.index])
        print(f"Loading calibration image {self.index+1}/{len(self.image_files)}: {img_path}")
        input_tensor = self._preprocess(img_path)
        self.index += 1

        return {self.input_name: input_tensor}

    def rewind(self):
        self.index = 0


# --- Sanity check: only runs when this file is executed directly ---
if __name__ == "__main__":
    reader = PCBCalibrationDataReader(
        image_list_path="calibration_images.txt",
        image_dir="/home/m250647ec/PCB_dataset/train/images"
    )

    sample = reader.get_next()
    print("Shape:", sample["images"].shape)
    print("Dtype:", sample["images"].dtype)
