import onnxruntime as ort
from calibration_reader import PCBCalibrationDataReader

sess = ort.InferenceSession(
    "/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best_int8.onnx"
)
reader = PCBCalibrationDataReader(
    image_list_path="calibration_images.txt",
    image_dir="/home/m250647ec/PCB_dataset/train/images"
)
sample = reader.get_next()
out = sess.run(None, sample)

arr = out[0]
print("Output shape:", arr.shape)
print("Min:", arr.min(), "Max:", arr.max())
print("Any NaN:", (arr != arr).any())
print("First few rows:\n", arr[0, :5])
