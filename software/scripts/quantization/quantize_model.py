from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from calibration_reader import PCBCalibrationDataReader

model_fp32 = "/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best.onnx"
model_int8 = "/home/m250647ec/ultralytics/runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best_int8.onnx"

reader = PCBCalibrationDataReader(
    image_list_path="calibration_images.txt",
    image_dir="/home/m250647ec/PCB_dataset/train/images"
)

quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=["Conv"],   # only quantize the actual convolutions
)

print(f"Quantized model saved to {model_int8}")
