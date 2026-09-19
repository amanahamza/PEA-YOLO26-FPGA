\# PEA-YOLO26n FPGA Accelerator



Hardware acceleration of PEA-YOLO26n — a lightweight PCB defect detector with a

novel PCB Edge Attention (PEA) module — targeting the PYNQ-Z2 (Zynq-7020) platform.



\## Project Status

\- Software pipeline: model export, INT8 quantization, golden-reference generation — complete

\- Hardware: MAC unit, MAC array, Branch-H sliding window — synthesized and verified (Vitis HLS)

\- In progress: full PEA module, generic backbone blocks, Vivado integration, PYNQ deployment



\## Key Results

| Metric | Value |

|---|---|

| FP32 mAP@0.5 | 93.5% |

| INT8 mAP@0.5 | 90.8% |

| Single MAC | 1 DSP, 2.1ns |

| 8-wide MAC array | 4 DSP, 6.3ns |

| Branch-H sliding window | 2 DSP, 7.07ns |



\## Structure

\- `hardware/` — Vitis HLS source files for verified hardware modules

\- `docs/` — project reports



\## Author

Amana T. (M250647EC), MTech EDT, NIT Calicut — supervised by Dr. Bindiya T.S.

