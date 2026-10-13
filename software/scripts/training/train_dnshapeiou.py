"""
Training script for YOLO26n-PEA with DN-ShapeIoU loss.
Original pea.yaml and loss.py are UNTOUCHED.
"""

import torch
from ultralytics import YOLO
from ultralytics.utils.dn_shapeiou import DNShapeIoULoss, shape_iou
import ultralytics.utils.loss as loss_module

# ── DN-ShapeIoU instance ─────────────────────────────────────────
_dn_loss = DNShapeIoULoss(scale=0.5, alpha=1.9, delta=3.0, momentum=0.97)

_original_bbox_iou = loss_module.bbox_iou

def patched_bbox_iou(box1, box2, xywh=True, GIoU=False,
                     DIoU=False, CIoU=False, WIoU=False, eps=1e-7):
    """
    Handles all bbox_iou call signatures including WIoU=True.
    Replaces WIoU with DN-ShapeIoU for box regression loss.
    """
    if WIoU:
        # This is the box regression call — replace with DN-ShapeIoU
        if xywh:
            b1 = torch.cat([box1[..., :2] - box1[..., 2:] / 2,
                            box1[..., :2] + box1[..., 2:] / 2], dim=-1)
            b2 = torch.cat([box2[..., :2] - box2[..., 2:] / 2,
                            box2[..., :2] + box2[..., 2:] / 2], dim=-1)
        else:
            b1, b2 = box1, box2

        # Move EMA buffer to correct device
        _dn_loss.loss_ema = _dn_loss.loss_ema.to(b1.device)

        # DN-ShapeIoU returns loss values — convert to iou-like (1 - loss)
        dn_loss_val = _dn_loss(b1, b2, reduction='none')
        return 1.0 - dn_loss_val  # YOLO uses (1 - iou) as loss internally

    # All other calls (assignment matching etc.) use original
    return _original_bbox_iou(box1, box2, xywh=xywh,
                               GIoU=GIoU, DIoU=DIoU,
                               CIoU=CIoU, eps=eps)

# Apply patch
loss_module.bbox_iou = patched_bbox_iou
print("✅ DN-ShapeIoU patch applied (WIoU signature fixed)")

# ── Verify pea.py is v1 ──────────────────────────────────────────
from ultralytics.nn.modules.pea import PEA
import inspect
sig = inspect.signature(PEA.__init__)
assert 'reduction' not in str(sig), \
    "pea.py has v3 code! Restore v1 first."
print("✅ pea.py is v1")

# ── Train ────────────────────────────────────────────────────────
model = YOLO("ultralytics/cfg/models/26/yolo26n_pea_dnsiou.yaml")

results = model.train(
    data          = "pku_pcb.yaml",
    epochs        = 300,
    imgsz         = 640,
    batch         = 16,
    optimizer     = "AdamW",
    pretrained    = "runs/detect/pcb_ablation/E_final_pea_300ep_batch16/weights/best.pt",
    lr0           = 0.001,
    lrf           = 0.01,
    cos_lr        = True,
    warmup_epochs = 3,
    patience      = 50,
    device        = 0,
    project       = "pcb_ablation",
    name          = "E_pea_dnshapeiou_300ep",
    amp           = False,
)

print("\n✅ Training complete")
