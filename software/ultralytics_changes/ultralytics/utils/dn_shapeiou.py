"""
DN-ShapeIoU Loss — Shape-aware bounding box regression
with dynamic nonlinear gradient weighting.

Based on:
  Shape-IoU: https://arxiv.org/abs/2312.17663
  DN (Dynamic Nonlinear): inspired by Wise-IoU dynamic focusing

Drop-in replacement for W-IoU in YOLO26n-PEA training.
Original loss.py is UNTOUCHED — this is a separate file.
"""

import torch
import torch.nn as nn


def shape_iou(box1, box2, scale=0.0, eps=1e-7):
    """
    Shape-IoU computation.
    Considers bounding box shape and scale for regression.

    Args:
        box1: predicted boxes (x1,y1,x2,y2)
        box2: ground truth boxes (x1,y1,x2,y2)
        scale: shape scale factor
        eps: numerical stability

    Returns:
        shape_iou: IoU value with shape weighting
    """
    # Basic IoU
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Enclosing box
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    # Shape weighting — horizontal and vertical coefficients
    # ww weights horizontal axis based on gt box shape
    # hh weights vertical axis based on gt box shape
    s_cw = (b2_x2 - b2_x1)
    s_ch = (b2_y2 - b2_y1)

    # Scale-aware shape weights
    ww = 2 * s_cw ** scale / (s_cw ** scale + s_ch ** scale + eps)
    hh = 2 * s_ch ** scale / (s_cw ** scale + s_ch ** scale + eps)

    # Centre points
    cx1 = (b1_x1 + b1_x2) / 2
    cy1 = (b1_y1 + b1_y2) / 2
    cx2 = (b2_x1 + b2_x2) / 2
    cy2 = (b2_y1 + b2_y2) / 2

    # Diagonal of enclosing box squared
    c2 = cw ** 2 + ch ** 2 + eps

    # Shape distance — weighted centre distance
    shape_distance = (ww * (cx1 - cx2) ** 2 +
                      hh * (cy1 - cy2) ** 2) / c2

    # Angle/shape loss — penalises aspect ratio difference
    wh = (w1 / h1 - w2 / h2).abs()
    shape_loss = (1 - torch.exp(-wh / 4)).sum(dim=-1, keepdim=True) * 0.5

    # Shape-IoU
    shape_iou_val = iou - shape_distance - shape_loss

    return shape_iou_val.squeeze(-1)


class DNShapeIoULoss(nn.Module):
    """
    DN-ShapeIoU Loss.

    Combines Shape-IoU with dynamic nonlinear gradient weighting
    (inspired by Wise-IoU). Reduces gradient contribution from
    low-quality samples (geometric mean outlier detection).

    FPGA note: This loss is TRAINING ONLY — zero inference cost.
               No hardware changes required.
    """

    def __init__(self, scale=0.0, alpha=1.9, delta=3.0,
                 momentum=0.97, eps=1e-7):
        """
        Args:
            scale:    shape scale factor (0.0 = neutral, higher = more shape-aware)
            alpha:    DN focusing hyperparameter
            delta:    DN focusing hyperparameter
            momentum: EMA momentum for running loss average
            eps:      numerical stability
        """
        super().__init__()
        self.scale    = scale
        self.alpha    = alpha
        self.delta    = delta
        self.momentum = momentum
        self.eps      = eps

        # Running EMA of loss for outlier detection
        self.register_buffer('loss_ema',
                             torch.tensor(1.0, dtype=torch.float32))

    def forward(self, pred_bboxes, gt_bboxes, reduction='mean'):
        """
        Args:
            pred_bboxes: predicted boxes (N, 4) in xyxy format
            gt_bboxes:   ground truth boxes (N, 4) in xyxy format
            reduction:   'mean', 'sum', or 'none'

        Returns:
            loss: DN-ShapeIoU loss value
        """
        # Compute Shape-IoU
        siou = shape_iou(pred_bboxes, gt_bboxes,
                         scale=self.scale, eps=self.eps)

        # Shape-IoU loss (1 - iou)
        loss = 1.0 - siou

        # ── Dynamic Nonlinear (DN) weighting ──────────────────────
        # Compute outlier degree tau = L* / EMA(L)
        # L* is non-gradient version (detached)
        with torch.no_grad():
            loss_star = loss.detach()
            # Update EMA
            self.loss_ema = (self.momentum * self.loss_ema +
                             (1 - self.momentum) * loss_star.mean())
            # Outlier degree
            tau = loss_star / (self.loss_ema + self.eps)
            # Dynamic gradient weight gamma
            # High-quality boxes (low tau) → small gamma → stable update
            # Low-quality boxes (high tau) → small gamma → reduce noise
            gamma = tau / (self.alpha * self.eps *
                           self.alpha ** (self.delta - tau))
            gamma = gamma.clamp(0.0, 10.0)  # safety clip

        # Apply dynamic weighting
        loss = gamma.detach() * loss

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
