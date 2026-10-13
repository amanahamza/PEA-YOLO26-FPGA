import torch
import torch.nn as nn


class PEA(nn.Module):
    """
    PCB Edge Attention v3 — Spatial + Channel combined.

    NEW in v3 vs v1:
      - Channel attention: learns WHICH channels activate for defects
      - Dilated branch:    captures multi-scale defect context (dilation=2)
      - Backbone insertion: placed at 128ch backbone P3 (richer features)

    Insertion points in yolo26n_pea_v3.yaml:
      Layer 5:  backbone P3, 128 channels, 80×80  <-- NEW
      Layer 18: neck P3,     64 channels,  80×80  <-- existing
    """
    def __init__(self, channels: int, groups: int = 4,
                 reduction: int = 4):
        super().__init__()
        assert channels % groups == 0
        mid = max(channels // reduction, 8)

        # Channel attention — WHICH channels matter for PCB defects
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Branch H — 3×1 depthwise → mouse bite (horizontal edge)
        self.branch_h = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=(3, 1), padding=(1, 0),
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        # Branch V — 1×3 depthwise → spur (vertical edge)
        self.branch_v = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), padding=(0, 1),
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        # Branch Local — grouped 3×3 → blob shapes (missing hole, short)
        self.branch_local = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=3, padding=1,
                      groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        # Branch Dilated — dilated 3×3 → multi-scale context (NEW in v3)
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=3, padding=2, dilation=2,
                      groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        self.fuse    = nn.Conv2d(channels, channels,
                                 kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Step 1: channel recalibration (NEW)
        ch_w = self.channel_att(x)       # B×C×1×1
        x_c  = x * ch_w                  # channel-weighted features

        # Step 2: spatial branches on calibrated features
        pool_v = x_c.mean(dim=3, keepdim=True)
        h_feat = self.branch_h(pool_v).expand(B, C, H, W)

        pool_h = x_c.mean(dim=2, keepdim=True)
        v_feat = self.branch_v(pool_h).expand(B, C, H, W)

        local_feat   = self.branch_local(x_c)
        dilated_feat = self.branch_dilated(x_c)   # NEW

        # Step 3: fuse all 4 branches
        fused = self.fuse(
            h_feat + v_feat + local_feat + dilated_feat
        )

        # Step 4: sigmoid gate on ORIGINAL x (residual — cannot degrade)
        return x * self.sigmoid(fused)
