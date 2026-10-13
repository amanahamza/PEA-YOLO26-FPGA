import torch
import torch.nn as nn

class PEA(nn.Module):
    """PCB Edge Attention v1 — ACTIVE VERSION"""
    def __init__(self, channels: int = 64, groups: int = 4):
        super().__init__()
        assert channels % groups == 0
        self.branch_h = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3,1),
                      padding=(1,0), groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU())
        self.branch_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1,3),
                      padding=(0,1), groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU())
        self.branch_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU())
        self.fuse    = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        h_feat = self.branch_h(x.mean(3, keepdim=True)).expand(B,C,H,W)
        v_feat = self.branch_v(x.mean(2, keepdim=True)).expand(B,C,H,W)
        local_feat = self.branch_local(x)
        return x * self.sigmoid(self.fuse(h_feat + v_feat + local_feat))
