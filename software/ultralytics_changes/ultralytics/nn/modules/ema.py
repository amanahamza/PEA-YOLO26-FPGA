import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention.
    Branch A: grouped 3x3 conv + BN + SiLU  (local spatial context)
    Branch B: horizontal-vertical avg-pool + 1x1 conv  (global context)
    Gate: sigmoid(A + B) applied to input residually.
    Zero softmax -> INT8 friendly. Groups -> parallel DSP48 on FPGA.
    """
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        assert channels % groups == 0, \
            f"EMA: channels ({channels}) must be divisible by groups ({groups})"
        self.groups = groups

        # Branch A
        self.conv3x3 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=groups, bias=False
        )
        self.bn  = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

        # Branch B
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch A: local grouped conv
        a = self.act(self.bn(self.conv3x3(x)))

        # Branch B: global H-V pooling (broadcast addition, no softmax)
        pool_h = x.mean(dim=2, keepdim=True)   # B x C x 1 x W
        pool_v = x.mean(dim=3, keepdim=True)   # B x C x H x 1
        b = self.conv1x1(pool_h + pool_v)      # broadcasts to B x C x H x W

        # Residual gating
        return x * self.sigmoid(a + b)
