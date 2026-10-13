import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFusion(nn.Module):
    """
    Lightweight bidirectional weighted fusion — adapted from ABF-YOLO Eq.7.

    Replaces simple Concat with learnable per-scale weighting:
        F_fusion = (w1*F1 + w2*F2 + ... ) / (sum(w) + eps)

    Uses ReLU+normalize instead of softmax for the weights
    -> FPGA/INT8 friendly (no exponential, no division by sum-of-exp).

    Takes a LIST of feature maps (possibly different channel counts
    and spatial sizes), aligns them with 1x1 convs + nearest resize,
    then fuses with learned weights.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        n = len(in_channels_list)
        # learnable fusion weights, one per input scale
        self.weights = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = 1e-4

        # 1x1 convs to align all inputs to same channel count
        self.align = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, bias=False)
            for c in in_channels_list
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, xs):
        # xs is a list of tensors (from Ultralytics Concat-style input)
        assert isinstance(xs, (list, tuple)), "BiFusion expects a list of inputs"

        # normalize weights (ReLU keeps non-negative, no softmax needed)
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)

        # use the first input's spatial size as target
        target_size = xs[0].shape[-2:]

        aligned = []
        for i, x in enumerate(xs):
            a = self.align[i](x)
            if a.shape[-2:] != target_size:
                a = F.interpolate(a, size=target_size, mode='nearest')
            aligned.append(w[i] * a)

        fused = aligned[0]
        for a in aligned[1:]:
            fused = fused + a

        return self.act(self.bn(fused))
