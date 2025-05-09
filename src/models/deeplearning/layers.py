import torch
from torch import Tensor
from torch.nn import Conv1d, Module


class GlobalAveragePooling1d(Module):
    """Performs global average pooling along spatial dimensions, i.e. takes
    an input x with shape (B, C, seq_length) and outputs (B, C)
    """

    def __init__(self, keepdim: bool = False):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=-1, keepdim=self.keepdim)


class SharedLinear(Module):
    """Just a wrapper around Conv1d with kernel_size=1.

    Expects input size (B, in_channels), outputs (B, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze().unsqueeze(-1)
        x = self.conv(x)
        x = x.squeeze()
        return x


class TargetDropout(Module):
    """Randomly drop out previous target information with probability p"""

    def __init__(self) -> None:
        super().__init__()
