from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from torch.nn import Conv1d, Linear, Module
from torch.nn.parameter import Parameter
from typing_extensions import Literal

# see https://arxiv.org/pdf/2006.10739.pdf#subsection.6.1
# https://hackmd.io/@zhaiyuchen/cs4245


def basic_encoding(x: Tensor, t: Tensor, dim: int) -> Tensor:
    pass


def positional_encoding(x: Tensor, t: Tensor) -> Tensor:
    pass


# see also https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/demo.ipynb
# for guidance
def gaussian_encoding(x: Tensor, t: Tensor, dim: int) -> Tensor:
    """
    Parameters
    ----------
    x: Tensor
        x.shape = (B, C, *SPATIAL)
    t:

    """
    B = np.random.standard_normal([x.shape[1], dim])
    pass


class FourierEncoder1D(Module):
    """Encode a 1-channel, 1D input"""

    def __init__(
        self,
        in_channels: int,
        seq_length: int,
        dim: int = 64,
        sigma: int = 1,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if dim % 2 != 0 or dim <= 0:
            raise ValueError("`out_channels` must be even.")

        self.in_channels = in_channels
        self.seq_length = seq_length
        self.dim = dim
        self.sigma = sigma
        self.trainable = trainable
        dist = torch.distributions.normal.Normal(loc=0, scale=self.sigma)
        conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=dim, kernel_size=1, bias=False)
        self.t = (
            torch.arange(self.seq_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )  # shape (1, 1, S)
        if self.in_channels == 1:
            if not self.trainable:
                B = dist.sample([self.dim, self.in_channels, 1])
                conv.weight = Parameter(B, requires_grad=False)
                # self.in_channels == 1 here so M.size = self.out_channels
                fs = torch.sin(2 * torch.pi * self.conv(self.t))  # fs.shape (1, dim, S)
                self.F = Parameter(fs, requires_grad=False)
        self.conv = conv

        self.initialized = False

    def forward(self, x: Tensor) -> Tensor:
        """must"""
        B, C, S = x.shape
        if not self.initialized:
            self.t = self.t.to(device=x.device)
        if C == 1:
            if not self.trainable:  # no time info, construct it locally
                x = x * self.F  # x now has shape (B, M.size, S), as desired
                return x
            else:
                fs = torch.sin(2 * torch.pi * self.conv(self.t))  # fs.shape (1, dim, S)
                return x * fs

        if C == 2:
            t = torch.arange(S).unsqueeze(1)  # shape (S, 1)
            fs = torch.sin(torch.pi * (self.M.ravel() * t).T)  # shape (M.size, S)
            print("fs: ", fs.shape)  # shape (M.size, S)
            fs = fs.reshape(self.in_channels, self.dim, S)
            print("x: ", x.shape)
            print("fd reshaped: ", fs.shape)
            x = x[:, None] * torch.sin(fs)  # x.shape == (B, self.in_channels, self.dim, S)
            x = x.reshape(B, -1, S)
            return x

        return torch.matmul(self.M, x)


# class FourierEncoder1D(Module):
#     def __init__(self, in_channels: int, dim: int = 64, sigma: int = 1) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         if dim % 2 != 0 or dim <= 0:
#             raise ValueError("`out_channels` must be even.")
#         self.dim = dim
#         self.sigma = sigma
#         dist = torch.distributions.normal.Normal(loc=0, scale=self.sigma)
#         M = dist.sample([self.dim, self.in_channels])
#         self.M = torch.nn.parameter.Parameter(M, requires_grad=False)

#     def forward(self, x: Tensor) -> Tensor:
#         """must"""
#         B, C, S = x.shape
#         if C == 1:  # no time info, construct it locally
#             # self.in_channels == 1 here so M.size = self.out_channels
#             t = torch.arange(S).unsqueeze(1)  # shape (S, 1)
#             fs = torch.sin(torch.pi * (self.M.ravel() * t).T)  # shape (M.size, S)
#             # x has shape (B, 1, S), so
#             x = x * fs  # x now has shape (B, M.size, S), as desired
#             return x
#         if C == 2:
#             t = torch.arange(S).unsqueeze(1)  # shape (S, 1)
#             fs = torch.sin(torch.pi * (self.M.ravel() * t).T)  # shape (M.size, S)
#             print("fs: ", fs.shape)  # shape (M.size, S)
#             fs = fs.reshape(self.in_channels, self.dim, S)
#             print("x: ", x.shape)
#             print("fd reshaped: ", fs.shape)
#             x = x[:, None] * torch.sin(fs)  # x.shape == (B, self.in_channels, self.dim, S)
#             x = x.reshape(B, -1, S)
#             return x

#         return torch.matmul(self.M, x)


class SineLinear1D(Module):
    def __init__(self, in_channels: int, out_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = torch.sin(2 * torch.pi * x)
        return x


if __name__ == "__main__":
    x = torch.rand([2, 1, 100])
    sin = SineLinear1D(1, 4)
    print(sin(x))

    enc = FourierEncoder1D(in_channels=1, dim=4)
    enc(x)
    x = torch.rand([2, 2, 100])
    enc = FourierEncoder1D(in_channels=2, dim=4)
    print(enc(x).shape)
