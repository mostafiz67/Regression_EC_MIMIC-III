# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from math import floor, prod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import torch
from pytorch_lightning import LightningModule
from sklearn.model_selection import ParameterGrid
from torch import Tensor
from torch.nn import (
    AvgPool1d,
    BatchNorm1d,
    Conv1d,
    Dropout2d,
    Flatten,
    LazyLinear,
    Linear,
    MaxPool1d,
    Mish,
    Module,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.nn.functional import interpolate, l1_loss, mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import mean_absolute_error
from typing_extensions import Literal

from src.models.deeplearning.arguments import (
    Conv1dArgs,
    GenericDeepLearningArgs,
    Pooling,
    WindowArgs,
)
from src.models.deeplearning.base import BaseLightningModel, Phase, TrainBatch, ValBatch
from src.models.deeplearning.layers import GlobalAveragePooling1d, SharedLinear


def padding_same_1d(input_shape: int, kernel: int, dilation: int) -> Tuple[int, int]:
    """Assumes symmetric (e.g. `(n, n, n)`) kernels, dilations, and stride=1

    Note that even kernels cause chaos since asymmetric padding is required.
    """

    def pad(k: int) -> Tuple[int, int]:
        p = max(k - 1, 0)
        p_top = p // 2
        p_bot = p - p_top
        return p_top, p_bot

    k = kernel + (kernel - 1) * (dilation - 1)  # effective kernel size
    return pad(k)


def conv1d_output_shape(
    input_size: int, kernel_size: int, padding: int, stride: int, dilation: int, output_padding=0
):
    """stolen from pytorch.test.quantization.test_quantized_op.py"""
    return (
        floor(
            (input_size + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1)) / stride
        )
        + 2 * output_padding
        + 1
    )


class Conv1D(BaseLightningModel):
    def __init__(
        self,
        model_args: Conv1dArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: Conv1dArgs
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        self.seq_length, self.input_channels = self.window_args.predictor_shape
        if self.model_args.resize.value is not None:
            self.seq_length = int(self.model_args.resize.value)
        self.output_length = self.window_args.target_shape[0]

        self.input = Conv1d(
            in_channels=self.input_channels,
            out_channels=self.model_args.in_out_ch.value,
            kernel_size=self.model_args.in_kernel_size.value,
            dilation=self.model_args.in_dilation.value,
            padding="same",
            padding_mode="reflect",
        )

        conv_layers: ModuleList = ModuleList()
        out_ch = in_ch = self.model_args.in_out_ch.value
        out_length = self.seq_length  # pooling will reduce
        for i in range(self.model_args.num_conv_layers.value):
            if i == 0:
                conv_layers.append(self.conv_group(in_ch, out_ch))
                continue
            pool = self.model_args.pooling.value and (i % self.model_args.pooling_freq.value == 0)
            drop = self.model_args.dropout.value and (i % self.model_args.dropout_freq.value == 0)
            depthwise = self.model_args.depthwise.value
            conv_args = dict(maxpool=pool, drop=drop, depthwise=depthwise)
            if pool:
                out_length = floor(((out_length - 2) / 2) + 1)  # see MaxPool1d docs
            if i % self.model_args.channel_expansion.value == 0:
                out_ch = min(out_ch * 2, self.model_args.max_channels.value)
                conv_layers.append(self.conv_group(in_ch, out_ch, **conv_args))
                in_ch = out_ch
            else:
                conv_layers.append(self.conv_group(in_ch, out_ch, **conv_args))

        gap = self.model_args.gap.value
        mish = self.model_args.mish.value
        reducer = GlobalAveragePooling1d() if gap else Flatten()

        output_layers = ModuleList([reducer])
        if gap:
            # output was (B, out_ch, self.seq_length) prior to GAP,
            # now (B, out_ch)
            lin_in_ch = out_ch
            output_layers.append(self._linear(lin_in_ch, self.output_length))
        else:
            # output was (B, out_ch, self.seq_length) prior to Flatten,
            # now (B, out_ch * self.seq_length)
            lin_in_ch = out_ch * out_length
            ch = self.model_args.linear_width.value
            output_layers.append(self._linear(lin_in_ch, ch))
            # negative `range` loop is same as range(0) i.e. noop
            for i in range(self.model_args.num_linear_layers.value - 1):
                output_layers.append(self._linear(ch, ch))
                output_layers.append(Mish(inplace=True) if mish else ReLU(inplace=True))
            output_layers.append(self._linear(ch, self.output_length))

        self.conv_layers = Sequential(*conv_layers)
        self.output_layer = Sequential(*output_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)  # conv1d expects (B, C, seq_length)
        if self.model_args.resize.value is not None:
            x = interpolate(x, self.model_args.resize.value)
        x = self.input(x)
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x

    def conv_group(
        self,
        in_channels: int,
        out_channels: int,
        maxpool: bool = False,
        drop: bool = False,
        depthwise: bool = False,
    ) -> Sequential:
        kernel = self.model_args.kernel_size.value
        dilation = self.model_args.dilation.value
        mish = self.model_args.mish.value
        ptype = self.model_args.pooling_type.value
        layers = [
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                dilation=dilation,
                groups=in_channels if depthwise else 1,
                padding="same" if self.model_args.pad.value else 0,
                padding_mode="reflect",
                bias=False,  # save compute since batch norm
            ),
            Mish(inplace=True) if mish else ReLU(inplace=True),
            BatchNorm1d(out_channels),
        ]
        if drop:
            layers.append(Dropout2d(self.model_args.dropout.value, inplace=True))
        if maxpool:
            layers.append(MaxPool1d(2, 2) if ptype is Pooling.Max else AvgPool1d(2, 2))

        return Sequential(*layers)

    def _linear(self, in_ch: int, out_ch: int) -> Union[LazyLinear, SharedLinear]:
        shared = self.model_args.shared_linear.value
        if shared:
            return SharedLinear(in_ch, out_ch)
        return Linear(in_ch, out_ch)

    def base_step(self, batch: TrainBatch, phase: Phase) -> Tuple[Tensor, Tensor, Tensor]:
        x, target, distances = batch
        preds = self(x)
        loss = mse_loss(preds, target)
        return preds, target, distances, loss
