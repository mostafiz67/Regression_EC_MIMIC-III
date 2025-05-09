# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from math import prod
from typing import Any, Tuple

import torch
from sklearn.model_selection import ParameterGrid
from torch import Tensor
from torch.nn import Conv1d, Linear
from torch.nn.functional import mse_loss

from src.models.deeplearning.arguments import (
    GenericDeepLearningArgs,
    LinearArgs,
    SineLinearArgs,
    WindowArgs,
)
from src.models.deeplearning.base import BaseLightningModel, Phase, TrainBatch
from src.models.deeplearning.fourier import FourierEncoder1D


class SimpleLinear(BaseLightningModel):
    def __init__(
        self,
        model_args: LinearArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        self.input_size = prod(self.window_args.predictor_shape)
        self.output_size = self.window_args.target_shape[0]
        self.linear = Linear(self.input_size, self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def base_step(self, batch: TrainBatch, phase: Phase) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x, target, distances = batch
        preds = self(x.flatten(start_dim=1))
        loss = mse_loss(preds, target)
        return preds, target, distances, loss


class SineLinear(BaseLightningModel):
    def __init__(
        self,
        model_args: SineLinearArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: SineLinearArgs
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        self.seq_length, self.in_channels = self.window_args.predictor_shape
        if self.in_channels != 1:
            raise ValueError(
                "Fourier features can be extracted only for positions. In the "
                "1D case, positions are 1D, and so one channel."
            )
        self.output_size = self.window_args.target_shape[0]
        self.sine_features = self.model_args.sine_features.value
        self.trainable = self.model_args.trainable.value
        self.scale = self.model_args.scale.value
        self.linear_size = self.seq_length * self.sine_features
        # self.extractor = Conv1d(self.in_channels, self.sine_features, kernel_size=1, bias=True)
        self.extractor = FourierEncoder1D(
            in_channels=1,
            seq_length=self.seq_length,
            dim=self.sine_features,
            sigma=self.scale,
            trainable=self.trainable,
        )
        self.linear = Linear(self.linear_size, self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:  # batch size one
            x = x.unsqueeze(0)
        # x.shape == (B, C, S)
        x = self.extractor(x)  # will output shape (B, sine_feat, S)
        x = x.flatten(start_dim=1)
        return self.linear(x)

    def base_step(self, batch: TrainBatch, phase: Phase) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x, target, distances = batch  # x.shape == (B, seq_length, C)
        preds = self(x.permute(0, 2, 1))
        loss = mse_loss(preds, target)
        return preds, target, distances, loss


LINEAR_GRID = list(
    ParameterGrid(
        dict(
            desired_predictor_window_minutes=[5, 10, 30, 60, 120],
            lag_minutes=[1, 5, 30, 60, 120],
            target_window_minutes=[5, 10, 30, 60],
            target_window_period_minutes=[0, 1, 5],
            include_prev_target_as_predictor=[True, False],
            include_predictor_times=[False],
            decimation=[250, 500],  # if n_windows=6008 @ decimation=250, is 3004 @ decimation=500
        )
    )
)
"""WindowArgs to tune for linear model"""
