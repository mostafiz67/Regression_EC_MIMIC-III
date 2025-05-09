# fmt: off
from audioop import bias

import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from copy import deepcopy
from math import floor, prod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import torch
from pytorch_lightning import LightningModule
from sklearn.model_selection import ParameterGrid
from torch import Tensor
from torch.nn import (
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
from torch.nn.functional import interpolate, l1_loss, mse_loss, softsign
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import mean_absolute_error
from typing_extensions import Literal

from src.constants import LACT_MAX_MMOL, LACT_MED_MMOL
from src.models.deeplearning.arguments import (
    Conv1dArgs,
    ConvGamArgs,
    GenericDeepLearningArgs,
    LinearArgs,
    LstmArgs,
    LstmGamArgs,
    SineLinearArgs,
    WindowArgs,
)
from src.models.deeplearning.base import BaseLightningModel, Phase, TrainBatch, ValBatch
from src.models.deeplearning.conv1d import Conv1D
from src.models.deeplearning.layers import GlobalAveragePooling1d, SharedLinear
from src.models.deeplearning.linear import SimpleLinear, SineLinear
from src.models.deeplearning.lstm import LSTM


class GAM(BaseLightningModel):
    """Implements a model:

        y = f(y_prev) + g(x) + b

    where:
        - y is the target to predict (length T)
        - x is the predictor wave (length p)
        - y_prev is previous target values (length p)
        - b is a bias (part of g in implementation)

    and
        - f: p -> T is a simple linear function
        - g: p -> T is modeled by some deep learner
    """

    def __init__(
        self,
        model_args: Union[ConvGamArgs, LstmGamArgs, SineLinearArgs],
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: Union[ConvGamArgs, LstmGamArgs, SineLinearArgs]
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        # ensure correct shapes are computed in conv model
        if self.window_args.predictor_shape[1] != 2:
            raise ValueError("")
        self.seq_length = self.window_args.predictor_shape[0]
        self.output_length = self.window_args.target_shape[0]
        self.g_window_args = deepcopy(window_args)
        self.g_window_args.include_prev_target_as_predictor.value = False
        # weights for weighted average of previous predictor
        # bias will be False because we want to leave that to g, and also
        # we want f(0) = 0 so that when there is no previous lactate (0)
        # that prediction degrades gracefully to `g(x) + median_lact`.
        self.f = Linear(self.seq_length, self.output_length, bias=False)
        self.g = None
        self.median = LACT_MED_MMOL / LACT_MAX_MMOL

    def forward(self, x: Tensor) -> Tensor:
        y_prev = x[:, :, 1]  # shape(B, S)
        x = x[:, :, 0, None]  # shape (B, S, 1)
        g = self.g(x)
        f = self.f(y_prev)
        return f + g + self.median

    def base_step(self, batch: TrainBatch, phase: Phase) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x, target, distances = batch
        preds = self(x)
        loss = mse_loss(preds, target)
        return preds, target, distances, loss


class ConstrainedGAM(GAM):
    def __init__(
        self,
        model_args: Union[ConvGamArgs, LstmGamArgs],
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: Union[ConvGamArgs, LstmGamArgs]
        super().__init__(model_args, generic_args, window_args, *args, **kwargs)
        seq_length = self.window_args.target_shape[0]
        self.scalar_bias = self.model_args.scalar_bias.value
        self.scalar_max = self.model_args.scalar_max.value
        bias_length = 1 if self.scalar_bias else seq_length
        max_length = 1 if self.scalar_max else seq_length
        self.g_max = Parameter(torch.full([max_length], self.median, dtype=torch.float32))
        self.g_bias = Parameter(torch.full([bias_length], 0, dtype=torch.float32))
        self.initialized = False

    def forward(self, x: Tensor) -> Tensor:
        if not self.initialized:
            self.g_max = self.g_max.to(device=x.device)
            self.g_bias = self.g_bias.to(device=x.device)
            self.initialized = True
        y_prev = x[:, :, 1]  # shape(B, S)
        x = x[:, :, 0, None]  # shape (B, S, 1)
        g = self.g(x)
        f = self.f(y_prev)
        g = self.g_max * softsign(g) + self.g_bias
        return f + g + self.median


class LinearGam(GAM):
    """g: p -> T is modeled by a Linear layer"""

    def __init__(
        self,
        model_args: LinearArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: LinearArgs
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        self.g = SimpleLinear(model_args, generic_args, self.g_window_args)

    def forward(self, x: Tensor) -> Tensor:
        y_prev = x[:, :, 1]  # shape(B, S)
        x = x[:, :, 0]  # shape (B, S)
        g = self.g(x)
        f = self.f(y_prev)
        return f + g + self.median


class SineLinearGam(GAM):
    """g: p -> T is modeled by a Linear layer"""

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
        self.g = SineLinear(model_args, generic_args, self.g_window_args)

    def forward(self, x: Tensor) -> Tensor:
        y_prev = x[:, :, 1]  # shape(B, S)
        x = x[:, :, 0].unsqueeze(1)  # shape (B, 1, S) for conv1d in SineLinear
        g = self.g(x)
        f = self.f(y_prev)
        return f + g + self.median


class ConvGam(GAM):
    """g: p -> T is modeled by a Conv1D CNN"""

    def __init__(
        self,
        model_args: ConvGamArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: ConvGamArgs
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        conv_args = self.model_args.get_g_args()
        self.g = Conv1D(conv_args, generic_args, self.g_window_args)


class LstmGam(GAM):
    """g: p -> T is modeled by an LSTM"""

    def __init__(
        self,
        model_args: LstmGamArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: LstmGamArgs
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        lstm_args = self.model_args.get_g_args()
        self.g = LSTM(lstm_args, generic_args, self.g_window_args)


class ConstrainedLstmGam(ConstrainedGAM):
    def __init__(
        self,
        model_args: LstmGamArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: LstmGamArgs
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        lstm_args = self.model_args.get_g_args()
        self.g = LSTM(lstm_args, generic_args, self.g_window_args)


class ConstrainedConvGam(ConstrainedGAM):
    def __init__(
        self,
        model_args: ConvGamArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: ConvGamArgs
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )
        conv_args = self.model_args.get_g_args()
        self.g = Conv1D(conv_args, generic_args, self.g_window_args)
