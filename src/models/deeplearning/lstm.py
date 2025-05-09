from typing import Any, Tuple

import torch
from torch import Tensor
from torch.nn import LSTM as LstmLayer
from torch.nn import Flatten, Hardswish, Linear, Mish, Module, ModuleList, MSELoss, ReLU, Sequential
from torch.nn.functional import interpolate

from src.models.deeplearning.arguments import GenericDeepLearningArgs, LstmArgs, WindowArgs
from src.models.deeplearning.base import BaseLightningModel, Phase, TrainBatch
from src.models.deeplearning.layers import SharedLinear


class LstmGAP(Module):
    """Assumes input of shape (B, seq_length, D * lstm_out_ch). Can only be used
    if using full sequence. Outputs (B, D * lstm_out_ch)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=1)


class LSTM(BaseLightningModel):
    def __init__(
        self,
        model_args: LstmArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.model_args: LstmArgs
        super().__init__(
            model_args=model_args,
            generic_args=generic_args,
            window_args=window_args,
            *args,
            **kwargs
        )

        self.seq_length, self.in_channels = self.window_args.predictor_shape
        if self.model_args.resize.value is not None:
            self.seq_length = int(self.model_args.resize.value)
        self.target_size = self.window_args.target_shape[0]
        # see documentation for torch LSTM output shape in bidirectional case
        self.D = 2 if self.model_args.bidirectional.value else 1
        proj = self.model_args.proj_size.value
        self.lstm_out_channels = proj if proj > 0 else self.model_args.num_hidden.value
        self.lstm_outshape = (self.seq_length, self.D * self.lstm_out_channels)

        self.lstm = LstmLayer(
            input_size=self.in_channels,
            hidden_size=self.model_args.num_hidden.value,
            num_layers=self.model_args.num_layers.value,
            bidirectional=self.model_args.bidirectional.value,
            dropout=self.model_args.dropout.value,
            proj_size=self.model_args.proj_size.value,
            batch_first=True,
        )

        # cannot use gap if only using final hidden state
        gap = self.model_args.gap.value and self.model_args.use_full_seq.value
        self.reducer = LstmGAP() if gap else Flatten()

        # two cases to handle: if using the full sequence, the output
        #       o, (h, c) = self.lstm
        # is such that o.shape is (B, *self.lstm_outshape) = (B, seq_length, D*lstm_out_ch), and
        # this must be reduced with either a LstmGAP or Flatten layer. If using only the final
        # hidden state, a GAP makes no sense and we are just dealing with
        #       o[:, -1, :] == h[-1]  # shape == (B, D*lstm_out_ch)
        if gap or (not self.model_args.use_full_seq.value):
            linear_in_ch = self.D * self.lstm_out_channels
        else:
            linear_in_ch = self.seq_length * self.D * self.lstm_out_channels
        ch = self.model_args.linear_width.value

        linear_layers = ModuleList()
        shared = self.model_args.shared_linear.value
        linear = SharedLinear if shared else Linear
        if self.model_args.num_linear_layers.value <= 1:
            # just create the final linear layer needed to get the right output size
            linear_layers.append(linear(linear_in_ch, self.target_size, bias=True))
            self.output = Sequential(*linear_layers)
            return

        for i in range(self.model_args.num_linear_layers.value - 1):
            linear_layers.append(linear(linear_in_ch if i == 0 else ch, ch))
            linear_layers.append(
                Mish(inplace=True) if self.model_args.mish.value else ReLU(inplace=True)
            )
        linear_layers.append(linear(ch, self.target_size, bias=True))
        self.output = Sequential(*linear_layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.model_args.resize.value is not None:
            x = interpolate(x.permute(0, 2, 1), self.model_args.resize.value).permute(0, 2, 1)
        x = self.lstm(x)[0]
        if not self.model_args.use_full_seq.value:
            x = x[:, -1, :]
        x = self.reducer(x)
        x = self.output(x)
        return x

    def base_step(self, batch: TrainBatch, phase: Phase) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # x -> [N , L , Hin]
        # N -> batch size
        # L -> sequence length
        # Hin -> input_size (number of features)
        x, target, distances = batch
        preds = self(x)
        loss = MSELoss()(preds, target)
        return preds, target, distances, loss
