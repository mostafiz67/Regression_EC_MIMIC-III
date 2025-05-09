# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()  # isort: skip
# fmt: on

from argparse import ArgumentParser
from contextlib import nullcontext
from platform import system
from typing import Optional

import torch
from _pytest.capture import CaptureFixture
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.models.deeplearning.arguments import (
    Conv1dArgs,
    GenericDeepLearningArgs,
    LstmArgs,
    WindowArgs,
)
from src.models.deeplearning.conv1d import Conv1D

BATCH_SIZE = 512
PBAR_REFRESH = max(int(800 / BATCH_SIZE), 1)  # every 100 samples seems about right


def test_conv_shape_configs(capsys: Optional[CaptureFixture]) -> None:
    ctxt = nullcontext() if capsys is None else capsys.disabled()
    with ctxt:
        device = "cpu" if system() == "Windows" else "cuda"
        window_args = WindowArgs.default(
            include_prev_target_as_predictor=True,
        )
        generic_args = GenericDeepLearningArgs.default(lr_init=3e-4)
        grid = list(
            ParameterGrid(
                dict(
                    resize=[None, 128],
                    num_conv_layers=[1, 2, 4],
                    channel_expansion=[1, 2],
                    num_linear_layers=[1, 2, 4],
                    shared_linear=[True, False],
                    gap=[True, False],
                    dropout=[None, 0.2],
                    dropout_freq=[1, 2],
                    pooling=[True, False],
                    pooling_freq=[1, 2],
                )
            )
        )
        for args in tqdm(grid, total=len(grid)):
            model_args = Conv1dArgs.default(
                in_kernel_size=3,
                in_dilation=1,
                in_out_ch=16,
                kernel_size=3,
                dilation=1,
                depthwise=False,
                linear_width=4,
                mish=False,
                **args,
            )
            model = Conv1D(model_args, generic_args, window_args).to(device=device)
            x = torch.rand([2, *window_args.predictor_shape]).to(device=device)
            try:
                model(x)
            except Exception as e:
                print(args)
                raise e
        return


if __name__ == "__main__":
    test_conv_shape_configs(None)
