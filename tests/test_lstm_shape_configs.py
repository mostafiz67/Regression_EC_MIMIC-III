# fmt: off
import sys  # isort:skip
from pathlib import Path

from unicodedata import bidirectional  # isort:skip
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
from pytorch_lightning import Trainer
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.constants import FULL_DATA
from src.models.deeplearning.arguments import (
    DataArgs,
    GenericDeepLearningArgs,
    LstmArgs,
    PreprocArgs,
    ProgramArgs,
    WindowArgs,
)
from src.models.deeplearning.base import ModelEvaluator
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.lstm import LSTM

BATCH_SIZE = 512
PBAR_REFRESH = max(int(800 / BATCH_SIZE), 1)  # every 100 samples seems about right


def test_lstm_shape_configs(capsys: Optional[CaptureFixture]) -> None:
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
                    proj_size=[0, 6],
                    use_full_seq=[True, False],
                    bidirectional=[True, False],
                    num_linear_layers=[1, 2, 3],
                    linear_width=[16],
                    shared_linear=[True, False],
                    gap=[True, False],
                )
            )
        )
        for args in tqdm(grid, total=len(grid)):
            model_args = LstmArgs.default(
                num_layers=1,
                hidden_size=16,
                dropout=0,
                **args,
            )
            model = LSTM(model_args, generic_args, window_args).to(device=device)
            x = torch.rand([2, *window_args.predictor_shape]).to(device=device)
            try:
                model(x)
            except Exception as e:
                print(args)
                raise e
        return


if __name__ == "__main__":
    test_lstm_shape_configs(None)
