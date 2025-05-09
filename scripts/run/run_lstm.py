# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

import os
from argparse import ArgumentParser
from contextlib import nullcontext
from platform import system
from typing import Optional

from _pytest.capture import CaptureFixture
from pytorch_lightning import Trainer
from sklearn.model_selection import ParameterGrid

from src.constants import FULL_DATA, FOLD_TRAIN_SIDS, FOLD_VAL_SIDS, FOLD_PRED_SIDS
from src.models.deeplearning.arguments import (
    DataArgs,
    EvaluationArgs,
    GenericDeepLearningArgs,
    LstmArgs,
    PreprocArgs,
    ProgramArgs,
    WindowArgs,
)
from src.models.deeplearning.base import ModelEvaluator
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.lstm import LSTM

CC_CLUSTER = os.environ.get("CC_CLUSTER")

BATCH_SIZE = 1024
PBAR_REFRESH = max(int(800 / BATCH_SIZE), 1)  # every 100 samples seems about right

VAL_INTERVAL = 1 / 3 if CC_CLUSTER is not None else 1 / 60
PBAR = CC_CLUSTER is None
NUM_WORKERS = 0
if CC_CLUSTER == "graham":
    NUM_WORKERS = 0


def run() -> None:
    print(f"Loading subjects from {FULL_DATA}")
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    trainer_args = parser.parse_known_args()[0]

    if system() == "Windows":
        delattr(trainer_args, "gpus")

    config = ProgramArgs(
        progress_bar_refresh_rate=PBAR_REFRESH, trainer_args=trainer_args, enable_progress_bar=PBAR
    )
    preproc_args = PreprocArgs.default()
    window_args = WindowArgs.default(
        include_prev_target_as_predictor=True,
        desired_predictor_window_minutes=120,
        target_window_period_minutes=30,
        target_window_minutes = 12*60,
        decimation=500,
    )
    generic_args = GenericDeepLearningArgs.default(lr_init=3e-4)
    train_args = dict(
        predecimated=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        target_interpolation=InterpMethod.linear,
        target_dropout=0,
    )
    val_args = {
        **train_args,
        **dict(target_interpolation=InterpMethod.linear, target_dropout=0),
    }
    if CC_CLUSTER is None:
        train_loader_args = DataArgs.default(phase="train", subjects=FOLD_TRAIN_SIDS, **train_args)
        val_loader_args = DataArgs.default(phase="val", subjects=FOLD_VAL_SIDS, **val_args)
        pred_loader_args = DataArgs.default(phase="pred", subjects=FOLD_PRED_SIDS, **val_args)
    else:
        train_loader_args = DataArgs.default(phase="train", subjects=FOLD_TRAIN_SIDS, **train_args)
        val_loader_args = DataArgs.default(phase="val", subjects=FOLD_VAL_SIDS, **val_args)
        pred_loader_args = DataArgs.default(phase="pred", subjects=FOLD_PRED_SIDS, **val_args)

    eval_args = EvaluationArgs.default()

    model = LSTM
    model_args = LstmArgs.default(
        resize=None,
        num_layers=1,
        hidden_size=32,
        proj_size=0,
        num_linear_layers=1,
        use_full_seq=False,
        dropout=0,
    )
    evaluator = ModelEvaluator(
        config=config,
        model=model,
        model_args=model_args,
        generic_args=generic_args,
        preproc_args=preproc_args,
        window_args=window_args,
        train_loader_args=train_loader_args,
        val_loader_args=val_loader_args,
        pred_loader_args=pred_loader_args,
        eval_args=eval_args,
    )
    train_info = evaluator.train()
    evaluator.validate(train_info, keep_pickles=True)


if __name__ == "__main__":
    run()
