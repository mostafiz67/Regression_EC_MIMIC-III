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
from platform import system

from pytorch_lightning import Trainer

from src.constants import FULL_DATA
from src.models.deeplearning.arguments import (
    Conv1dArgs,
    DataArgs,
    EvaluationArgs,
    GenericDeepLearningArgs,
    PreprocArgs,
    ProgramArgs,
    WindowArgs,
)
from src.models.deeplearning.base import ModelEvaluator
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.conv1d import Conv1D

CC_CLUSTER = os.environ.get("CC_CLUSTER")
BATCH_SIZE = 128 if CC_CLUSTER is not None else 32
VAL_INTERVAL = 1 / 3 if CC_CLUSTER is not None else 1 / 60
PBAR_REFRESH = max(int(800 / BATCH_SIZE), 1)  # every 100 samples seems about right
PBAR = CC_CLUSTER is None
NUM_WORKERS = 4
if CC_CLUSTER == "graham":
    NUM_WORKERS = 3


def run_conv1d() -> None:
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
        target_window_period_minutes=30,
        decimation=125,
    )
    generic_args = GenericDeepLearningArgs.default()
    train_args = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        target_interpolation=InterpMethod.previous,
        target_dropout=0,
    )
    val_args = {
        **train_args,
        **dict(target_interpolation=InterpMethod.previous, target_dropout=0),
    }
    if CC_CLUSTER is None:
        train_loader_args = DataArgs.default("train", **train_args)
        val_loader_args = DataArgs.default("val", **val_args)
        pred_loader_args = DataArgs.default("pred", **val_args)
    else:
        train_loader_args = DataArgs.default("train", subjects="random", **train_args)
        val_loader_args = DataArgs.default("val", subjects="random", **val_args)
        pred_loader_args = DataArgs.default("pred", subjects="random", **val_args)

    eval_args = EvaluationArgs.default(val_interval_hrs=VAL_INTERVAL)

    model = Conv1D
    model_args = Conv1dArgs.default(
        resize=None,
        in_kernel_size=7,
        in_dilation=2,
        # in_out_ch=64,
        in_out_ch=32,
        kernel_size=3,
        dilation=1,
        depthwise=False,
        num_conv_layers=8,
        channel_expansion=2,
        max_channels=32,
        gap=True,
        mish=False,
        num_linear_layers=1,
        linear_width=256,
        shared_linear=True,
        dropout=None,
        dropout_freq=4,
        pooling=True,
        pooling_freq=2,
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
    evaluator.validate(train_info, keep_pickles=False)


if __name__ == "__main__":
    run_conv1d()
