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

from src.constants import FULL_DATA, FOLD_TRAIN_SIDS, FOLD_PRED_SIDS, FOLD_VAL_SIDS
from src.models.deeplearning.arguments import (
    DataArgs,
    EvaluationArgs,
    GenericDeepLearningArgs,
    LinearArgs,
    PreprocArgs,
    ProgramArgs,
    WindowArgs,
)
from src.models.deeplearning.base import ModelEvaluator
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.linear import SimpleLinear

BATCH_SIZE = 32
PBAR_REFRESH = max(int(800 / BATCH_SIZE), 1)  # every 100 samples seems about right
PBAR = os.environ.get("CC_CLUSTER") is None


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
        target_window_period_minutes=30,  # 
        desired_predictor_window_minutes=120,
        target_window_minutes = 60*12,
        decimation=500,
    )
    generic_args = GenericDeepLearningArgs.default()
    train_args = dict(
        batch_size=BATCH_SIZE,
        num_workers=0,
        target_interpolation=InterpMethod.linear,
        target_dropout=0,
    )
    val_args = {
        **train_args,
        **dict(target_interpolation=InterpMethod.previous, target_dropout=0),
    }
    # train_loader_args = DataArgs.default("train", subjects="random", **train_args)
    # val_loader_args = DataArgs.default("val", subjects="random", **val_args)
    # pred_loader_args = DataArgs.default("pred", subjects="random", **val_args)
    train_loader_args = DataArgs.default(phase="train", subjects=FOLD_TRAIN_SIDS, **train_args)
    val_loader_args = DataArgs.default(phase="val", subjects=FOLD_VAL_SIDS, **val_args)
    pred_loader_args = DataArgs.default(phase="pred", subjects=FOLD_PRED_SIDS, **val_args)
    eval_args = EvaluationArgs.default(val_interval_hrs=2 / 60)
    model = SimpleLinear
    model_args = LinearArgs()
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
    evaluator.validate(train_info, keep_pickles=True) #Change True


if __name__ == "__main__":
    run()
