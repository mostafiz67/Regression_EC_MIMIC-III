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
    ConvGamArgs,
    DataArgs,
    EvaluationArgs,
    GenericDeepLearningArgs,
    LinearArgs,
    LstmArgs,
    LstmGamArgs,
    Pooling,
    PreprocArgs,
    ProgramArgs,
    SineLinearArgs,
    WindowArgs,
)
from src.models.deeplearning.base import ModelEvaluator, estimate_runtimes
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.gam import (
    ConstrainedConvGam,
    ConstrainedLstmGam,
    ConvGam,
    LinearGam,
    LstmGam,
    SineLinearGam,
)
from src.preprocess.spikes import SpikeRemoval

CC_CLUSTER = os.environ.get("CC_CLUSTER")
BATCH_SIZE = 128 if CC_CLUSTER is not None else 1024
VAL_INTERVAL = 1 / 3 if CC_CLUSTER is not None else 1 / 60
PBAR_REFRESH = max(int(800 / BATCH_SIZE), 1)  # every 100 samples seems about right
PBAR = CC_CLUSTER is None
NUM_WORKERS = 4
if CC_CLUSTER == "graham":
    NUM_WORKERS = 3


def run_gam() -> None:
    print(f"Loading subjects from {FULL_DATA}")
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    trainer_args = parser.parse_known_args()[0]

    if system() == "Windows":
        delattr(trainer_args, "gpus")

    config = ProgramArgs(
        progress_bar_refresh_rate=PBAR_REFRESH, trainer_args=trainer_args, enable_progress_bar=PBAR
    )
    preproc_args = PreprocArgs.default(spike_removal=SpikeRemoval.Low)
    window_args = WindowArgs.default(
        include_prev_target_as_predictor=True,
        target_window_period_minutes=30,
        #
        decimation=500,
    )
    # generic_args = GenericDeepLearningArgs.default(lr_decay=0.65)
    generic_args = GenericDeepLearningArgs.default(batch_size=BATCH_SIZE)
    train_args = dict(
        predecimated=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        target_interpolation=InterpMethod.linear,
        target_dropout=0,
    )
    # For sure we actually likely want pred target_interpolation to be the same as training.
    # Otherwise we can't see that the predictions are correctly tracking toward future lactate
    # values.
    #
    # The mystery is that while training on e.g. InterpMethod.linear, if validating with
    # InterpMethod.previous, we see steady decreases in the validation error still. I can't quite
    # make sense of this, except perhaps that our val_mae (max) metric may be the reason. When
    # training with previous interpolation (which is harder to learn, and has more surprising)
    # discontinuities, the val_mae max falls slowly, and only after training loss is already quite
    # low. E.g. the model gets more confident in its predictions and the improvement in the
    # validation error is mostly due to reduction of maximum validation errors magnitude.
    #
    # However, when training with any continuous interpolation method, and validating on the *same*
    # continuous method, there are no more large jumps (or shouldn't be, generally) to cause huge
    # val_mae (max) values. And indeed in the training curves we see here val_mae (max) rapidly
    # falls to a much lower value than when val interpolation is "previous", and the val_max (max)
    # mostly stays constant, or even starts to increase slightly almost immediately (far before)
    # val_mae (max) did when val interpolation is discontinuous.
    #
    # So what interpolation to use in validation? IMO, it should be the same as in train and pred.
    # The machine can only really attempt to generalize the ABP -> interpolated lactate pattern
    # seen in training subjects to unseen subjects. The way to evaluate if this is happening is
    # to ensure the validation subjects have the same interpolation. *After* we still log a non-
    # interpolated error, and plot actual lactate positions, so we can still judge to what extent
    # good performance is because some interpolations are "easier" to fit, and what matches the
    # true lactate best.

    val_args = {
        **train_args,
        **dict(target_interpolation=InterpMethod.linear, target_dropout=0),
    }
    pred_args = {
        **train_args,
        **dict(target_dropout=0),
    }
    if CC_CLUSTER is None:
        train_loader_args = DataArgs.default("train", **train_args)
        val_loader_args = DataArgs.default("val", **val_args)
        pred_loader_args = DataArgs.default("pred", **pred_args)
    else:
        train_loader_args = DataArgs.default("train", subjects="random", **train_args)
        val_loader_args = DataArgs.default("val", subjects="random", **val_args)
        pred_loader_args = DataArgs.default("pred", subjects="random", **pred_args)

    eval_args = EvaluationArgs.default()

    # model = ConvGam
    model = LstmGam
    # model = ConstrainedLstmGam
    # model = ConstrainedConvGam
    # model = SineLinearGam
    # model = LinearGam

    if model is ConvGam:
        g_args = Conv1dArgs.default(
            resize=None,
            in_kernel_size=7,
            in_dilation=2,
            # in_out_ch=64,
            in_out_ch=32,
            kernel_size=3,
            dilation=1,
            depthwise=False,
            num_conv_layers=2,
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
            pooling_freq=1,
        )
        model_args = ConvGamArgs.default(**g_args)
    elif model is ConstrainedConvGam:
        g_args = Conv1dArgs.default(
            resize=256,
            in_kernel_size=7,
            in_dilation=2,
            # in_out_ch=64,
            in_out_ch=32,
            kernel_size=3,
            dilation=1,
            depthwise=False,
            pad=True,
            num_conv_layers=4,
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
            pooling_freq=1,
            pooling_type=Pooling.Avg,
        )
        model_args = ConvGamArgs.default(**g_args)
        # eval_args = EvaluationArgs.default(val_interval_hrs=VAL_INTERVAL, estimate_runtimes=False)
    elif model is LstmGam:
        generic_args = GenericDeepLearningArgs.default(lr=3e-4)
        g_args = LstmArgs.default(
            resize=None, #256
            num_layers=1,
            hidden_size=32,
            proj_size=0,
            num_linear_layers=1,
            use_full_seq=False,
            dropout=0,
        )
        model_args = LstmGamArgs.default(**g_args)
    elif model is ConstrainedLstmGam:
        generic_args = GenericDeepLearningArgs.default(lr=3e-4)
        g_args = LstmArgs.default(
            resize=None,
            num_layers=1,
            hidden_size=32,
            proj_size=0,
            num_linear_layers=1,
            use_full_seq=False,
            dropout=0,
        )
        model_args = LstmGamArgs.default(**g_args)
    elif model is SineLinearGam:
        generic_args = GenericDeepLearningArgs.default(lr_decay=0.65, batch_size=BATCH_SIZE)
        model_args = SineLinearArgs(sine_features=128, scale=1.0, trainable=True)
        eval_args = EvaluationArgs.default(val_interval_hrs=VAL_INTERVAL, estimate_runtimes=False)
    elif model is LinearGam:
        model_args = LinearArgs()
        generic_args = GenericDeepLearningArgs.default(lr=1e-2)
        eval_args = EvaluationArgs.default(val_interval_hrs=VAL_INTERVAL, estimate_runtimes=False)
    else:
        raise ValueError("Unrecognized additive model type.")
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
    evaluator.validate(train_info, keep_pickles=False) #


if __name__ == "__main__":
    run_gam()
