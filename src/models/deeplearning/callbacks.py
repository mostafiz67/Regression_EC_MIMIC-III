import os
from typing import List

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
    TQDMProgressBar,
)

from src.models.deeplearning.arguments import ProgramArgs

MAX_TIME = "22:00:00:00"  # "DD:HH:MM:SS", D = days, H = hours, M = minutes, S = seconds

SLURM_TMPDIR = os.environ.get("SLURM_TMPDIR")
SHOW_PBAR = os.environ.get("CC_CLUSTER") is None


# see https://github.com/PyTorchLightning/pytorch-lightning/issues/8126#issuecomment-935923373
class UninterruptableModelCheckpoint(ModelCheckpoint):
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_checkpoint(trainer)

    def on_keyboard_interrupt(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_checkpoint(trainer)

SLURM_TMPDIR = os.environ.get("SLURM_TMPDIR")
SHOW_PBAR = os.environ.get("CC_CLUSTER") is None


def get_callbacks(args: ProgramArgs) -> List[Callback]:
    # NOTE: Too early to implement early stopping. Still need to see some full training curves.
    # However, commented our arguments are roughly reasonable starting points.
    pbar_args = (
        dict(refresh_rate=args.progress_bar_refresh_rate.value)
        if hasattr(args, "progress_bar_refresh_rate")
        else dict()
    )
    cbs = [
        LearningRateMonitor(logging_interval="epoch"),
        # EarlyStopping(
        #     monitor="val/mae (mmol/L)",
        #     min_delta=0.25 * (1 / MILLIMOLAR_TO_MGDL),  # improvement of quarter mg/dL not meaningful
        #     patience=10,
        #     mode="min",
        #     check_finite=True,
        #     check_on_train_epoch_end=True,
        # ),
        ModelCheckpoint(
            filename="{epoch}_{val_loss:1.3f}",
            monitor="val_loss",
            save_last=True,
            save_top_k=1,
            mode="min",
            auto_insert_metric_name=True,
            save_weights_only=False,
            every_n_epochs=1,
        ),
        # UninterruptableModelCheckpoint(
        #     filename="{epoch}_{train_loss:1.3f}",
        #     monitor="train_loss",
        #     save_last=True,
        #     save_top_k=1,
        #     mode="min",
        #     auto_insert_metric_name=True,
        #     save_weights_only=False,
        #     every_n_epochs=1,
        # ),
        Timer(duration=MAX_TIME, interval="step"),  # prevent overly-long training
        TQDMProgressBar(**pbar_args) if SHOW_PBAR else None,
    ]
    return [cb for cb in cbs if cb is not None]
