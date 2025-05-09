from __future__ import annotations

import gc
import logging
import os
import pickle
import shutil
import traceback
import warnings
from abc import abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from functools import reduce
from math import ceil, floor
from pathlib import Path
from pickle import load as pickle_load
from time import time
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    no_type_check,
)
from uuid import UUID, uuid1
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    spearman_corrcoef,
)
from tqdm import tqdm
from typing_extensions import Literal

from src._logging.base import LOGGER
from src.constants import BASE_FREQUENCY, DL_LOGS, INTERP_THRESHOLD, LACT_MAX_MMOL
from src.constants import MILLIMOLAR_TO_MGDL as MGDL
from src.constants import ensure_dir
from src.models.deeplearning.arguments import (
    DataArgs,
    EvaluationArgs,
    GenericDeepLearningArgs,
    PreprocArgs,
    ProgramArgs,
    UnpackableArgs,
    WindowArgs,
)
from src.models.deeplearning.batch_scaling import find_max_batch
from src.models.deeplearning.callbacks import get_callbacks
from src.models.deeplearning.dataloader import TrainBatch, ValBatch, WindowDataset
from src.models.deeplearning.tables import merge_dfs
from src.models.deeplearning.utils import window_dimensions
from src.models.deeplearning.validation import TrainingInfo, ValidationResult, ValidationResults

Phase = Literal["train", "val", "test", "predict"]
PredictOutput = Any

PBAR = os.environ.get("CC_CLUSTER") is None


@dataclass
class Runtimes:
    n_train: int
    n_val: int
    train_batch: float
    train_epoch: float
    val_batch: float
    val_epoch: float
    pred_batch: float
    pred_epoch: float

    def __str__(self) -> str:
        lines = ["Estimated runtimes:\n"]
        lines.append(
            f"\t Train epoch:       {self.train_epoch / 3600:1.2f} hrs ({self.train_epoch / 60:1.2f} mins)"
        )
        lines.append(
            f"\t 1000 train batches: {self.train_batch * 1000 / 60:1.1f} mins ({self.train_batch * 1000:1.1f} s)"
        )
        lines.append(
            f"\t Val epoch:         {self.val_epoch / 3600:1.1f} hrs ({self.val_epoch / 60:1.2f} mins)"
        )
        lines.append(
            f"\t 1000 val batches:   {self.val_batch / 60:1.1f} mins ({self.val_batch * 1000:1.1f} s)"
        )
        lines.append(
            f"\t All predictions:   {self.pred_epoch / 3600:1.2f} hrs ({self.pred_epoch / 60:1.2f} mins)"
        )
        return "\n".join(lines)

    __repr__ = __str__


def to_pickle(object: Any, path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def window_loader(
    preproc_args: PreprocArgs, window_args: WindowArgs, data_args: DataArgs
) -> DataLoader:
    dataset = WindowDataset(**{**data_args.dataset_args, **window_args, **preproc_args})
    loader = DataLoader(dataset, **data_args.loader_args)
    return loader


def estimate_runtimes(
    config: ProgramArgs,
    model_cls: Type[BaseLightningModel],
    model_args: UnpackableArgs,
    generic_args: GenericDeepLearningArgs,
    preproc_args: PreprocArgs,
    window_args: WindowArgs,
    train_loader_args: DataArgs,
    val_loader_args: DataArgs,
    pred_loader_args: DataArgs,
    eval_args: EvaluationArgs,
) -> Runtimes:
    N = 50
    model = model_cls(model_args=model_args, generic_args=generic_args, window_args=window_args)

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning.core").setLevel(logging.ERROR)
    batch_size = find_max_batch(
        model,
        config,
        generic_args,
        preproc_args,
        window_args,
        train_loader_args,
        val_loader_args,
    )
    # generic_args.batch_size.value = batch_size
    train_loader_args.batch_size.value = batch_size
    val_loader_args.batch_size.value = batch_size
    pred_loader_args.batch_size.value = batch_size
    logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)
    logging.getLogger("lightning").setLevel(logging.DEBUG)
    logging.getLogger("pytorch_lightning.core").setLevel(logging.DEBUG)

    trainer: Trainer = Trainer.from_argparse_args(
        config.trainer_args.value,
        max_steps=N,
        val_check_interval=N,
        limit_val_batches=N,
        limit_predict_batches=N * 2,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        profiler=SimpleProfiler(),
    )

    LOGGER.info(
        "Setting up loaders for runtime estimation (can take a while when decimation <= 25)"
    )
    pred_loader_args.shuffle.value = True  # better for estimation
    val_loader_args.shuffle.value = True  # better for estimation
    train_loader = window_loader(preproc_args, window_args, train_loader_args)
    val_loader = window_loader(preproc_args, window_args, val_loader_args)
    pred_loader = window_loader(preproc_args, window_args, pred_loader_args)
    LOGGER.info("Done loader setup.")
    LOGGER.info(f"Estimating runtimes ({N} batches)")
    with catch_warnings():
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        warnings.simplefilter("ignore", UserWarning)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.predict(model, dataloaders=pred_loader)
        logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)
    pred_loader_args.shuffle.value = False
    val_loader_args.shuffle.value = False
    train_batch_times = trainer.profiler.recorded_durations["run_training_batch"]
    val_batch_times = trainer.profiler.recorded_durations["evaluation_step_and_end"]
    pred_batch_times = trainer.profiler.recorded_durations["predict_step"]

    train_batch = np.mean(train_batch_times)
    val_batch = np.mean(val_batch_times)
    pred_batch = np.mean(pred_batch_times)  # first can be like 100 times longer

    train_epoch = train_batch * len(train_loader)
    val_epoch = val_batch * len(val_loader)

    # If model training is slow, prediction will be extremely slow. In such cases, 24h - prediction_time
    # will be negative. In such cases, we need to disable the predict step.
    pred_epoch = pred_batch * len(pred_loader)
    return Runtimes(
        n_train=len(train_loader),
        n_val=len(val_loader),
        train_batch=train_batch,
        train_epoch=train_epoch,
        val_batch=val_batch,
        val_epoch=val_epoch,
        pred_batch=pred_batch,
        pred_epoch=pred_epoch,
    )


def get_run_params(
    runtimes: Runtimes, max_time: float = 22.0, val_interval_hrs: float = 0.5
) -> Tuple[Dict[str, Any], bool]:
    """Get train run arguments based on estimated runtimes, max allowed time, and validation frequency.

    Returns
    -------
    train_overrides: Dict[str, Any]

    do_predict: bool

    Notes
    -----
    Depending on `runtimes`, we need to adjust:

        --max_steps
        --max_epochs
        --val_check_interval
        --limit_val_batches

    Our unknowns are T (max_steps), V (val_check_interval). If set max_epochs=1000, we just worry
    about steps. `max_time` is based on SLURM / CCanada limits, e.g. 24 hours, minus a safety 2h.
    However, we also need to leave room for predictions and plotting. So total time budget really
    is max_time - runtimes.pred_epoch (note this ignore plotting which can easily be another hour).

    """
    budget_hrs = max_time - 1 - runtimes.pred_epoch / 3600
    do_predict = True
    if budget_hrs <= 1:
        # no time to run predictions
        budget_hrs = max_time
        do_predict = False

    max_steps = floor((budget_hrs * 3600) / runtimes.train_batch)
    max_epochs = min(floor((budget_hrs * 3600) / runtimes.train_epoch) + 1, 30)
    # how many steps it takes to be at roughly `val_interval_hrs`
    interval_steps = floor((val_interval_hrs * 3600) / runtimes.train_batch)
    # lightning tends to cut train steps out for the val check so we can ignore that adding issue
    # however lightning also requires val_check_interval <= max_steps
    val_check_interval = min(interval_steps, max_steps, runtimes.n_train)
    # however we need to be sure the val_epoch isn't itself inordinately long
    val_step = runtimes.train_batch + runtimes.val_batch  # combined for validation
    limit_val_batches = 1.0
    if (runtimes.n_val * runtimes.train_batch / 3600) > val_interval_hrs:
        # if validation takes longer than half time between vals
        # note we always just use train_batch because it is larger
        limit_val_batches = floor(((val_interval_hrs / 2) * 3600) / runtimes.train_batch)

    args = dict(
        max_steps=max_steps,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
    )

    val_duration = args["val_check_interval"] * runtimes.train_batch  # s
    if val_duration < 60:
        interval = f"{int(val_duration)} s"
    elif val_duration < 3600:
        interval = f"{int(val_duration / 60)} min"
    else:
        interval = f"{val_duration / 3600:1.1f} hrs"
    LOGGER.info(f"Training overrides: {args}")
    LOGGER.info(f"Validation check interval: {interval}")
    LOGGER.info(f"Validation batch limit: {limit_val_batches}")
    return args, do_predict


def predict_loaders(
    preproc_args: PreprocArgs,
    window_args: WindowArgs,
    data_args: DataArgs,
    eval_args: EvaluationArgs,
) -> Iterator[DataLoader]:
    sids = data_args.subjects.value
    batch = eval_args.subjects_per_batch.value
    limit = eval_args.limit_pred_subjects.value

    if isinstance(sids, tuple):
        sids = list(sids)
    if not isinstance(sids, list):
        raise TypeError(
            "Predict data_args must specify subject ids as either a list or tuple of strings (i.e. the sids)."
        )

    if limit is not None:
        sids = sids[:limit]
    # want at most `subjects_per_batch` in a pred batch
    d = ceil(len(sids) / batch)
    sid_groups = np.array_split(np.array(sids), d)
    for sids in sid_groups:
        d_args = {**data_args.dataset_args, **dict(subjects=sids)}
        dataset = WindowDataset(**{**d_args, **window_args, **preproc_args})
        loader = DataLoader(dataset, **data_args.loader_args)
        yield loader


class BaseLightningModel(LightningModule):
    @abstractmethod
    def __init__(
        self,
        model_args: UnpackableArgs,
        generic_args: GenericDeepLearningArgs,
        window_args: WindowArgs,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.generic_args = generic_args
        self.window_args = window_args
        # torch.multiprocessing.set_sharing_strategy("file_system")

    @abstractmethod
    def base_step(self, batch: TrainBatch, phase: Phase) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """This should be the ONLY method to implement in a new class.

        Parameters
        ----------
        batch: TrainBatch
            The training batch. This function must *always* operate on a type equivalent
            to TrainBatch, e.g. a Tuple of (x, y, interpolated_distances)

        phase: Literal["train", "val", "test", "predict"]
            Phase of model development.

        Returns
        -------
        preds: Tensor
            The resulting predictions which must have the same shape as the `target` below.

        target: Tensor
            The y_true values of the target. Likely just passed through from a `batch` argument.

        distances: Tensor
            The distances (in hours) of the interpolated y_true values from the nearest
            non-interpolated target value.

        loss: Tensor
            The result of `criterion(pred, target)`, where `preds` and `target` are as above.
            Different models might use different loss functions, so we can't just assume a
            default loss, and must return it in the abstract function.

        Notes
        -----
        TLDR: A new model should only need to define a `model.forward` and a `model.base_step`
        that returns a `Tuple[Tensor, Tensor, Tensor]` of `(pred, target, loss)`. Then all other
        steps and logging, metrics are automatically implemented for free.

        **Details**: The only really unique thing about a model is its `forward` implementation.
        Likewise, in the vast majority of cases, the training, validation, and testing procedures
        also really only need to passs a tensor through `model.forward` and then do something with
        the results.  Generally, the passing through `model.forward` is the same, and so the only
        thing that differs between model phases (training, validation, testing) is what is done with
        the results of the `model.forward` call. So the function `base_step` here just handles that
        shared part.

        In particular, all *predictive* models approaching a specific problem, no matter how
        unique the problem, still just need to produce a batch of predictions. Once how those
        predictions are produced depends on the model, but once created, they are all the same
        for a particular problem domain, and so the same basic approach can handle them.

        Thus, this `base_step` is often all that needs to be implemented in order to automatically
        infer training, validation, and test steps. This is how e.g. Tensorflow is able to
        automatically generate training, validation, testing procedures given only a `forward`,
        even though PyTorch Lightning does not.
        """

    def log_train_metrics(self, preds: Tensor, target: Tensor, distances: Tensor) -> None:
        mae = mean_absolute_error(preds, target)
        mse = mean_squared_error(preds * LACT_MAX_MMOL, target * LACT_MAX_MMOL)
        batch_size = preds.shape[0]
        n_sampled = self.global_step * batch_size
        self.log("train_loss", mse, prog_bar=False)
        self.log("train/mae (mmol/L)", mae * LACT_MAX_MMOL, prog_bar=False)
        self.log("train/mae (mg/dL)", mae * LACT_MAX_MMOL * MGDL, prog_bar=True)
        self.log("train/mse (mmol/L)^2", mse, prog_bar=False)
        self.log("train/N_sampled", n_sampled, prog_bar=True)

    def log_val_metrics(self, preds: Tensor, target: Tensor, distances: Tensor) -> None:
        mae = mean_absolute_error(preds, target)
        mse = mean_squared_error(preds * LACT_MAX_MMOL, target * LACT_MAX_MMOL)
        aes = torch.abs(preds - target)

        ae_max = torch.max(aes)
        p_corr = pearson_corrcoef(distances.ravel(), aes.ravel())

        self.log("val_loss", mse, prog_bar=False)  # for model checkpointing
        self.log("val/mae (mmol/L)", mae * LACT_MAX_MMOL, prog_bar=False)
        self.log("val/mae (mg/dL)", mae * LACT_MAX_MMOL * MGDL, prog_bar=True)
        self.log("val/mae_max (mg/dL)", ae_max * LACT_MAX_MMOL * MGDL, prog_bar=True)
        self.log("val/mse (mmol/L)^2", mse, prog_bar=False)
        self.log("val/pearson(d, mae)", p_corr, prog_bar=False)

        # NOTE: These are very
        # idx = distances < INTERP_THRESHOLD
        # non_interp = torch.mean(aes[idx])
        # we want to monitor if there is a correlation between interpolation
        # distance and error, but also an inflated version of this correlation
        # note we used spearman correlation previously, but this was too slow
        # ae_maxs = torch.max(aes, dim=1)[0]  # shape (B,)
        # d_mins = torch.min(distances, dim=1)[0]
        # max_corr = pearson_corrcoef(d_mins.ravel(), ae_maxs.ravel())
        # if idx.sum() > 1:
        #     p_corr_noninterp = pearson_corrcoef(distances[idx].ravel(), aes[idx].ravel())
        # else:
        #     p_corr_noninterp = torch.nan

        # self.log("val/mae (non-interp) (mg/dL)", non_interp * LACT_MAX_MMOL * MGDL, prog_bar=False)
        # self.log("val/pearson(d_min, mae_max)", max_corr, prog_bar=False)
        # self.log("val/pearson(d, mae) (non-interp)", p_corr_noninterp, prog_bar=False)

    @no_type_check
    def training_step(self, batch: TrainBatch, *args: Any, **kwargs: Any) -> Tensor:
        preds, target, distances, loss = self.base_step(batch, "train")
        self.log_train_metrics(preds, target, distances)
        return loss

    @no_type_check
    def validation_step(self, batch: ValBatch, *args: Any, **kwargs: Any) -> None:
        train_batch = batch[:3]
        preds, target, distances, loss = self.base_step(train_batch, "val")
        self.log_val_metrics(preds, target, distances)
        return loss

    def predict_step(
        self,
        batch: ValBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, Tuple[str, ...]]:
        """Get predictions for final plotting and analyses

        Returns
        -------
        preds: ndarray
            predicted values

        target: ndarray
            True values

        distances: ndarray
            Target distances from nearest non-interpolated point. Same shape as `target`

        y_hours: ndarray
            Target times.

        sids: Tuple[str, ...]
            Subject ids used for batch samples.
        """
        train_batch = batch[:3]
        preds, target, distances, loss = self.base_step(train_batch, "predict")
        x, target, distances, x_hours, y_hours, sids = batch
        # x.shape == (B, predictor_size, n_channels)
        # x_hours.shape == (B, predictor_size)
        # target.shape == (B, target_size)
        # distances.shape == (B, target_size) == target.shape
        # y_hours.shape == (B, target_size) == target.shape
        # preds.shape == (B, target_size) == target.shape

        return (
            preds.to(device="cpu").numpy().astype(np.float32),
            target.to(device="cpu").numpy().astype(np.float32),
            distances.to(device="cpu").numpy().astype(np.float32),
            y_hours.to(device="cpu").numpy().astype(np.float32),
            sids,
        )

    def configure_optimizers(self) -> Any:
        l2 = self.generic_args.weight_decay.value
        step = self.generic_args.lr_step.value
        gamma = self.generic_args.lr_decay.value
        lr = self.generic_args.lr_init.value
        opt = Adam(self.parameters(), weight_decay=l2, lr=lr)
        sched = StepLR(opt, step_size=step, gamma=gamma)
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=sched,
                interval="step",
            ),
        )


class ModelEvaluator:
    def __init__(
        self,
        config: ProgramArgs,
        model: Type[BaseLightningModel],
        model_args: UnpackableArgs,
        generic_args: GenericDeepLearningArgs,
        preproc_args: PreprocArgs,
        window_args: WindowArgs,
        train_loader_args: DataArgs,
        val_loader_args: DataArgs,
        pred_loader_args: DataArgs,
        eval_args: EvaluationArgs,
        profile: Optional[Literal["advanced", "pytorch"]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(model, LightningModule):
            raise TypeError("Argument to `model` must be the model class, not instance.")

        self.prev_uuid: Optional[str] = None
        self.uuid = uuid1().hex
        self.config = config
        self.model_cls = model
        self.model_args = model_args
        self.generic_args = generic_args
        self.eval_args = eval_args
        self.preproc_args = preproc_args
        self.window_args = window_args
        self.train_loader_args = train_loader_args
        self.val_loader_args = val_loader_args
        self.pred_loader_args = pred_loader_args
        self.do_predict: bool = True
        self.profile = profile
        self.ckpt = None

        self.all_metrics: Optional[DataFrame] = None
        self.train_metrics: Optional[DataFrame] = None
        self.val_metrics: Optional[DataFrame] = None
        self.lr_metrics: Optional[DataFrame] = None

        seed_everything(seed=None, workers=True)
        train_overrides = self.configure_loaders_to_runtimes()
        # fmt: off
        self.model = model(
            model_args=self.model_args,
            generic_args=self.generic_args,
            window_args=self.window_args
        )
        # fmt: on
        prev = "+prev_lact" if self.window_args.include_prev_target_as_predictor.value else ""
        root_dir = DL_LOGS / f"{model.__name__}{prev}"
        prof_args = dict(dirpath=DL_LOGS, filename=f"{model.__name__}_profile")
        if self.profile == "pytorch":
            profiler = PyTorchProfiler(**prof_args)
        elif self.profile == "advanced":
            profiler = AdvancedProfiler(**prof_args)
        else:
            profiler = None

        self.trainer: Trainer = Trainer.from_argparse_args(
            config.trainer_args.value,
            default_root_dir=root_dir,
            callbacks=get_callbacks(self.config),
            enable_progress_bar=PBAR,
            profiler=profiler,
            **train_overrides,
        )
        if self.trainer.log_dir is None:
            raise RuntimeError("No log directory found, logging was not set up correctly.")

        log_dir = Path(self.trainer.log_dir)
        if not log_dir.exists():
            log_dir.mkdir(exist_ok=True, parents=True)
            # raise RuntimeError("No log directory found, something went wrong.")
        with open(log_dir / "uuid.txt", "w") as handle:
            handle.write(f"{self.uuid}\n")
        self.save_hparams(log_dir)
        LOGGER.info(f"\n{self}\n")

    def configure_loaders_to_runtimes(self) -> Dict[str, Any]:
        if self.eval_args.estimate_runtime.value:
            runtimes = estimate_runtimes(
                config=self.config,
                model_cls=self.model_cls,
                model_args=self.model_args,
                generic_args=self.generic_args,
                preproc_args=self.preproc_args,
                window_args=self.window_args,
                train_loader_args=self.train_loader_args,
                val_loader_args=self.val_loader_args,
                pred_loader_args=self.pred_loader_args,
                eval_args=self.eval_args,
            )

            LOGGER.info(str(runtimes))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            train_overrides, do_predict = get_run_params(
                runtimes, val_interval_hrs=self.eval_args.val_interval_hrs.value
            )
            self.do_predict = do_predict

            # manual overrides
            limit = train_overrides["limit_val_batches"]
            train_args = self.config.trainer_args.value
            if limit != 1.0 or (train_args.limit_val_batches is not None):
                # with reduced validation size want a random mix of windows
                self.val_loader_args.preshuffle.value = True
            if train_args.limit_val_batches is not None:
                train_overrides["limit_val_batches"] = train_args.limit_val_batches
            if train_args.val_check_interval is not None:
                train_overrides["val_check_interval"] = train_args.val_check_interval
            if train_args.max_steps is not None:
                train_overrides["max_steps"] = train_args.max_steps
        else:
            train_overrides = dict()
            train_args = self.config.trainer_args.value
            if train_args.limit_val_batches is not None:
                # with reduced validation size want a random mix of windows
                self.val_loader_args.preshuffle.value = True
            if train_args.limit_predict_batches is None:
                self.do_predict = True
            else:
                self.do_predict = train_args.limit_predict_batches > 0
        return train_overrides

    def save_hparams(self, log_dir: Path) -> None:
        hparams_dir = ensure_dir(log_dir / "hparams")
        self.config.to_pickle(hparams_dir / "config.hparams.pickle")
        self.model_args.to_pickle(hparams_dir / "model.hparams.pickle")
        self.generic_args.to_pickle(hparams_dir / "generic.hparams.pickle")
        self.preproc_args.to_pickle(hparams_dir / "preproc.hparams.pickle")
        self.window_args.to_pickle(hparams_dir / "window.hparams.pickle")
        self.train_loader_args.to_pickle(hparams_dir / "train_loader.hparams.pickle")
        self.val_loader_args.to_pickle(hparams_dir / "val_loader.hparams.pickle")
        self.pred_loader_args.to_pickle(hparams_dir / "pred_loader.hparams.pickle")
        self.eval_args.to_pickle(hparams_dir / "eval.hparams.pickle")

    def train(self) -> TrainingInfo:
        train_loader = window_loader(self.preproc_args, self.window_args, self.train_loader_args)
        val_loader = window_loader(self.preproc_args, self.window_args, self.val_loader_args)
        train_sids = DataFrame(train_loader.dataset.sids)  # type: ignore
        val_sids = DataFrame(val_loader.dataset.sids)  # type: ignore
        logdir = Path(self.trainer.logger.experiment.log_dir)  # type: ignore
        train_sids.to_json(logdir / "train_sids.json")
        val_sids.to_json(logdir / "val_sids.json")

        filterwarnings("ignore", message="The dataloader", category=UserWarning)
        filterwarnings("ignore", message="This overload of nonzero", category=UserWarning)
        print(self.model)
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=self.ckpt,
        )
        self.tableify_logs()
        return TrainingInfo(self.trainer, self.train_loader_args)

    @no_type_check
    def validate(self, train_info: TrainingInfo, keep_pickles: bool = False) -> None:
        """Run the predict loop and get results.

        Parameters
        ----------
        train_info: TrainingInfo
            Result of evaluator.train(). Used to pass training info to plotting.

        keep_pickles: bool = False
            If True, keep prediction batch pickles around. Useful for testing only.
        """

        if not self.do_predict:
            LOGGER.error("Prediction/plotting will take too long. Skipping this step.")
            return
        filterwarnings("ignore", message="The dataloader", category=UserWarning)
        filterwarnings("ignore", message="This overload of nonzero", category=UserWarning)
        # NOTE: can very easily explode memory with this...
        # pred_loader = window_loader(self.preproc_args, self.window_args, self.pred_loader_args)
        logdir = Path(self.trainer.logger.experiment.log_dir)
        outdir = logdir / "pred_batches"
        pred_sids = DataFrame(self.pred_loader_args.subjects)
        pred_sids.to_json(logdir / "pred_sids.json")

        result_files: List[Path] = []
        for i, pred_loader in tqdm(
            enumerate(
                predict_loaders(
                    self.preproc_args, self.window_args, self.pred_loader_args, self.eval_args
                )
            ),
            desc="Prediction batch",
        ):
            outfile = outdir / f"results{i:02d}.pickle"
            results_path = outfile if outfile.exists() else self.validate_predict_batch(i, train_info, pred_loader)
            result_files.append(results_path)
        for i, path in enumerate(result_files):
            with open(path, "rb") as handle:
                results: ValidationResults = pickle_load(handle)
            LOGGER.info(f"Loaded validation batch data at {path}")
            LOGGER.info(f"Saving errors statistics for validation batch {i:02d}")
            results.add_to_summary_table()
            LOGGER.info(f"Plotting all predictions for validation batch {i:02d}")
            results.plot_all_preds()
            LOGGER.info(f"Plotting all prediction percentiles for validation batch {i:02d}")
            results.plot_all_percentiles()
            LOGGER.info(f"Plotting non-interpolated error dists for validation batch {i:02d}")
            results.plot_error_distributions(interpolated=False)
            results = None
            pred_loader = None
            if not keep_pickles:
                path.unlink()
                LOGGER.info(f"Removed validation batch data at {path}")
            gc.collect()

    @no_type_check
    def validate_predict_batch(
        self, pred_idx: int, train_info: TrainingInfo, pred_loader: DataLoader
    ) -> ValidationResults:
        """Run the predict loop and get results.

        Parameters
        ----------
        train_info: TrainingInfo
            Result of evaluator.train(). Used to pass training info to plotting.

        pickle: bool = False
            If True, pickle final ValidationResults. Useful for testing only.
        """
        # TODO: create val loaders per subject based on decimation to handle memory costs?
        # types start as below (B = batch_size)
        preds: Tuple[ndarray]  # preds.shape == (B, target_size) == target.shape
        targets: Tuple[ndarray]  # target.shape == (B, target_size)
        distances: Tuple[ndarray]  # distances.shape == (B, target_size) == target.shape
        y_hours: Tuple[ndarray]  # y_hours.shape == (B, target_size) == target.shape
        batched_sids: Tuple[Tuple[str, ...]]  # batched_sids.shape == (B,)

        pred_sids = DataFrame(pred_loader.dataset.sids)
        logdir = Path(self.trainer.logger.experiment.log_dir)
        outdir = logdir / "pred_batches"
        if not outdir.exists():
            outdir.mkdir(exist_ok=True)
        pred_sids.to_json(outdir / f"pred_sids{pred_idx:02d}.json")

        if self.ckpt:
            outputs: List[Any] = self.trainer.predict(
                model=self.model,
                dataloaders=pred_loader,
                ckpt_path=self.ckpt,
                return_predictions=True,
            )
        else:
            outputs: List[Any] = self.trainer.predict(
                dataloaders=pred_loader, ckpt_path="best", return_predictions=True
            )
        preds, targets, distances, y_hours, batched_sids = zip(*outputs)

        # overwrite returned values: save memory
        preds = np.concatenate(preds, axis=0).astype(np.float32)
        targets = np.concatenate(targets, axis=0).astype(np.float32)
        distances = np.concatenate(distances, axis=0).astype(np.float32)
        y_hours = np.concatenate(y_hours, axis=0).astype(np.float32)

        sids = []
        for sid_batch in batched_sids:
            sids.extend(sid_batch)
        all_sids = np.array(sids)

        uniq_sids = np.unique(sids).tolist()

        subj_idxs = []
        for sid in uniq_sids:
            subj_idxs.append(all_sids == sid)

        results: List[ValidationResult] = []
        for sid, idx in zip(uniq_sids, subj_idxs):
            results.append(
                ValidationResult(
                    sid=sid,
                    preds=preds[idx],
                    targets=targets[idx],
                    distances=distances[idx],
                    y_hours=y_hours[idx],
                )
            )
        result = ValidationResults(
            results=results,
            logdir=logdir,
            pred_idx=pred_idx,
            train_info=train_info,
            model_args=self.model_args,
            window_args=self.window_args,
            train_loader_args=self.train_loader_args,
            pred_loader_args=self.pred_loader_args,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            lr_metrics=self.lr_metrics,
        )
        outfile = outdir / f"results{pred_idx:02d}.pickle"
        LOGGER.info(f"Pickling validation batch {pred_idx:02d} results")
        to_pickle(result, outfile)
        LOGGER.info(f"Validation batch results {pred_idx:02d} at {outfile}")
        return outfile

    @classmethod
    def restore_from_ckpt(
        cls, uuid: UUID, last: bool = False
    ) -> Tuple[ModelEvaluator, TrainingInfo]:
        from src.models.deeplearning.arguments import (
            Conv1dArgs,
            ConvGamArgs,
            LinearArgs,
            LstmArgs,
            LstmGamArgs,
        )
        from src.models.deeplearning.conv1d import Conv1D
        from src.models.deeplearning.gam import (
            ConstrainedConvGam,
            ConstrainedLstmGam,
            ConvGam,
            LstmGam,
        )
        from src.models.deeplearning.linear import SimpleLinear
        from src.models.deeplearning.lstm import LSTM

        Model = Union[
            SimpleLinear, Conv1D, LSTM, ConvGam, ConstrainedConvGam, LstmGam, ConstrainedLstmGam
        ]
        MODELS: List[Model] = [
            SimpleLinear,
            Conv1D,
            LSTM,
            ConvGam,
            ConstrainedConvGam,
            LstmGam,
            ConstrainedLstmGam,
        ]
        ARGS: Dict[Model, UnpackableArgs] = {
            SimpleLinear: LinearArgs,
            Conv1D: Conv1dArgs,
            LSTM: LstmArgs,
            ConvGam: ConvGamArgs,
            ConstrainedConvGam: ConvGamArgs,
            LstmGam: LstmGamArgs,
            ConstrainedLstmGam: LstmGamArgs,
        }

        def matches(file: Path) -> UUID:
            with open(file, "r") as handle:
                return UUID(handle.readline().strip()) == uuid

        def load_pickle(file: Path) -> Any:
            with open(file, "rb") as handle:
                return pickle.load(handle)

        def get_model(log_dir: Path) -> Model:
            name = log_dir.parent.parent.name.split("+")[0]
            for model in MODELS:
                if model.__name__ == name:
                    return model
            raise ValueError(f"Could not find model matching {name} from {log_dir}")

        def load_hparams(log_dir: Path, model: Model) -> Namespace:
            hparams_dir = log_dir / "hparams"
            margs = ARGS[model]
            return Namespace(
                **dict(
                    config=ProgramArgs.from_pickle(hparams_dir / "config.hparams.pickle"),
                    model_args=margs.from_pickle(hparams_dir / "model.hparams.pickle"),
                    generic_args=GenericDeepLearningArgs.from_pickle(
                        hparams_dir / "generic.hparams.pickle"
                    ),
                    preproc_args=PreprocArgs.from_pickle(hparams_dir / "preproc.hparams.pickle"),
                    window_args=WindowArgs.from_pickle(hparams_dir / "window.hparams.pickle"),
                    train_loader_args=DataArgs.from_pickle(
                        hparams_dir / "train_loader.hparams.pickle", phase="train"
                    ),
                    val_loader_args=DataArgs.from_pickle(
                        hparams_dir / "val_loader.hparams.pickle", phase="val"
                    ),
                    pred_loader_args=DataArgs.from_pickle(
                        hparams_dir / "pred_loader.hparams.pickle", phase="pred"
                    ),
                    eval_args=EvaluationArgs.from_pickle(hparams_dir / "eval.hparams.pickle"),
                )
            )

        def load_metrics(log_dir: Path) -> Namespace:
            metrics = log_dir / "metrics"
            train = metrics / "train.json"
            val = metrics / "val.json"
            lr = metrics / "lr.json"
            _all = metrics / "all.json"
            return Namespace(
                **dict(
                    train=pd.read_json(train) if train.exists() else None,
                    val=pd.read_json(val) if val.exists() else None,
                    lr=pd.read_json(lr) if lr.exists() else None,
                    all=pd.read_json(_all) if _all.exists() else None,
                )
            )

        def move_pred_batches(log_dir: Path, new_log_dir: Path) -> None:
            batches = log_dir / "pred_batches"
            if batches.exists():
                if not new_log_dir.exists():
                    new_log_dir.mkdir(exist_ok=True, parents=True)
                shutil.move(str(batches), str(new_log_dir))

        uuids = list(filter(matches, DL_LOGS.rglob("uuid.txt")))
        if len(uuids) != 1:
            raise RuntimeError("A cosmic event has occurred.")
        log_dir = uuids[0].parent
        model_cls = get_model(log_dir)
        hparams = load_hparams(log_dir, model_cls)
        metrics = load_metrics(log_dir)
        config = hparams.config
        model_args = hparams.model_args
        generic_args = hparams.generic_args
        preproc_args = hparams.preproc_args
        window_args = hparams.window_args
        train_loader_args = hparams.train_loader_args
        val_loader_args = hparams.val_loader_args
        pred_loader_args = hparams.pred_loader_args
        eval_args = hparams.eval_args
        if last:
            ckpt = list(log_dir.rglob("last.ckpt"))[0]
        else:
            ckpt = list(filter(lambda p: "last" not in p.stem, log_dir.rglob("*.ckpt")))[0]
        kwargs = dict(model_args=model_args, generic_args=generic_args, window_args=window_args)
        model = model_cls.load_from_checkpoint(ckpt, **kwargs)
        LOGGER.info(f"Loaded model from checkpoint file {ckpt}")
        LOGGER.info(f"{model}")

        # setup the class, skipping some setup steps since we are already going
        new = cls.__new__(cls)
        new.prev_uuid = uuid.hex
        new.uuid = uuid1().hex
        new.config = config
        new.model_cls = model_cls
        new.model_args = model_args
        new.model = model
        new.generic_args = generic_args
        new.eval_args = eval_args
        new.preproc_args = preproc_args
        new.window_args = window_args
        new.train_loader_args = train_loader_args
        new.val_loader_args = val_loader_args
        new.pred_loader_args = pred_loader_args
        new.do_predict = True
        new.profile = None
        new.ckpt = ckpt

        new.all_metrics = metrics.all
        new.train_metrics = metrics.train
        new.val_metrics = metrics.val
        new.lr_metrics = metrics.lr

        prev = "+prev_lact" if new.window_args.include_prev_target_as_predictor.value else ""
        root_dir = DL_LOGS / f"{model_cls.__name__}{prev}"
        new.trainer = Trainer.from_argparse_args(
            config.trainer_args.value,
            default_root_dir=root_dir,
            callbacks=get_callbacks(new.config),
            enable_progress_bar=PBAR,
            resume_from_checkpoint=ckpt,
        )
        new_log_dir = Path(new.trainer.log_dir)
        if not new_log_dir.exists():
            new_log_dir.mkdir(exist_ok=True, parents=True)
        move_pred_batches(log_dir, new_log_dir)
        # Create a chain of uuid's that can be traversed. We have to do this because
        # of Tensorboard inadequacies, e.g. needs a new directory for a new run, and
        # cannot reuse an existing directory and add to it.
        with open(new_log_dir / "uuid.txt", "w") as handle:
            handle.write(f"{new.uuid}\n")
        with open(new_log_dir / "prev_uuid.txt", "w") as handle:
            handle.write(f"{new.prev_uuid}\n")
        new.save_hparams(new_log_dir)
        print(new.trainer.log_dir)
        LOGGER.info(f"\n{new}\n")
        return new, TrainingInfo(new.trainer, new.train_loader_args)

    def tableify_logs(self) -> None:
        # magic below from https://stackoverflow.com/a/45899735
        logdir = self.trainer.logger.experiment.log_dir  # type: ignore
        accum = EventAccumulator(logdir)
        accum.Reload()
        metric_names = [tag for tag in accum.Tags()["scalars"] if tag != "hp_metric"]
        all_metrics, lr_metrics, train_metrics, val_metrics, other_metrics = {}, {}, {}, {}, {}
        for metric in metric_names:
            walltimes, steps, values = zip(*accum.Scalars(metric))
            df = DataFrame({"wtime": walltimes, "step": steps, metric: values})
            all_metrics[metric] = df
            if "lr" in metric:
                lr_metrics[metric] = df
            elif "train" in metric:
                train_metrics[metric] = df
            elif "val" in metric:
                val_metrics[metric] = df
            else:
                other_metrics[metric] = df

        outdir = Path(logdir).resolve() / "metrics"
        outdir.mkdir(exist_ok=True, parents=True)

        if len(train_metrics) > 0:
            train = reduce(merge_dfs, train_metrics.values())
            train.to_json(outdir / "train.json")
            self.train_metrics = train
        if len(all_metrics) > 0:
            all = reduce(merge_dfs, all_metrics.values()).drop(columns="wtime")
            all.to_json(outdir / "all.json")
            self.all_metrics = all
        if len(lr_metrics) > 0:
            lr = reduce(merge_dfs, lr_metrics.values())
            lr.to_json(outdir / "lr.json")
            self.lr_metrics = lr
        if len(val_metrics) > 0:
            val = reduce(merge_dfs, val_metrics.values())
            val.to_json(outdir / "val.json")
            self.val_metrics = val
        LOGGER.debug(f"Metrics saved to {outdir}")

    def __str__(self) -> str:
        def tab(args: UnpackableArgs) -> str:
            s = str(args)
            lines = s.split("\n")
            return "\n".join([f"    {line}" for line in lines])

        lines = [f"{self.__class__.__name__} {self.uuid} with training args: {{"]
        lines.append(tab(self.model_args))
        lines.append(tab(self.generic_args))
        lines.append(tab(self.train_loader_args))
        lines.append(tab(self.preproc_args))
        lines.append(tab(self.window_args))
        lines.append("}")
        return "\n".join(lines)

    __repr__ = __str__
