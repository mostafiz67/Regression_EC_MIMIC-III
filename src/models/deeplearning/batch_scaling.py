from __future__ import annotations

import gc
import traceback
from multiprocessing import Process
from typing import TYPE_CHECKING

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src._logging.base import LOGGER
from src.constants import ROOT
from src.models.deeplearning.arguments import (
    DataArgs,
    GenericDeepLearningArgs,
    PreprocArgs,
    ProgramArgs,
    WindowArgs,
)
from src.models.deeplearning.dataloader import WindowDataset

if TYPE_CHECKING:
    from src.models.deeplearning.base import BaseLightningModel

BATCH_FILE = ROOT / "batch_size.txt"


def _find_max_batch(
    model: BaseLightningModel,
    config: ProgramArgs,
    generic_args: GenericDeepLearningArgs,
    preproc_args: PreprocArgs,
    window_args: WindowArgs,
    train_loader_args: DataArgs,
    val_loader_args: DataArgs,
) -> None:
    """Double batch size until we get a memory error, then return the last working batch size."""

    # we only run this function in a subprocess since the errors can't be caught
    # this silences those errors messages from being logged
    # sys.stdout = open(os.devnull, "w")
    # sys.stderr = open(os.devnull, "w")
    N = 1
    dataset = WindowDataset(**{**train_loader_args.dataset_args, **window_args, **preproc_args})
    LOGGER.info("Finding maximum batch size")
    batch_size = 1
    with open(BATCH_FILE, "w") as handle:
        handle.write(f"{batch_size}")

    while batch_size < 1024:
        batch_size *= 2
        train_loader_args.batch_size.value = batch_size
        val_loader_args.batch_size.value = batch_size
        LOGGER.info(f"\tTesting batch_size={batch_size}")
        train_loader = DataLoader(dataset, **train_loader_args.loader_args)
        val_loader = DataLoader(dataset, **val_loader_args.loader_args)
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
        )
        try:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            with open(BATCH_FILE, "w") as handle:
                handle.write(f"{batch_size}")
            trainer = None
            train_loader = None
            val_loader = None
            gc.collect()
            torch.cuda.empty_cache()
        except:
            err = traceback.format_exc()
            if "CUDA out of memory" not in err:
                LOGGER.error(err)
            LOGGER.info(
                f"\tFailed at batch_size={batch_size}. Setting batch_size={batch_size // 2}"
            )
            with open(BATCH_FILE, "w") as handle:
                handle.write(f"{batch_size // 2}")
            return


def find_max_batch(
    model: BaseLightningModel,
    config: ProgramArgs,
    generic_args: GenericDeepLearningArgs,
    preproc_args: PreprocArgs,
    window_args: WindowArgs,
    train_loader_args: DataArgs,
    val_loader_args: DataArgs,
) -> int:
    kwargs = dict(
        model=model,
        config=config,
        generic_args=generic_args,
        preproc_args=preproc_args,
        window_args=window_args,
        train_loader_args=train_loader_args,
        val_loader_args=val_loader_args,
    )
    if BATCH_FILE.exists():
        BATCH_FILE.unlink(missing_ok=True)
    p = Process(target=_find_max_batch, kwargs=kwargs)
    p.start()
    p.join()
    with open(BATCH_FILE, "r") as handle:
        batch_size = int(handle.read().strip().replace("\n", ""))
    return batch_size
