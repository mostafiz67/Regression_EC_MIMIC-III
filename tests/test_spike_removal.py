# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from typing_extensions import Literal

from src.constants import DEV_SIDS
from src.models.deeplearning.containers.deepsubject import DeepSubject
from src.models.deeplearning.utils import best_rect
from src.preprocess.spikes import SpikeRemoval

if __name__ == "__main__":
    subjects = DeepSubject.initialize_sids_with_defaults(
        sids=DEV_SIDS["pred"], spike_removal=SpikeRemoval.Low
    )
    spiked = DeepSubject.initialize_sids_with_defaults(sids=DEV_SIDS["pred"], spike_removal=None)
    subject: DeepSubject
    spikey: DeepSubject
    sbn.set_style("darkgrid")
    for subject, spikey in tqdm(zip(subjects, spiked), total=len(subjects)):
        nrows, ncols = best_rect(len(subject.waves))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)
        for wave, spike, ax in zip(subject.waves, spikey.waves, axes.ravel()):
            ax.plot(spike.hours.numpy(), spike.values.numpy(), color="black", lw=0.5, alpha=0.5)
            ax.plot(wave.hours.numpy(), wave.values.numpy(), color="#ffa424", lw=0.5, alpha=0.8)
        fig.suptitle(f"{subject.sid}")
        fig.set_size_inches(w=20, h=16)
        fig.tight_layout()
        plt.show()
