import matplotlib as mpl

mpl.use("Agg")

import matplotlib.style as mplstyle

mplstyle.use("fast")
mpl.rcParams["path.simplify_threshold"] = 1.0
mpl.rcParams["path.simplify"] = True
mpl.rcParams["agg.path.chunksize"] = 10000


# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()  # isort: skip
# fmt: on

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from pandas import Series
from tqdm import tqdm

from src.constants import (
    BASE_FREQUENCY,
    CLINICAL_DB,
    DEV_RAND_VARIABLE_SIDS,
    FULL_DATA,
    MEMORY,
    MILLIMOLAR_TO_MGDL,
)
from src.constants import PREDICTOR_MEDIAN_OF_IQRS as IQR
from src.constants import PREDICTOR_MEDIAN_OF_MEDIANS as MED
from src.models.deeplearning.containers.deepsubject import DeepSubject
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.utils import best_rect, window_dimensions
from src.preprocess.containers.wave import Wave

SIDS = [
    "p000188",
    "p000377",
    "p000439",
    "p000491",
    "p000515",
    "p000618",
    "p000625",
    "p000652",
    "p000638",  # weird
    # "p000668",
    # "p000695",
    # "p000719",
]
PREDICTOR_WIN_MINS = 120
DECIMATION = 250
RSIZE = 125  # size for rolling_avg, max and min
PATIENTS_INFO = CLINICAL_DB / "PATIENTS.csv.gz"


def plot_rolling_info(subject: DeepSubject, ax: plt.Axes) -> None:
    RSIZE = (5 * 60 * BASE_FREQUENCY) // subject.decimation
    DROP = ["count", "50%"]
    SORT = ["mean", "min", "max", "std"]
    waves = [wave.values.numpy().astype(np.float32) * IQR + MED for wave in subject.waves]
    wave = pd.Series(np.concatenate(waves), dtype=np.float32)
    roll = wave.rolling(RSIZE)
    rmax, rmin = roll.max(), roll.min()
    desc = wave.describe(percentiles=[]).drop(columns=DROP).loc[SORT].to_dict()
    rmax_desc = rmax.describe(percentiles=[]).drop(columns=DROP).loc[SORT].to_dict()
    rmin_desc = rmin.describe(percentiles=[]).drop(columns=DROP).loc[SORT].to_dict()

    desc_all = r"ABP : $\mu$={:>5.2f},[{:>5.2f}, {:>5.2f}] ({:>5.2f})".format(*desc.values())
    desc_max = r"rmax: $\mu$={:>5.2f},[{:>5.2f}, {:>5.2f}] ({:>5.2f})".format(*rmax_desc.values())
    desc_min = r"rmin: $\mu$={:>5.2f},[{:>5.2f}, {:>5.2f}] ({:>5.2f})".format(*rmin_desc.values())
    info = "\n".join((desc_all, desc_max, desc_min))
    font_args = dict(fontsize=6, fontfamily="monospace", verticalalignment="top")
    bbox = dict(facecolor="white", alpha=0.7, edgecolor="white")
    ax.text(x=0.02, y=0.98, s=info, transform=ax.transAxes, bbox=bbox, **font_args)


def plot_waves(subject: DeepSubject, i: int, ax: plt.Axes) -> None:
    PALETTE = sns.color_palette(palette="colorblind", n_colors=10)
    BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK, GREY, YELLOW, TEAL = PALETTE
    RSIZE = (5 * 60 * BASE_FREQUENCY) // subject.decimation

    for w, wave in enumerate(subject.waves):
        hrs = wave.hours
        vals = Series(wave.values.numpy().astype(np.float32) * IQR + MED)
        roll = vals.rolling(RSIZE)
        rmax, rmin, rmed = roll.max(), roll.min(), roll.median()
        line_args = dict(lw=0.5, alpha=0.7)
        label = (i == 0) and (w == 0)
        ax.plot(hrs, rmax, label="rolling max" if label else None, color=RED, **line_args)
        ax.plot(hrs, rmin, label="rolling min" if label else None, color=GREEN, **line_args)
        ax.plot(hrs, rmed, label="rolling med" if label else None, color=BLUE, **line_args)


def plot_lactate(subject: DeepSubject, i: int, ax: plt.Axes) -> None:
    lact_hrs = subject.lactate.hours
    h0 = subject.waves[0].hours_0
    hf = subject.waves[-1].hours_f
    idx = (lact_hrs > h0) & (lact_hrs < hf)
    lact_hrs = lact_hrs[idx]
    lact_vals = subject.lactate.values[idx] * MILLIMOLAR_TO_MGDL
    ax2 = ax.twinx()
    label = "lactate (mg/dL)" if (i == 0) else None
    if len(lact_vals) == 0:
        lact_0 = subject.lactate.interpolator.predict(np.array(h0))
        lact_f = subject.lactate.interpolator.predict(np.array(hf))
        ax2.scatter(h0, lact_0, s=3.5, color="black", marker="<", label=label)
        ax2.scatter(hf, lact_f, s=3.5, color="black", marker=">", label=label)
    else:
        ax2.scatter(lact_hrs, lact_vals, s=3.5, color="black", label=label)
    t = np.linspace(h0, 1.1*hf, 200)
    lact_interp = subject.lactate.interpolator.predict(t) * MILLIMOLAR_TO_MGDL
    ax2.scatter(t, lact_interp, s=3.5, color="black", alpha=0.3)
    ax2.set_ylim(0, 150)


def plot_subject(subject: DeepSubject, i: int, ax: plt.Axes, extend_to_dod: bool) -> None:
    # plot rolling / smoothed info and summaries
    plot_rolling_info(subject, ax)
    plot_waves(subject, i, ax)
    plot_lactate(subject, i, ax)
    dod_hr = subject.demographics.dod_hrs  # plot time of death as red line, if in data
    ax.set_title(subject.demographics.info(), fontdict={"fontsize": 10})
    if dod_hr is None:
        return
    if extend_to_dod:
        ax.axvline(dod_hr, color="red", label="TOD" if i == 0 else None)
        if ax.get_xlim()[1] < dod_hr:
            ax.set_xlim(0, dod_hr + 5)
        return
    if dod_hr <= subject.waves[-1].hours_f:
        ax.axvline(dod_hr, color="red", label="TOD" if i == 0 else None)


@MEMORY.cache()
def get_runs(
    sids: List[str] = None, limit: int = None, interp: InterpMethod = InterpMethod.linear
) -> List[DeepSubject]:
    subjects = DeepSubject.initialize_sids_with_defaults(sids=sids, interp_method=interp)
    if limit is not None:
        subjects = subjects[:limit]
    runs = []
    for s in tqdm(subjects, total=len(subjects), desc="Splitting subjects into runs"):
        runs.extend(s.split_at_gaps(max_gap_hrs=48))
    return runs


@MEMORY.cache()
def get_shorter_runs(
    sids: List[str] = None,
    limit: int = None,
    interp: InterpMethod = InterpMethod.linear,
    zombies_only: bool = False,
) -> List[DeepSubject]:
    PRED_MIN = 30
    DEC = 500
    TARG_MIN = 6 * 60  # 6 hours
    subjects: DeepSubject = DeepSubject.initialize_sids_with_defaults(
        sids=sids,
        predictor_window_size=window_dimensions(PRED_MIN, DEC)[0],
        lag_minutes=0,
        target_window_minutes=TARG_MIN,
        decimation=DEC,
        interp_method=interp,
    )
    if zombies_only:
        subjects = [s for s in subjects if DeepSubject.is_zombie(s)]
    if limit is not None:
        subjects = subjects[:limit]
    runs = []
    for s in tqdm(subjects, total=len(subjects), desc="Splitting subjects into runs"):
        runs.extend(s.split_at_gaps(max_gap_hrs=48))
    return runs


def plot_all_subjects(sids: List[str] = None) -> None:
    runs = get_runs(sids, limit=None, interp=InterpMethod.linear)
    nrows, ncols = best_rect(len(runs))
    plt.ioff()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, squeeze=False)
    subject: DeepSubject
    ax: plt.Axes
    for i, (ax, subject) in tqdm(
        enumerate(zip(axes.ravel(), runs)),
        total=len(runs),
        desc="Plotting",
        dynamic_ncols=True,
    ):
        plot_subject(subject, i, ax, extend_to_dod=False)
    axes.ravel()[0].set_ylim(-5, 300)
    fig.suptitle("Rolling window size: 5 min")
    fig.set_size_inches(w=4 * ncols, h=1 + 4 * nrows)
    fig.legend(loc="lower right")
    fig.tight_layout()
    fig.text(s="Time since first measurement (hrs)", x=0.42, y=0.02)
    fig.text(s="ABP (mm/Hg)", x=0.02, y=0.46, rotation="vertical")
    fig.subplots_adjust(top=0.934, bottom=0.07, left=0.06, right=0.968, hspace=0.17, wspace=0.206)

    if sids is None:
        outfile = ROOT / "abp_lact_ALL.png"
    else:
        if len(sids) == 1:
            outfile = ROOT / f"abp_lact_{sids[0]}.png"
        else:
            outfile = ROOT / f"abp_lact{sids[0]}-{sids[-1]}.png"

    fig.savefig(outfile, dpi=200)
    print(f"Saved plot to: {outfile}")
    # plt.show()
    plt.close()


def plot_zombies(sids: List[str] = None) -> None:
    runs = get_shorter_runs(sids, limit=None, interp=InterpMethod.linear, zombies_only=True)
    nrows, ncols = best_rect(len(runs))
    plt.ioff()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, squeeze=False)
    subject: DeepSubject
    ax: plt.Axes
    for i, (ax, subject) in tqdm(
        enumerate(zip(axes.ravel(), runs)),
        total=len(runs),
        desc="Plotting",
        dynamic_ncols=True,
    ):
        plot_subject(subject, i, ax, extend_to_dod=True)
    axes.ravel()[0].set_ylim(-5, 300)
    fig.suptitle("Rolling window size: 5 min")
    fig.set_size_inches(w=4 * ncols, h=1 + 4 * nrows)
    fig.legend(loc="lower right")
    fig.tight_layout()
    fig.text(s="Time since first measurement (hrs)", x=0.42, y=0.02)
    fig.text(s="ABP (mm/Hg)", x=0.02, y=0.46, rotation="vertical")
    fig.subplots_adjust(top=0.934, bottom=0.07, left=0.06, right=0.968, hspace=0.17, wspace=0.206)

    if sids is None:
        outfile = ROOT / "zombies.png"
    else:
        if len(sids) == 1:
            outfile = ROOT / f"zombies_{sids[0]}.png"
        else:
            outfile = ROOT / f"zombies_{sids[0]}-{sids[-1]}.png"

    fig.savefig(outfile, dpi=200)
    print(f"Saved plot to: {outfile}")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    NORMAL_ZOMBIES = sorted(["p001885", "p003474", "p021219", "p023591", "p013715", "p064965"])
    STRANGE_ZOMBIES = sorted(["p056963", "p049613", "p047858", "p095957", "p053865", "p095776"])
    STRANGE_SUBJECTS = sorted([f"p0{sid}" for sid in [22322, 22393, 28611, 69857, 96879]])

    # plot_all_subjects(DEV_RAND_VARIABLE_SIDS["train"])
    # plot_all_subjects()
    # plot_zombies(NORMAL_ZOMBIES)
    # plot_zombies(STRANGE_ZOMBIES)
    plot_all_subjects(STRANGE_SUBJECTS)
