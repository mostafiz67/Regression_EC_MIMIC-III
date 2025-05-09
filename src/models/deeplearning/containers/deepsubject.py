from __future__ import annotations

import os
import traceback
from concurrent.futures import process
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from math import ceil, floor
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, Type
from warnings import filterwarnings

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import interpolate
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src._logging.base import LOGGER
from src.constants import BASE_FREQUENCY, MEMORY, ON_NIAGARA, PREDECIMATED_DATA
from src.constants import PREDICTOR_MEDIAN_OF_IQRS as IQR
from src.constants import PREDICTOR_MEDIAN_OF_MEDIANS as MD
from src.constants import ROOT
from src.models.deeplearning.arguments import WindowArgs
from src.models.deeplearning.containers.demographics import Demographics
from src.models.deeplearning.containers.lactate import InterpMethod, Lactate
from src.models.deeplearning.utils import (
    decimated_folder,
    validate_predecimation,
    window_dimensions,
)
from src.models.deeplearning.windows import get_wave_endpoints
from src.preprocess.containers.wave import Wave
from src.preprocess.spikes import SpikeRemoval

WaveStart = Tuple[Wave, int]

MAX_WORKERS = None
CC_CLUSTER = os.environ.get("CC_CLUSTER")
if CC_CLUSTER == "siku":
    MAX_WORKERS = 8

@dataclass
class WaveSlicer:
    wave: Wave
    start: int
    end: int


def count_usable_windows(
    wave: Wave,
    lactate: Lactate,
    predictor_window_size: int,
    lag_minutes: float,
    target_window_minutes: float,
    decimation: int,
) -> int:
    """
    Returns
    -------
    count: int
        How many windows are usable.
    """
    T = float(decimation / (BASE_FREQUENCY * 3600))  # in hours
    W = predictor_window_size  # count
    W_hours = (W - 1) * T
    lag_hrs = lag_minutes / 60
    target_hrs = target_window_minutes / 60
    x_start, x_end = wave.hours_0, wave.hours_f
    y_start, y_end = lactate.hours[0], lactate.hours[-1]

    # Degenerate cases
    if x_end <= y_start or y_end <= x_start:  # no valid predictor windows
        return 0
    if W > len(wave):  # predictor window larger than x
        return 0
    if wave.hours_at(W - 1) > y_end:  # first window already off of y
        return 0
    if y_start + W_hours > x_end:
        # first x window can only be as early as y_hours[0], but in this
        # case not enough room in x for window
        return 0

    S = max(ceil((y_start - x_start) / T), 0)
    F = min(floor((y_end - x_start) / T) - W - 1, len(wave) - W - 1)
    if F < S:
        return 0
    N = ceil((lag_hrs + target_hrs) / T)
    F_lag = F - N
    if F_lag < S:
        return 0
    return F_lag - S


def count_usable_windows(
    wave: Wave,
    lactate: Lactate,
    predictor_window_size: int,
    lag_minutes: float,
    target_window_minutes: float,
    decimation: int,
) -> int:
    """
    Returns
    -------
    count: int
        How many windows are usable.
    """
    T = float(decimation / (BASE_FREQUENCY * 3600))  # in hours
    W = predictor_window_size  # count
    W_hours = (W - 1) * T
    lag_hrs = lag_minutes / 60
    target_hrs = target_window_minutes / 60
    x_start, x_end = wave.hours_0, wave.hours_f
    y_start, y_end = lactate.hours[0], lactate.hours[-1]

    # Degenerate cases
    if x_end <= y_start or y_end <= x_start:  # no valid predictor windows
        return 0
    if W > len(wave):  # predictor window larger than x
        return 0
    if wave.hours_at(W - 1) > y_end:  # first window already off of y
        return 0
    if y_start + W_hours > x_end:
        # first x window can only be as early as y_hours[0], but in this
        # case not enough room in x for window
        return 0

    S = max(ceil((y_start - x_start) / T), 0)
    F = min(floor((y_end - x_start) / T) - W - 1, len(wave) - W - 1)
    if F < S:
        return 0
    N = ceil((lag_hrs + target_hrs) / T)
    F_lag = F - N
    if F_lag < S:
        return 0
    return F_lag - S


@dataclass
class WindowStatArgs:
    mapping: WaveStart
    predictor_size: int
    lag_minutes: float
    target_window_minutes: float
    target_size: int
    lactate: Lactate


def get_mapping_stats(args: WindowStatArgs) -> DataFrame:
    with torch.no_grad():
        wave, x_start = args.mapping
        x_end = x_start + args.predictor_size
        x = wave.values[x_start:x_end].numpy()
        x_hours = wave.hours[x_start:x_end]
        y_start = x_hours[-1] + args.lag_minutes / 60
        y_end = y_start + args.target_window_minutes / 60
        y_prev = args.lactate.interpolator.predict(x_hours).numpy()
        y_hours = torch.linspace(y_start, y_end, args.target_size)
        y = args.lactate.interpolator.predict(y_hours)
        y_x = interpolate(y.reshape(1, 1, -1), len(x)).squeeze().numpy()
        filterwarnings("ignore", "An input array is constant")

        return DataFrame(
            {
                "corr_s(x,y_prev)": spearmanr(x, y_prev)[0],
                "corr_s(x,y)": spearmanr(x, y_x)[0],
                "corr_s(y_prev,y)": spearmanr(y_prev, y_x)[0],
            },
            index=[0],
        )


@dataclass
class ParallelArgs:
    source: Path
    sid: str
    decimation: int
    predecimated: bool
    spike_removal: Optional[SpikeRemoval]
    interp_method: Optional[InterpMethod]
    ignore_errs: bool

    def __init__(
        self,
        source: Path,
        sid: str,
        decimation: int,
        predecimated: bool,
        spike_removal: Optional[SpikeRemoval],
        interp_method: Optional[InterpMethod],
        ignore_errs: bool,
    ) -> None:
        self.sid = sid
        self.decimation = decimation
        self.predecimated = predecimated
        self.spike_removal = spike_removal
        self.interp_method = interp_method
        self.ignore_errs = ignore_errs
        self.source = decimated_folder(self.decimation) if self.predecimated else source


@dataclass
class FilterArgs:
    path: Path
    predictor_window_size: int
    lag_minutes: float
    target_window_minutes: float
    decimation: int
    predecimated: bool


class Chart:
    def __init__(self, chart_path: Path) -> None:
        self.path = chart_path
        self.data = pd.read_parquet(self.path)


class DeepSubject:
    """Class for holding all data paths WITHOUT ACTUALLY LOADING ANY LARGE FILES

    Parameters
    ----------
    subj_path: Path
        Folder containing condensed subject data. E.g. folder must have a name of the form
        `f"p{n:<06}"` where `n` is an integer, e.g. `p000033`, and the like, and where the only
        contents of `subj_path` are "<timestamp>.parquet" files (potentially zero or many),
        "chart_df.parquet" (one file at most), and "lactate_df.parquet" (one file at most).

    decimation: int
        Decimation value needed when loading and calculating indices, batch counts, periods, etc.

    """

    def __init__(
        self,
        subj_path: Path,
        decimation: int,
        predecimated: bool,
        normalize: Tuple[float, float] = (MD, IQR),
        spike_removal: Optional[SpikeRemoval] = None,
        lact_interpolation: InterpMethod = InterpMethod.previous,
    ) -> None:
        self.path: Path = subj_path
        self.decimation: int = decimation
        self.predecimated = predecimated
        self.interpolation = lact_interpolation
        if self.predecimated:
            validate_predecimation(self.decimation)
        self.normalize = normalize
        self.spike_removal = spike_removal
        self.id: str = self.path.stem
        self.sid = self.id
        self.wave_paths: List[Path] = sorted((self.path / "waves").rglob("*.parquet"))
        if len(self.wave_paths) == 0:
            raise FileNotFoundError(f"No wave data at {self.path}")
        self.first_wave: Path = self.wave_paths[0]
        self.waves: List[Wave] = [
            Wave(
                path,
                self.first_wave,
                decimation=decimation,
                predecimated=predecimated,
                normalize=normalize,
                spike_removal=spike_removal,
            )
            for path in self.wave_paths
        ]

        self.lact_path: Path = subj_path / "lactate.parquet"
        self.lactate: Lactate = Lactate(
            self.lact_path, self.first_wave, interp_method=self.interpolation
        )
        self.chart_path: Path = subj_path / "chart.parquet"
        self.chart: Optional[Chart] = Chart(self.chart_path) if self.chart_path.exists() else None
        self.demographics = Demographics(self.sid, self.first_wave)

    def clean(self, method: str = "previous") -> List[pd.DataFrame]:
        return [wave.cleaned(method=method) for wave in self.waves if not wave.is_empty()]

    def is_constant(self) -> bool:
        return self.n_lact == 0

    def get_wave_slicers(
        self,
        desired_predictor_window_minutes: float,
        lag_minutes: float = 0,
        target_window_minutes: int = 24 * 60,
    ) -> List[WaveSlicer]:
        i = 0
        predictor_size = window_dimensions(
            desired_minutes=desired_predictor_window_minutes, decimation=self.decimation
        )[0]
        slicers: List[WaveSlicer] = []
        for w, wave in enumerate(self.waves):
            wave.values
            result = get_wave_endpoints(
                wave,
                lactate=self.lactate,
                predictor_window_size=predictor_size,
                lag_minutes=lag_minutes,
                target_window_minutes=target_window_minutes,
                decimation=self.decimation,
                full_stats=False,
            )
            if result is None:
                continue
            S, F = result[:2]
            slicers.append(WaveSlicer(wave, S, F + 1))
        return slicers

    def get_window_stats(
        self,
        desired_predictor_window_minutes: float,
        lag_minutes: float = 0,
        target_window_minutes: int = 24 * 60,
        random_windows_per_subject: int = 1000,
    ) -> DataFrame:
        predictor_size = window_dimensions(
            desired_minutes=desired_predictor_window_minutes, decimation=self.decimation
        )[0]
        target_size = window_dimensions(
            desired_minutes=target_window_minutes, decimation=self.decimation
        )[0]
        slicers = self.get_wave_slicers(
            desired_predictor_window_minutes=desired_predictor_window_minutes,
            lag_minutes=lag_minutes,
            target_window_minutes=target_window_minutes,
        )
        dfs = []
        slicer: WaveSlicer
        for slicer in slicers:
            df = slicer.wave.target_correlations(
                target=self.lactate,
                start=slicer.start,
                end=slicer.end,
                predictor_size=predictor_size,
                lag_minutes=lag_minutes,
                target_window_minutes=target_window_minutes,
                target_size=target_size,
            )
            dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)

        # map_decimation = ceil(len(mappings) / random_windows_per_subject)
        # args = [
        #     WindowStatArgs(
        #         mapping=mappings[i],
        #         predictor_size=predictor_size,
        #         lag_minutes=lag_minutes,
        #         target_window_minutes=target_window_minutes,
        #         target_size=target_size,
        #         lactate=self.lactate,
        #     )
        #     for i in range(len(mappings))[::map_decimation]
        # ]
        # dfs = process_map(get_mapping_stats, args, chunksize=1, disable=True, max_workers=80)
        # df = pd.concat(dfs, axis=0, ignore_index=True)
        return df

    @cached_property
    def n_lact(self) -> int:
        h0 = self.waves[0].hours_0
        hf = self.waves[-1].hours_f
        t = self.lactate.hours
        intersecting = (h0 <= t) & (t <= hf)
        # if intersecting are all false, lactate will be constant
        return len(t[intersecting])

    @staticmethod
    def is_zombie(self: DeepSubject) -> bool:
        dod = self.demographics.dod_hrs or np.nan
        if np.isnan(dod):
            return False
        hf_waves = self.waves[-1].hours_f
        hf_lact = self.lactate.hours[-1]
        wave_zombie: bool = hf_waves > dod
        lact_zombie: bool = hf_lact > dod
        return wave_zombie or lact_zombie

    def split_at_gaps(self, max_gap_hrs: float = 24) -> List[DeepSubject]:
        """Do whenever distance between adjacent waves ends/starts is greater
        than or equal to `max_gap_hrs`, split the subject into runs"""
        if len(self.waves) == 1:
            return [self]
        gaps_ = [0]
        for i in range(len(self.waves) - 1):
            gaps_.append(self.waves[i + 1].hours_0 - self.waves[i].hours_f)
        gaps = np.array(gaps_)
        if np.all(gaps < max_gap_hrs):
            return [self]

        subjects: List[DeepSubject] = []
        idx = [0, *np.where(gaps >= max_gap_hrs)[0], len(self.waves)]
        rids = list(range(len(idx) - 1))
        for i, rid in enumerate(rids):
            start, stop = idx[i], idx[i + 1]
            subjects.append(self.to_run(rid, self.wave_paths[start:stop]))
        return subjects

    def to_run(self, run_id: int, wave_paths: List[Path]) -> DeepSubject:
        cls: Type[DeepSubject] = self.__class__
        new: DeepSubject = cls.__new__(cls)
        new.path = self.path
        new.decimation = self.decimation
        new.predecimated = self.predecimated
        new.normalize = self.normalize
        new.spike_removal = self.spike_removal
        new.id = new.path.stem
        new.sid = new.id
        setattr(new, "run_id", run_id)

        new.wave_paths = sorted(wave_paths)
        new.first_wave = wave_paths[0]
        new.waves = [
            Wave(
                path,
                new.first_wave,
                decimation=new.decimation,
                predecimated=new.predecimated,
                normalize=new.normalize,
                spike_removal=new.spike_removal,
            )
            for path in new.wave_paths
        ]

        new.lact_path = self.path / "lactate.parquet"
        new.lactate = Lactate(new.lact_path, new.first_wave, interp_method=self.interpolation)
        new.chart_path = self.path / "chart.parquet"
        new.chart = Chart(new.chart_path) if new.chart_path.exists() else None
        new.demographics = Demographics(self.sid, self.first_wave)
        return new

    @staticmethod
    def initialize_sids_with_defaults(
        sids: Optional[List[str]],
        **kwargs: Any,
    ) -> List[DeepSubject]:
        """
        return DeepSubject.initialize_from_sids(
            sids=sids,
            source=PREDECIMATED_DATA,
            predictor_window_size=window_dimensions(PRED_MIN, DEC)[0],
            lag_minutes=0,
            target_window_minutes=TARG_MIN,
            decimation=DEC,
            predecimated=True,
            spike_removal=None,
            interp_method=InterpMethod.previous,
            progress=sids is None or len(sids) > 5,
            **kwargs,
        )
        """
        return p__initialize_sids_with_defaults(sids, **kwargs)

    @staticmethod
    def find_optimal_chunksize() -> List[DeepSubject]:
        """Optimal seems to be 2"""
        PRED_MIN = 30
        DEC = 500
        TARG_MIN = 24 * 60  # 24 hours

        source = decimated_folder(DEC)
        paths = sorted(source.glob("p*"))[:100]
        dfs = []
        for chunksize in [1, 2, 4, 8, 12, 16]:
            start = time()
            paths = DeepSubject.filter_usable(
                paths,
                predictor_window_size=window_dimensions(PRED_MIN, DEC)[0],
                lag_minutes=0,
                target_window_minutes=TARG_MIN,
                decimation=DEC,
                predecimated=True,
                progress=False,
                chunksize=chunksize,
            )
            sids = [p.stem for p in paths]
            args = [
                ParallelArgs(source, sid, DEC, True, InterpMethod.previous, False) for sid in sids
            ]
            subjects: List[DeepSubject] = process_map(
                load_deepsubject_by_id,
                args,
                chunksize=chunksize,
                desc="Initializing deep learning subjects",
                disable=False,
                max_workers=MAX_WORKERS,
            )
            duration = time() - start
            dfs.append(pd.DataFrame(dict(chunksize=chunksize, duration=duration), index=[0]))
        df = pd.concat(dfs, axis=0, ignore_index=True)
        print(df.to_markdown(tablefmt="simple"))

    @staticmethod
    def initialize_from_sids(
        sids: Optional[List[str]],
        source: Path,
        predictor_window_size: int,
        lag_minutes: float,
        target_window_minutes: float,
        decimation: int,
        predecimated: bool,
        interp_method: InterpMethod,
        spike_removal: SpikeRemoval = None,
        ignore_errors: bool = False,
        progress: bool = True,
    ) -> List[DeepSubject]:
        """Loads (initializes, since very little data is actually loaded) DeepSubject
        objects from sids in parallel using tqdm `process_map`
        """
        source = decimated_folder(decimation) if predecimated else source
        if sids is None:
            paths = sorted(source.glob("p*"))
            paths = DeepSubject.filter_usable(
                paths,
                predictor_window_size,
                lag_minutes,
                target_window_minutes,
                decimation,
                predecimated,
                progress,
            )
            sids = [p.stem for p in paths]
        chunksize = 8 if ON_NIAGARA else 2
        args = [
            ParallelArgs(
                source=source,
                sid=sid,
                decimation=decimation,
                predecimated=predecimated,
                spike_removal=spike_removal,
                interp_method=interp_method,
                ignore_errs=ignore_errors,
            )
            for sid in sids
        ]
        subjects: List[DeepSubject] = process_map(
            load_deepsubject_by_id,
            args,
            chunksize=chunksize,
            desc="Initializing deep learning subjects",
            disable=not progress,
            max_workers=MAX_WORKERS,
        )
        return [s for s in subjects if s is not None]

    @staticmethod
    def initialize_from_sids_unfiltered(
        sids: Optional[List[str]],
        source: Path,
        decimation: int,
        predecimated: bool,
        spike_removal: SpikeRemoval = None,
        interp_method: InterpMethod = InterpMethod.previous,
        ignore_errors: bool = False,
        progress: bool = True,
    ) -> List[DeepSubject]:
        """Loads (initializes, since very little data is actually loaded) DeepSubject
        objects from sids in parallel using tqdm `process_map`
        """
        source = decimated_folder(decimation) if predecimated else source
        if sids is None:
            paths = sorted(source.glob("p*"))
            sids = [p.stem for p in paths]
        chunksize = 8 if ON_NIAGARA else 2
        args = [
            ParallelArgs(
                source=source,
                sid=sid,
                decimation=decimation,
                predecimated=predecimated,
                spike_removal=spike_removal,
                interp_method=interp_method,
                ignore_errs=ignore_errors,
            )
            for sid in sids
        ]
        subjects: List[DeepSubject] = process_map(
            load_deepsubject_by_id,
            args,
            chunksize=chunksize,
            desc="Initializing deep learning subjects",
            disable=not progress,
            max_workers=MAX_WORKERS,
        )
        return [s for s in subjects if s is not None]

    @staticmethod
    def initialize_all_from_directory(
        root: Path,
        predictor_window_size: int,
        lag_minutes: float,
        target_window_minutes: float,
        decimation: int,
        predecimated: bool,
        spike_removal: SpikeRemoval = None,
        interp_method: InterpMethod = InterpMethod.previous,
        ignore_errors: bool = False,
        limit: int = None,
        progress: bool = True,
    ) -> List[DeepSubject]:
        """Initialize deep subjects from `root`. Note `root` must ba a folder containing only folders
        of the form `p000033` and the like, with each such folder containing the `*.parquet` files."""
        paths = sorted(root.glob("p*"))
        paths = DeepSubject.filter_usable(
            paths,
            predictor_window_size,
            lag_minutes,
            target_window_minutes,
            decimation,
            predecimated,
            progress,
        )
        idx = slice(None) if limit is None else slice(0, limit)
        sids = [p.stem for p in paths[idx]]
        return DeepSubject.initialize_from_sids_unfiltered(
            sids=sids,
            source=root,
            decimation=decimation,
            predecimated=predecimated,
            spike_removal=spike_removal,
            interp_method=interp_method,
            ignore_errors=ignore_errors,
            progress=progress,
        )

    @staticmethod
    def initialize_all_from_window_args(
        root: Path,
        predecimated: bool,
        window_args: WindowArgs,
        spike_removal: SpikeRemoval = None,
        interp_method: InterpMethod = InterpMethod.previous,
        limit: int = None,
        progress: bool = True,
        ignore_errors: bool = False,
    ) -> List[DeepSubject]:
        decimation = window_args.decimation.value
        root = decimated_folder(decimation) if predecimated else root
        return DeepSubject.initialize_all_from_directory(
            root=root,
            predictor_window_size=window_dimensions(
                window_args.desired_predictor_window_minutes.value, decimation
            )[0],
            lag_minutes=window_args.lag_minutes.value,
            target_window_minutes=window_args.target_window_minutes.value,
            decimation=window_args.decimation.value,
            predecimated=predecimated,
            spike_removal=spike_removal,
            interp_method=interp_method,
            limit=limit,
            progress=progress,
            ignore_errors=ignore_errors,
        )

    @staticmethod
    def initialize_all_from_directory_unfiltered(
        root: Path,
        decimation: int,
        predecimated: bool,
        spike_removal: SpikeRemoval = None,
        interp_method: InterpMethod = InterpMethod.previous,
        ignore_errors: bool = False,
        limit: int = None,
        progress: bool = True,
    ) -> List[DeepSubject]:
        """Initialize deep subjects from `root`. Note `root` must ba a folder containing only folders
        of the form `p000033` and the like, with each such folder containing the `*.parquet` files."""
        root = decimated_folder(decimation) if predecimated else root
        paths = sorted(root.glob("p*"))
        idx = slice(None) if limit is None else slice(0, limit)
        sids = [p.stem for p in paths[idx]]
        return DeepSubject.initialize_from_sids_unfiltered(
            sids=sids,
            source=root,
            decimation=decimation,
            predecimated=predecimated,
            spike_removal=spike_removal,
            interp_method=interp_method,
            ignore_errors=ignore_errors,
            progress=progress,
        )

    @staticmethod
    def filter_usable(
        paths: List[Path],
        predictor_window_size: int,
        lag_minutes: float,
        target_window_minutes: float,
        decimation: int,
        predecimated: bool,
        progress: bool,
        chunksize: int = 2,
    ) -> List[Path]:
        usable = []
        filter_args = [
            FilterArgs(
                path,
                predictor_window_size,
                lag_minutes,
                target_window_minutes,
                decimation,
                predecimated,
            )
            for path in paths
        ]
        usable = process_map(
            filter_usable_at_path,
            filter_args,
            desc="Getting subjects with wave / lactate data",
            disable=not progress,
            chunksize=chunksize,
            max_workers=MAX_WORKERS,
        )
        return [p for p in usable if p is not None]

    @staticmethod
    def get_usable_sids(
        source: Path,
        desired_predictor_window_minutes: int,
        decimation: int,
        predecimated: bool,
        spike_removal: SpikeRemoval = None,
        target_window_minutes: int = 24 * 60,
        lag_minutes: float = 0,
        outfile: Path = None,
    ) -> Tuple[List[str], ndarray, int]:
        source = decimated_folder(decimation) if predecimated else source
        pred_size = window_dimensions(desired_predictor_window_minutes, decimation)[0]
        subs = DeepSubject.initialize_all_from_directory(
            root=source,
            predictor_window_size=pred_size,
            lag_minutes=lag_minutes,
            target_window_minutes=target_window_minutes,
            decimation=decimation,
            predecimated=predecimated,
            spike_removal=spike_removal,
            ignore_errors=True,
            limit=None,
            progress=True,
        )
        sids: List[str] = [s.sid for s in subs]
        counts = [s.n_lact for s in subs]
        n_zombies = int(np.sum([DeepSubject.is_zombie(s) for s in subs]))
        outfile = outfile or (ROOT / "usable_sids.txt")
        with open(outfile, "w") as handle:
            for sid in sids:
                handle.write(f"{sid}\n")
        print(f"Usable sids for given config saved to {outfile}")
        return sids, np.array(counts), n_zombies


def load_deepsubject_by_id(args: ParallelArgs) -> Optional[DeepSubject]:
    path = Path(args.source / args.sid)
    try:
        return DeepSubject(
            path,
            decimation=args.decimation,
            predecimated=args.predecimated,
            spike_removal=args.spike_removal,
            lact_interpolation=args.interp_method,
        )
    except FileNotFoundError:
        LOGGER.error(f"File missing at {path}")
        LOGGER.error(traceback.format_exc())
    except IndexError:
        LOGGER.error(traceback.format_exc())
    except Exception as e:
        LOGGER.critical(traceback.format_exc())
        raise e
        # if not args.ignore_errs:
        #     raise e
    return None


def filter_usable_at_path(args: FilterArgs) -> Optional[Path]:
    path = args.path
    wave_paths: List[Path] = sorted((path / "waves").rglob("*.parquet"))
    lact_path: Path = path / "lactate.parquet"
    if (len(wave_paths) != 0) and (lact_path.exists()):
        n = 0
        for wave_path in wave_paths:
            try:
                wave = Wave(wave_path, wave_paths[0], args.decimation, args.predecimated)
            except ValueError as e:
                if "Wave contains insufficient data" in str(e):
                    print(e)
                else:
                    raise RuntimeError("Unexpected error:") from e
            try:
                lact = Lactate(lact_path, wave_paths[0])
            except ValueError:
                print(f"Insufficient lactate data for interpolation in {lact_path}")
                continue
            n_windows = count_usable_windows(
                wave,
                lact,
                args.predictor_window_size,
                args.lag_minutes,
                args.target_window_minutes,
                args.decimation,
            )
            n += n_windows
        if n > 1:
            return path


@MEMORY.cache()
def p__initialize_sids_with_defaults(sids: Optional[List[str]], **kwargs: Any) -> List[DeepSubject]:
    PRED_MIN = 30
    DEC = 500
    TARG_MIN = 24 * 60  # 24 hours
    base_args = dict(
        sids=sids,
        source=PREDECIMATED_DATA,
        predictor_window_size=window_dimensions(PRED_MIN, DEC)[0],
        lag_minutes=0,
        target_window_minutes=TARG_MIN,
        decimation=DEC,
        predecimated=True,
        spike_removal=None,
        interp_method=InterpMethod.previous,
        progress=sids is None or len(sids) > 5,
    )
    final_args = {**base_args, **kwargs}
    return DeepSubject.initialize_from_sids(**final_args)
