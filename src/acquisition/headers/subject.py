from __future__ import annotations

import pickle
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from matplotlib.pyplot import Axes
from pandas import DataFrame, Timedelta
from tqdm.contrib.concurrent import process_map

from src._logging.base import LOGGER
from src.acquisition.headers.group import HeaderGroup
from src.acquisition.headers.layout import LayoutHeader
from src.acquisition.headers.master import MasterHeader
from src.acquisition.headers.segment import SegmentHeader
from src.constants import CONDENSE_ROOT, DATA, MEMORY, ON_NIAGARA, RESULTS

# NOTE: see https://github.com/MIT-LCP/mimic-code/issues?q=matched+waveform
# for useful hints and tips (maybe)

Overlap = Tuple[pd.Timedelta, pd.Timedelta]


class RawSubject:
    def __init__(self, subj_path: Path, check_dats: bool = True) -> None:
        self.path: Path
        self.check_dats: bool
        self.sid: str
        self.id: str
        self.all_headers: List[Path]
        self.master_headers: List[MasterHeader]
        self.layouts: List[LayoutHeader]
        self.header_groups: List[HeaderGroup] = []
        self.n_headers: int
        self.start: Timedelta
        self.end: Timedelta

        self.path = subj_path
        self.id = self.sid = self.path.stem

        self.all_headers = sorted(subj_path.rglob("*.hea"))
        master_paths = sorted(filter(self.is_master, self.all_headers))
        self.master_headers = [MasterHeader(path) for path in master_paths]
        self.layouts = [LayoutHeader(master.layout) for master in self.master_headers]
        self.check_dats = check_dats
        if self.check_dats:
            for master, layout in zip(self.master_headers, self.layouts):
                self.header_groups.append(HeaderGroup(master, layout))
        self.n_headers = int(sum(map(lambda m: len(m.info), self.master_headers)))

        self.start = self.master_headers[0].start
        self.end = self.master_headers[-1].end

    def plot_segments(
        self,
        chart_targets: List[str] = [],
        targets: List[pd.Timestamp] = [],
        labels: List[str] = [],
        modality: str = None,
        show_overlaps: bool = True,
        plot_dats: bool = True,
    ) -> None:
        def to_hours(t: pd.Timedelta) -> float:
            return float(t.total_seconds() / 3600)

        ax: Axes
        groups: List[HeaderGroup]
        if modality is not None:
            groups = [group for group in self.header_groups if modality in group.layout.modalities]
        else:
            groups = self.header_groups
        if len(groups) == 0:
            return

        # sbn.set_style("darkgrid")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_ylim(bottom=0, top=8)
        palette = sbn.color_palette("deep")
        t0 = min([g.master.start for g in groups])
        t0_master = t0
        target_times = [to_hours(target - t0) for target in targets]
        chart_times = []
        for chart_target in chart_targets:
            chart_target_times = []
            for chart_time in chart_target:
                time = to_hours(chart_time - t0)
                chart_target_times.append(time)
            chart_times.append(chart_target_times)
        # t0_dat = np.min(dat_times)
        # dat_times = (dat_times - t0_dat) / 3.6e12
        gap_seen = False
        for g, group in enumerate(groups):
            # plot from master header file
            df = group.master.info
            start = (df.start - t0).apply(to_hours)
            end = (df.end - t0).apply(to_hours)
            label = group.layout.path.name
            color = g
            for i, (hname, x1, x2, segment) in enumerate(
                zip(df["header"], start, end, group.segments)
            ):
                if modality not in segment.modalities:
                    continue
                if i == 0:
                    ax.scatter(
                        x1,
                        1,
                        color=[palette[g % len(palette)]],
                        label=f"{label} (master)",
                        marker=">",
                        s=20.0,
                    )
                is_gap = str(hname)[0] == "~"
                if not is_gap:
                    ax.plot(
                        (x1, x2),
                        (1, 1),
                        color=palette[color],
                    )
                    color = (i + 1) % len(palette)
                else:
                    ax.axvspan(
                        x1,
                        x2,
                        0,
                        2,
                        color="red",
                        alpha=0.3,
                        label="missing" if not gap_seen else None,
                        lw=0.01,
                    )
                    gap_seen = True
        # plot from header files
        color = 0
        for group in groups:
            for seg in group.segments:
                t0 = min([seg.start, t0])
        scatter_labeled = False
        for g, group in enumerate(groups):
            label = group.layout.path.name
            for i, seg in enumerate(group.segments):
                if modality not in seg.modalities:
                    continue
                x = (to_hours(seg.start - t0), to_hours(seg.end - t0))
                ax.scatter(
                    x,
                    (2, 2),
                    color="black",
                    label=None if scatter_labeled else "segment start/end",
                    s=0.25,
                )
                scatter_labeled = True
                ax.axvspan(
                    x[0], x[1], 0, 2, color="black", alpha=0.05, lw=0.01, label="segment region"
                )
                color = (i + 1) % len(palette)

        for g, group in enumerate(groups):
            for i, (start, end) in enumerate(group.read_dats()):
                if start is None:
                    continue
                start = to_hours(start - t0_master)
                end = to_hours(end - t0_master)
                plt.scatter(
                    (start, end),
                    (3 + i * 0.02, 3 + i * 0.02),
                    color="blue",
                    s=0.5,
                    label=None if g != 0 or i != 0 else "dat start/end",
                )

        for i, target_time in enumerate(target_times):
            ax.scatter(
                target_time,
                4,
                color="orange",
                marker="*",
                label=None if i != 0 else "lact_time",
            )

        colors = [
            "red",
            "purple",
            "green",
            "yellow",
            "orange",
            "brown",
            "pink",
            "maroon",
            "blue",
            "black",
        ]
        chart_markers = ["*", "X", "o", "v", "x", ",", "1", "2", "3", "4"]
        chart_height = 4
        for i, (chart_time, c, cm, label_) in enumerate(
            zip(chart_times, colors, chart_markers, labels), start=1
        ):
            for j, time in enumerate(chart_time):
                ax.scatter(
                    time,
                    chart_height + (i / 5),
                    color=c,
                    marker=cm,
                    s=0.3,
                    label=None if j != 0 else label_,
                )

        if show_overlaps:
            overlaps = self.get_overlaps()
            for i, (o_lo, o_hi) in enumerate(overlaps):
                start, end = to_hours(o_lo - t0), to_hours(o_hi - t0)
                ax.axvspan(
                    start,
                    end,
                    0,
                    2,
                    color="blue",
                    alpha=0.10,
                    lw=0.01,
                    label="overlap" if i == 0 else None,
                )

        ax.set_title("All modalities" if modality is None else f"Modality = {modality}")
        ax.set_xlabel("Hours since first measurement")
        ax.set_yticks([])
        ax.legend().set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()

        sid = str(self.path.stem)
        n_headers = int(sum(map(lambda g: len(g.master.info), groups)))
        fig.legend(handles, labels, loc="lower right")
        fig.suptitle(f"{sid} ({n_headers} headers, {len(groups)} joined)")
        fig.set_size_inches(w=14, h=6)
        plt.savefig(RESULTS / f"chart+lact/unique_events/{sid}")
        LOGGER.info(f"Plot {self.path.name} saved successfully.")
        # plt.show()
        plt.close()

    def get_date(self, file: Path) -> pd.DatetimeIndex:
        header_files = [f for f in file.parent.rglob("p*.hea") if "n.hea" not in f.name]
        for header in header_files:
            with open(header) as h:
                lines = h.readlines()
            for line in lines:
                if file.stem in line:
                    header_date_splits = header.stem.split("-")[1:]
                    h_year, h_month, h_day, h_hour, h_min = header_date_splits
                    return h_year, h_month, h_day

    def plot_dat_files(self, times: List[Tuple[Any, Any]]) -> None:
        times = times[::50]
        # wave = wave[::50]
        # time /= 3.6e12  # time is in ns
        for time in times:
            plt.scatter(time, (1, 1), color="blue")
        # plt.savefig(RESULTS / f"lactate/dat/{self.path.name}")
        # LOGGER.info(f"Plot {self.path.name} saved successfully.")
        plt.show()
        # plt.close()

    def summary_df(self) -> DataFrame:
        dfs = []
        for master, layout in zip(self.master_headers, self.layouts):
            df = master.info.copy().loc[:, ["start", "end"]]
            df["label"] = layout.path.name
            dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    def get_overlaps(self) -> List[Overlap]:
        """Returns list of regions (start, end) where HeaderGroups overlap"""
        bounds = []
        for group in self.header_groups:
            bounds.append((group.start, group.end))
        bounds = sorted(bounds, key=lambda b: b[0])
        overlaps: List[Overlap] = []
        for i in range(len(bounds) - 1):
            lo1, hi1 = bounds[i]
            lo2, hi2 = bounds[i + 1]
            if lo2 < hi1:
                overlaps.append((hi1, lo2))
        return overlaps

    def overlapping_groups(
        self, time: pd.Timestamp, allow_gap: bool = True, return_segments: bool = False
    ) -> Union[List[HeaderGroup], Tuple[List[HeaderGroup], List[SegmentHeader]]]:
        """Given `time`, returns the HeaderGroup(s) that intersects with that time.

        Parameters
        ----------
        time: pd.Timestamp
            The time to compare HeaderGroup.start and HeaderGroup.end with

        allow_gap: bool = True
            If True, check MasterHeader and ensure that `time` does not lie on
            a gap / disconnect (e.g. a "~" row in the master header).
            If False, actually check each segment in the HeaderGroup and ensure
            `time` falls within this segment's start and end.

        return_segments: bool = False
            Ignored unless `allow_gap` is False.

        Returns
        -------
        groups: List[HeaderGroup]
            The intersecting groups. Is an empty list if no intersections with `time`.
        """
        intersects = []
        if allow_gap:
            for i, master in enumerate(self.master_headers):
                if master.start < time and master.end >= time:
                    intersects.append(self.header_groups[i])
            return intersects
        segments = []
        for group in self.header_groups:
            for segment in group.segments:
                if segment.start < time and segment.end >= time:
                    segments.append(segment)
                    intersects.append(group)
        if return_segments:
            return intersects, segments
        return intersects

    @staticmethod
    def load_by_ids(ids: List[str], check_dat: bool = False) -> List[RawSubject]:
        chunksize = 8 if ON_NIAGARA else 2
        subjects = process_map(
            process_subject_by_ids, ids, chunksize=chunksize, desc="Loading subjects"
        )
        subjects = [s for s in subjects if s is not None]
        return subjects

    @staticmethod
    def is_master(path: Path) -> bool:
        stem = path.stem
        return stem.startswith("p") and stem[-1] != "n"

    @staticmethod
    def load_all_from_directory(physionet_root: Path, limit: int = None) -> List[RawSubject]:
        """Constructs a list of RawSubject instances by recursively searching starting from
        physionet_root` and reading and parsing all `.hea` files.

        Parameters
        ----------
        physionet_root: Path
            Should be the `physionet.org` folder produced via `wget` download of the Matched Waveform Data,
            or a directory that contains that folder.

        limit: int = None
            If not None, load only up to a maximum of `limit` subjects.
        """
        return _load_all_from_directory_cached(physionet_root, limit)

    def __str__(self) -> str:
        lines = [f"Subject {self.id}", f"{len(self.master_headers)} master header files: "]
        for master in self.master_headers:
            lines.append(str(master))
        return "\n".join(lines)

    __repr__ = __str__


def _load_all_from_directory_cached(physionet_root: Path, limit: int = None) -> List[RawSubject]:
    PICKLE = (
        Path(MEMORY.cachedir) / f"{physionet_root.parent.name}_{physionet_root.name}_{limit}.pickle"
    )
    if PICKLE.exists():
        with open(PICKLE, "rb") as file:
            return pickle.load(file)

    pd.options.mode.chained_assignment = None  # silence phony copy warnings

    LOGGER.info("Getting unique subject folders... ")
    subject_dirs = sorted(set(map(lambda p: p.parent, physionet_root.rglob("*layout.hea"))))
    LOGGER.info("Done getting subject folders")

    idx = slice(None) if limit is None else slice(0, limit)
    paths = [s for s in subject_dirs[idx]]
    chunksize = 8 if ON_NIAGARA else 2
    workers = 80 if ON_NIAGARA else 2
    subjects: List[RawSubject] = process_map(
        load_raw_subject,
        paths,
        chunksize=chunksize,
        desc="Loading and parsing subject data/metadata",
        max_workers=workers,
    )
    subjects = [s for s in subjects if s is not None]

    PICKLE.parent.mkdir(exist_ok=True, parents=True)
    with open(PICKLE, "wb") as handle:
        LOGGER.info(f"Caching results of `load_all_from_directory` in {PICKLE}")
        pickle.dump(subjects, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return subjects


def load_raw_subject(path: Path) -> Optional[RawSubject]:
    try:
        return RawSubject(path, check_dats=True)
    except FileNotFoundError:
        LOGGER.error(f"Missing data for subject at {path}")
    except IndexError:
        LOGGER.error(traceback.format_exc())
    return None


def process_subject_by_ids(id: str) -> Optional[RawSubject]:
    root = (
        CONDENSE_ROOT / "mimic-iii-matched-waveforms/physionet.org/files/mimic3wdb/1.0/matched"
        if ON_NIAGARA
        else DATA / "mimic-iii-matched-waveforms/physionet.org/files/mimic3wdb/1.0/matched"
    )
    template = f"{root}/p0{{}}/{{}}"
    try:
        return RawSubject(Path(template.format(id[2], id)))
    except IndexError:
        LOGGER.error(traceback.format_exc())
        LOGGER.error(f"Corrupt id: {id}")
    return None
