from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from wfdb.io import rdsamp

from src._logging.base import LOGGER
from src.acquisition.headers.dat import Datfile
from src.acquisition.headers.layout import LayoutHeader
from src.acquisition.headers.master import MasterHeader
from src.acquisition.headers.segment import SegmentHeader

# NOTE: see https://github.com/MIT-LCP/mimic-code/issues?q=matched+waveform
# for useful hints and tips (maybe)


class HeaderGroup:
    def __init__(self, master: MasterHeader, layout: LayoutHeader) -> None:
        self.master: MasterHeader = master
        self.layout: LayoutHeader = layout
        self.segments: List[Optional[SegmentHeader]] = []
        self.dats: List[Optional[Datfile]] = []
        self.segment_lengths: List[int] = []
        for i, fname in enumerate(master.info["header"]):
            seg_length = int(master.info.iloc[i]["T"])
            self.segment_lengths.append(seg_length)
            if "~" in str(fname):
                self.segments.append(None)
                self.dats.append(None)
                continue
            datestamp = str(master.info["start"].iloc[i]).split(" ")[0]
            header_file = master.path.parent / fname
            seg_header = SegmentHeader(header_file, datestamp)
            dat_path = Path(str(header_file).replace(".hea", ".dat"))
            self.segments.append(seg_header)
            self.dats.append(Datfile(dat_path, seg_header))
        self.start = self.master.info["start"].iloc[0]
        self.end = self.master.info["end"].iloc[-1]

    def join_segments(self) -> DataFrame:
        """Reduce a HeaderGroup segments from N starts and ends to J starts and ends. E.g. 40
        header files to 3-4 waves with only gaps less than threshold specified in config."""
        raise RuntimeError("There is no need to join now since the ~ lines indicate breaks.")

    def segment_stats(self) -> DataFrame:
        """Return a DataFrame showing various useful summary stats for teh segments"""
        starts, ends = self.master.info["start"], self.master.info["end"]

    def analyze_discrepancies(self) -> DataFrame:
        """Compare master vs. segment header date starts.

        Returns
        -------
        df: DataFrame
            DataFrame with columns:
                "m_start", "seg_start", "m_T", "seg_T", "seg_f", "m_end", "seg_end", "diff_start", "diff_T"
        """
        df = pd.DataFrame()
        seg_starts, seg_fs, seg_Ts = [], [], []
        for seg in self.segments:
            if seg is None:
                continue
            seg_starts.append(seg.start)
            seg_fs.append(seg.freq)
            seg_Ts.append(seg.timepoints)
        df = pd.DataFrame(
            {
                "m_start": self.master.info.loc[:, "start"].copy(),
                "seg_start": seg_starts,
                "m_T": self.master.info.loc[:, "T"].copy(),
                "seg_T": seg_Ts,
                "seg_f": seg_fs,
                "m_end": self.master.info.loc[:, "end"].copy(),
            }
        )
        secs = df["seg_T"] * df["seg_f"] / 60
        df["seg_end"] = df["seg_start"] + secs.apply(lambda s: pd.Timedelta(seconds=s))
        df["diff_start"] = df["m_start"] - df["seg_start"]
        df["diff_T"] = df["m_T"] - df["seg_T"]
        return df

    # fix this based on dat.py!
    def read_dats(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        # rdsamp expects no file extension
        start_times = []
        end_times = []
        # waves = []
        for dat, segment in zip(self.dats, self.segments):
            try:
                waveforms, info = rdsamp(str(dat.path).replace(".dat", ""), warn_empty=True)
            except FileNotFoundError as e:
                LOGGER.error(f"Could not find dat file: {dat.path}")
                return [(None, None)]
            if "ABP" not in info["sig_name"]:
                # LOGGER.debug(f"{dat.name}:  No ABP data.")
                continue
            wave_idx = info["sig_name"].index("ABP")
            wave = waveforms[:, wave_idx]
            year, month, day = segment.start.year, segment.start.month, segment.start.day
            start = info["base_time"]
            hr, minute, sec = start.hour, start.minute, start.second
            start_time = pd.to_datetime(f"{year}-{month}-{day} {hr}:{minute}:{sec}")
            hrs = info["sig_len"] / (info["fs"] * 60 * 60)
            end_time = start_time + pd.Timedelta(hours=hrs)
            # time = pd.date_range(start=start_time, end=end, periods=info["sig_len"]).to_numpy()
            if np.all(np.isnan(wave)):
                # LOGGER.debug(f"{dat.name}:  All NaN.")
                continue
            else:
                # LOGGER.debug(
                #    f"{dat.name}:  mean={np.nanmean(wave):0.2f}, sd={np.nanstd(wave, ddof=1):0.2f} "
                #    f"Percentiles: {np.round(np.nanpercentile(wave, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 0)}"
                # )
                start_times.append(start_time)
                end_times.append(end_time)
                # waves.append(wave)
        starts, ends = np.array(start_times), np.array(end_times)
        time = list(zip(starts, ends))
        # time = np.concatenate(times, axis=0)
        # wave = np.concatenate(waves, axis=0)
        return time
