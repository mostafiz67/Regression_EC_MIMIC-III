from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Memory
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from src.acquisition.headers.dat import Datfile
from src.acquisition.headers.group import HeaderGroup
from src.constants import BASE_FREQUENCY, ROOT, WAVE_VAL

MEMORY = Memory(ROOT / "__WAVEFORM_CACHE__", compress=False)


class Waveform:
    def __init__(self, group: HeaderGroup, modality: str = "ABP"):
        self.group = group
        self.modality = modality
        # self.segments = [Segment(dat, modality) for dat in group.dats]

    def as_contiguous(
        self, drop_border_nans: bool = True
    ) -> Optional[Tuple[DataFrame, pd.Timestamp]]:
        """Combines waves within the header group of this class into a single usable DataFrame.

        Parameters
        ----------
        drop_border_nans: bool = True
            If True, removes any contiguous chunks of leading and trailing NaN values on either side
            of the raw waves, and corrects the start time accordingly. Non-border NaN values are
            left in place.

        Returns
        -------
        (df, start): Optional[Tuple[DataFrame, pd.Timestamp]]
            Returns None if:

                * all segments in the header group are empty (can happen believe it or not)
                * all segments in the header group are NaN
                * if dropping border NaNs produces an empty contiguous wave

            Otherwise returns (df, start):

                df: DataFrame
                    A single-column DataFrame `df` of the wave data. Column is `modality`, which
                    should for now always be `src.constants.WAVE_VAL`, i.e. "ABP".

                start: the new (possibly corrected) start time if border NaNs were removed.
        """
        if self.modality != WAVE_VAL:
            raise NotImplementedError(
                "This code has only been implemented with `modality='ABP'` in mind."
            )

        times = None
        total_times = np.sum(self.group.segment_lengths)
        if total_times <= 0:
            return None

        result = self.join_dats()  # this can rarely raise if files are corrupt
        if result is None:
            return None

        waves, start = result
        if start is None:
            return None
        df = pd.DataFrame({self.modality: waves})

        if drop_border_nans:
            times = pd.date_range(start, freq=f"{1/BASE_FREQUENCY}S", periods=total_times)
            start_idx = df[WAVE_VAL].first_valid_index()
            end_idx = df[WAVE_VAL].last_valid_index()
            if (start_idx is None) or (end_idx is None):
                return None
            df = df.iloc[start_idx : end_idx + 1]
            if len(df) <= 1:
                return None
            start = pd.Timestamp(times[start_idx])

        return df, start

    def join_dats(self) -> Optional[Tuple[ndarray, pd.Timestamp]]:
        """
        Returns
        -------
        (wave, start): Optional[Tuple[ndarray, pd.Timestamp]]
            Returns None if all segments are empty or are all NaN. Otherwise returns
            wave values as a NumPy array and the start time as a Pandas Timestamp.
        """
        # If the first dat file (or dat files) is (or are) None we have no initial
        # time value to grab to use to generate timepoints. So we have to grab the
        # timepoint from the first actual dat file.
        modality_waves: List[ndarray] = []
        start: Optional[pd.Timestamp] = None
        dat: Datfile
        for dat, seg_length in tqdm(
            zip(self.group.dats, self.group.segment_lengths),
            desc="Reading dat files",
            total=len(self.group.segment_lengths),
            disable=True,
        ):
            if dat is None:
                # There can still be missing segements or ~ columns in the header file and
                # unfortunately time continues to increment over these gaps, so we can't just
                # skip over them. For now we just append a block of NaNs of the right size instead.
                modality_waves.append(np.full(seg_length, np.nan))
                continue
            starttime, wave = dat.read(self.modality)
            if start is None:  # first dat read
                start = starttime
            if wave is None:
                # Same as above, filler to maintain correct times. However, this `wave` can be none
                # because the selected modality is simply not present just at this time.
                modality_waves.append(np.full(seg_length, np.nan))
                continue

            if len(wave) != seg_length:
                raise ValueError(
                    f"Datfile {dat} has number of timepoints that do not match number "
                    f"indicated by master header file {seg_length}."
                )
            modality_waves.append(wave)
        if len(modality_waves) == 0:
            return None

        modality_wave: ndarray = np.concatenate(modality_waves, axis=0)
        return modality_wave, start

    @staticmethod
    def drop_border_nans(df: DataFrame) -> DataFrame:
        start = df[WAVE_VAL].first_valid_index()
        end = df[WAVE_VAL].last_valid_index()
        return df.iloc[start:end]
