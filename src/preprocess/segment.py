from math import ceil
from typing import Optional, Tuple

import pandas as pd
from numpy import ndarray
from wfdb import rdsamp

from src.acquisition.headers.dat import Datfile


class Segment:
    def __init__(self, datfile: Datfile, modality: str = "ABP"):
        self.datfile = datfile
        self.modality = modality
        self.start = self.datfile.start
        self.end = self.datfile.end
        self.freq = self.datfile.freq
        self.timepoints = self.datfile.timepoints

    def time(self, decimation: int = 1) -> Optional[pd.DatetimeIndex]:
        if self.modality not in self.datfile.modalities:
            return None
        timepoints = ceil(self.timepoints / decimation)
        return pd.date_range(start=self.start, end=self.end, periods=timepoints)

    def read(self, decimation: int = 1) -> Optional[ndarray]:
        waveforms, info = rdsamp(str(self.datfile.path).replace(".dat", ""), warn_empty=True)
        if self.modality not in info["sig_name"]:
            return None
        wave_idx = info["sig_name"].index(self.modality)
        wave = waveforms[:, wave_idx][::decimation]
        return wave

    def data(self, decimation: int = 1) -> Optional[Tuple[pd.DatetimeIndex, ndarray]]:
        time = self.time(decimation)
        wave = self.read(decimation)

        return time, wave
