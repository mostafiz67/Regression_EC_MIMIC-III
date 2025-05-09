import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from numpy import ndarray
from wfdb import rdsamp

from src._logging.base import LOGGER
from src.acquisition.headers.segment import SegmentHeader


class Datfile:
    """Class for the simplest header .hea files that correspond to a single .dat file with
    filenames like `3704341_0004.hea`"""

    def __init__(self, path: Path, segment: SegmentHeader) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No dat file at {path}")
        self.path = path
        self.segment = segment
        self.start, self.end, self.freq, self.timepoints, self.modalities = self.parse()

    def parse(self) -> Tuple[pd.Timestamp, pd.Timestamp, int, int, List[str]]:
        waveforms, info = rdsamp(str(self.path).replace(".dat", ""), warn_empty=True)
        segment = self.segment
        year, month, day = segment.start.year, segment.start.month, segment.start.day
        start = info["base_time"]
        hr, minute, sec = start.hour, start.minute, start.second
        start_time = pd.to_datetime(f"{year}-{month}-{day} {hr}:{minute}:{sec}")
        hrs = info["sig_len"] / (info["fs"] * 60 * 60)
        end_time = start_time + pd.Timedelta(hours=hrs)
        # time = pd.date_range(start=start_time, end=end, periods=info["sig_len"]).to_numpy()
        return start_time, end_time, info["fs"], info["sig_len"], info["sig_name"]

    def read(self, modality: str = "ABP") -> Tuple[pd.Timestamp, Optional[ndarray]]:
        """Actually read the .dat file and load the values in `modality` into a NumPy array.

        Returns
        -------
        start: pd.Timestamp
            Start time of the wave segment. Combines information from wfdb `rdsamp` (hour, minute, second)
            and from the filename (year, month, day) of the segment header for this .dat file to ensure the
            start time is actually correct.

        Raises
        ------
        RuntimeError:
            If a valid start time cannot be constructed and returned.
        """
        waveforms, info = rdsamp(str(self.path).replace(".dat", ""), warn_empty=True)
        segment = self.segment
        year, month, day = segment.start.year, segment.start.month, segment.start.day
        try:
            start = info["base_time"]
            hr, minute, sec = start.hour, start.minute, start.second
            start_time = pd.to_datetime(f"{year}-{month}-{day} {hr}:{minute}:{sec}")
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            raise RuntimeError(
                f"Could not construct valid pd.Timestamp for .dat file at {self.path}. "
                "This likely means corrupt data in the .dat file or the associated .hea file"
            ) from e

        if modality not in info["sig_name"]:
            return start_time, None
        idx = info["sig_name"].index(modality)
        return start_time, waveforms[:, idx]
