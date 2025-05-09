from pathlib import Path
from typing import List, Tuple

import pandas as pd


class SegmentHeader:
    """Class for the simplest header .hea files that correspond to a single .dat file with
    filenames like `3704341_0004.hea`"""

    def __init__(self, path: Path, datestamp: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No segement header at {path}")
        self.path = path
        self.datestamp = datestamp
        self.start, self.freq, self.timepoints, self.modalities = self.parse()
        self.end = self.start + pd.Timedelta(seconds=self.timepoints / self.freq)

    def start_time(self) -> pd.Timestamp:
        pass

    def parse(self) -> Tuple[pd.Timestamp, float, int, List[str]]:
        with open(self.path) as header:
            lines = [line.replace("\n", "") for line in header.readlines()]
        firstline = lines[0]
        firstsplits = firstline.split(" ")
        time = self.parse_time(firstsplits[-1])
        freq = float(firstsplits[2])
        timepoints = int(firstsplits[3])
        if freq <= 0:
            raise RuntimeError(f"Invalid frequency in {self.path}. \n{firstline}\n{lines}")
        if timepoints <= 0:
            raise RuntimeError(f"Invalid timepoints in file {self.path}. \n{firstline}\n{lines}")
        # process the rest
        lines = [line for line in lines[1:] if not line.startswith("#")]  # remove comments
        splits = [line.split(" ") for line in lines]
        modalities = [split[-1] for split in splits]
        return time, freq, timepoints, modalities

    def parse_time(self, line: str) -> pd.Timestamp:
        """Parses time, splits it and returns it in hours, minutes and seconds

        Notes
        -----
        Example segment (.dat) header files:

            3713820_0001 2 125 15000 12:35:06.147
            3713820_0001.dat 80 24/mV 8 0 0 9875 0 II
            3713820_0001.dat 80 24/mV 8 0 0 -1750 0 MCL1

            3700696_0060 3 125 812747 22:22:15.550
            3700696_0060.dat 80 21/mV 8 0 0 -3433 0 II
            3700696_0060.dat 80 75/mV 8 0 1 -20704 0 MCL1
            3700696_0060.dat 80 0.833333(-100)/mmHg 8 0 -40 15549 0 ABP

            3700696_0061 3 125 5733    11:37.525
            3700696_0061.dat 80 21/mV 8 0 -72 5338 0 II
            3700696_0061.dat 80 75/mV 8 0 -72 932 0 MCL1
            3700696_0061.dat 80 0.833333(-100)/mmHg 8 0 -72 -8109 0 ABP

        Possible formats for the timestamp (lol):

            Case 1: 12:35:06.147  (HH:MM:SS.xxx)
            Case 2: 4:47:26.263   (H:MM:SS.xxx)
            Case 3: 49:05.758     (MM:SS.xxx)
            Case 4: 9:40.209      (M:SS.xxx)
            ----------------------------------  <-- no microseconds after
            Case 5: 01:11:31      (HH.MM.SS)
            Case 6: 8:11:02       (H:MM.SS)
            Case 7: 52:46         (MM:SS)
            Case 8: 9:40          (M:SS)

        Easiest way to handle is to add microseconds if not present, since this
        makes Case 5 -> 1, 6 -> 2, 7 -> 3, and 8 -> 2. Then the number of ":"
        differentiates betwen 1/2 or 3/4, and 3/4 can be front zero padded to make
        case 1/2. Then case 2 is padded to case 1.
        """
        if line.find(".") < 0:
            line = f"{line}.000"  # add missing microseconds
        if line[1] == ":":  # zero pad short, now only in case 1 or 3
            line = f"0{line}"
        if len(line) < 12:  # change Case 3 to 1
            line = f"00:{line}"
        return pd.Timestamp(f"{self.datestamp} {line}")

    def __str__(self) -> str:
        return (
            f"{self.path.name} {self.modalities}: {self.start} +{self.timepoints} @ {self.freq}Hz"
        )

    __repr__ = __str__
