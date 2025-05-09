import traceback
from io import StringIO
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from pandas import DataFrame

from src._logging.base import LOGGER
from src.constants import BASE_FREQUENCY

"""
NOTES
-----
# Basic Assumption

Segment headers do not list start dates (e.g. the 2150-01-01 portion of the timestamp). Thus
they must be computed by summing durations and/or timepoints from the first date indicated
by either the layout or master header files.


If you read the MIMIC database description (https://physionet.org/content/mimic3wdb/1.0/):

> Each recording comprises two records (a waveform record and a matching numerics record) in
> a single record directory (“folder”) with the name of the record. [...] In almost all
> cases, **the waveform records comprise multiple segments**, each of which can be read as a
> separate record. EACH SEGMENT CONTAINS AN UNINTERRUPTED RECORDING OF A SET OF
> SIMULTANEOUSLY OBSERVED SIGNALS, and the signal gains do not change at any time during the
> segment. Whenever the ICU staff changed the signals being monitored or adjusted the
> amplitude of a signal being monitored, this event was recorded in the raw data dump, and a
> new segment begins at that time.

> Each COMPOSITE WAVEFORM RECORD [???] includes a list of the segments that comprise it in
> its master header file. The list begins on the second line of the master header with a
> LAYOUT HEADER FILE that specifies all of the signals that are observed in any segment
> belonging to the record. Each segment has its own header file and (except for the layout
> header) a matching (binary) signal (.dat) file. oCCASIONALLY, THE MONITOR MAY BE
> DISCONNECTED ENTIRELY FOR A SHORT TIME; THESE INTERVALS ARE RECORDED AS GAPS IN THE MASTER
> HEADER FILE, BUT THERE ARE NO HEADER OR SIGNAL FILES CORRESPONDING TO GAPS.

So a master header files defines a "composite waveform record".But this contradicts the first
paragraph which states that a single record directory is a single record. It is not that clear what
they mean by "a record" or "a segment" in any of the above, but probably we *assume* segment = .dat
file., a segment is a set of (simultaneous) signals. They do *not* clearly state that *within* a
group of segments associated with a layout file ("header group") that signals happen one after
another, but we are ASSUMING that is how they happen.

However the matched db (https://physionet.org/content/mimic3wdb-matched/1.0/) has an extra
complication.

> Frequently there are multiple waveform and numerics record pairs associated with a given
> clinical record; all of them will appear in the same subdirectory in such a case, and
> their names will indicate their chronologic sequence. For example, MIMIC-III Clinical
> Database record p000079 has been matched with two waveform and numerics record pairs,
> named:
>
>   p000079-2175-09-26-01-25 and p000079-2175-09-26-01-25n
>   p000079-2175-09-26-12-28 and p000079-2175-09-26-12-28n

So if two HeaderGroups overlap, we can't really use that subject.
"""


class MasterHeader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.info, self.layout = self.parse()
        try:
            self.start = self.info["start"].iloc[0]
            self.end = self.info["end"].iloc[-1]
        except IndexError as e:
            LOGGER.critical(traceback.format_exc())
            raise RuntimeError(f"MasterHeader at {self.path} could not be parsed.") from e

    def parse(self) -> Tuple[DataFrame, Path]:
        """Convert a master header file to a dataframe of header-files and their correct times

        Notes
        -----
        A master header file generally looks like this:

            p000033-2116-12-24-12-35/6 2 125 10102500 12:35:06.147 24/12/2116
            3713820_layout 0
            3713820_0001 15000
            ~ 219
            3713820_0002 3379563
            3713820_0003 6700437
            ~ 7281
            # Location: micu

        The tilde lines (e.g. the `~ 219` line above) are important! They indicate a pause in
        recording, with the numerical value being the number of timepoints of the pause, where
        the frequency is the 125 indicated in the first line (all files have frequency of 125Hz).
        Even though there is no actual header file for a ~ line, time is still passing, and so we
        need to factor these in when computing start and end times and/or when considering if we
        should consider our waves to be contiguous or not.

        The second line should always be a layout file (0 timepoints), and also there are occasional
        useless comment lines starting with "#" to be stripped out.
        """
        line: str
        lines: Any

        with open(self.path) as master:
            lines = master.readlines()

        for i, line in enumerate(lines):  # strip comments
            if line.startswith("#"):
                lines[i] = None

        # extract starttime
        firstline, secondline = lines[0], lines[1]
        lines = lines[2:]
        if self.path.stem not in firstline:
            raise ValueError(
                f"Parse error for master file {master}. First line ({firstline}) is malformed"
            )
        time, date = firstline.split(" ")[-2:]
        day, month, year = date.split("/")
        t0 = pd.Timestamp(f"{year}-{month}-{day} {time}")

        # extract layout_file
        layout = Path(self.path.parent / f"{secondline.split(' ')[0]}.hea")
        if not layout.exists():
            raise FileNotFoundError(f"Cannot find layout file {layout}.")

        # convert to DataFrame
        text = "\n".join([line for line in lines if line is not None])
        buffer = StringIO(text)  # better than reading the file twice!
        df = pd.read_table(buffer, sep=" ", names=["header", "T"])
        df.header = df.header.apply(lambda s: Path(s + ".hea"))
        # duration of each header file in seconds is number of timepoints (T column) / FREQ
        dt = (df["T"].astype(int) / BASE_FREQUENCY).apply(lambda t: pd.Timedelta(seconds=t))
        # We trust the master header starttime `t0` and number of timepoints so can calculate header
        # start/end times. There are tiny differences between the start times computed this way and
        # the values stored in the actual header files, but these are minute and not a concern to
        # us as we are just trying to preview subject waveform overlaps and durations.
        ends = t0 + dt.cumsum()  # dt.cumsum() is total time elapsed after each segment
        starts = ends - dt
        df["start"] = starts
        df["end"] = ends
        return df, layout

    def __str__(self) -> str:
        lines = []
        lines.append(
            f"    {self.path.name} / {self.layout.name}: {len(self.info)} "
            f"header files @ {self.info['start'][0]}"
        )

        rows = [f"{8*' '}{row}" for row in str(self.info.iloc[:3]).split("\n")]
        if len(self.info) > 3:
            rows.append(f"{8*' '}...")
        lines.extend(rows)
        return "\n".join(lines)

    __repr__ = __str__
