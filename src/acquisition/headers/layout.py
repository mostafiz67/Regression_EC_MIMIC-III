from pathlib import Path
from typing import List, Tuple

import pandas as pd


class LayoutHeader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.start, self.modalities = self.parse()

    def parse(self) -> Tuple[pd.Timestamp, List[str]]:
        """Get time info from header file

        Notes
        -----
        Example layout file

            3165928_layout 7 125 0 16:26:13.320
            ~ 0 512/mV 11 0 -512 0 0 I
            ~ 0 512/mV 11 0 -2 0 0 II
            ~ 0 512/mV 11 0 -512 0 0 AVR
            ~ 0 512(-257)/mV 10 0 0 0 0 V
            ~ 0 1023(-512)/pm 10 0 0 0 0 RESP
            ~ 0 1023(-512)/NU 10 0 0 0 0 PLETH
            ~ 0 5.11333(-523)/mmHg 11 0 -512 0 0 ABP

        Layout files NEVER contain a comment. Here are also some sample first lines:

        3737936_layout 2 125 0 13:11:06.147
        3647298_layout 4 125 0 21:11:31
        3168852_layout 4 125 0  5:53:03
        3746356_layout 4 125 0  3:10:17.024
        3805787_layout 5 125 0 20:03:22.826
        3860035_layout 5 125 0 21:01:58
        3002540_layout 5 125 0 10:43:11.672
        3034224_layout 8 125 0  2:00:53.352
        3290804_layout 5 125 0  8:01:17.416
        3346682_layout 7 125 0  5:59:41.912
        3407610_layout 7 125 0 18:53:39.200
        3924895_layout 6 125 0 16:05:13.264
        3943625_layout 7 125 0 18:57:26
        3255538_layout 5 125 0 21:51:12.668
        3807277_layout 5 125 0  1:20:18.014
        3931528_layout 10 125 0 13:03:28.592
        3972293_layout 5 125 0 18:27:12.668
        3042872_layout 2 125 0  4:22:30.181
        3245316_layout 3 125 0 15:14:06.769
        3341251_layout 3 125 0    20:06.769
        3317579_layout 3 125 0  3:45:29.450
        3392522_layout 3 125 0 22:35:29.450
        3223561_layout 3 125 0 13:51:45.218
        3606357_layout 5 125 0  3:07:01.105
        3685211_layout 5 125 0 20:29:01.105
        3921064_layout 5 125 0 13:54:01.105
        3775142_layout 4 125 0 13:09:18.107

        So we can see exploit this to make getting the time very easy by
        splitting on "125 0 "
        """
        with open(self.path) as file:
            lines = file.readlines()
        # parse time
        firstline = lines[0].replace("\n", "")
        # need bytes to make mutable
        timestr = bytearray(f"{firstline.split('125 0 ')[1]:<12}".encode())
        timestr[2:3] = b":"  # for byte arrays s[1:2] = b"c" assigns char "c" to s[2]...
        timestr[5:6] = b":"  # for byte arrays s[1:2] = b"c" assigns char "c" to s[2]...
        timestr[8:9] = b"."
        stamp = timestr.decode().replace(" ", "0")
        start = pd.Timestamp(f"1970-01-01 {stamp}")  # obvious placeholder year
        # parse modalities
        lines = lines[1:]
        modalities = [line.split(" ")[-1].strip("\n") for line in lines]
        return start, modalities

    def __str__(self) -> str:
        fpath = self.path.relative_to(self.path.parent.parent)
        return f"{fpath} @ {self.start} {self.modalities}"

    __repr__ = __str__
