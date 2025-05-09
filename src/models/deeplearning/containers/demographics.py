# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()  # isort: skip
# fmt: on

import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src._logging.base import LOGGER
from src.constants import PATIENTS_INFO


class Demographics:
    def __init__(self, sid: str, first_wave: Path) -> None:
        self.age: Optional[float] = None
        self.dod: Optional[pd.Timestamp] = None
        self.dob: Optional[pd.Timestamp] = None
        self.dod_hrs: Optional[float] = None
        self.sid = sid
        self.sex: Literal["M", "F"]  # always defined in table

        df = pd.read_csv(PATIENTS_INFO, compression="gzip")
        df.index = df["SUBJECT_ID"].apply(lambda id: f"p{id:06}")
        self.sex = str(df.loc[sid, "GENDER"]).upper()
        try:
            start = pd.Timestamp.to_pydatetime(pd.to_datetime(first_wave.stem))
            self.dod = pd.to_datetime(df.loc[sid, "DOD"])
            self.dob = pd.to_datetime(df.loc[sid, "DOB"])
        except:
            LOGGER.error(traceback.format_exc())
            LOGGER.error(
                "Pandas is probably throwing an error it has chosen to disguise as an "
                "'OverflowError', but which cannot be caught as such. Details above. "
            )
            self.dob = None
            self.dod = None
            self.dod_hrs = None
            self.age = None
            return

        if pd.isna(self.dod):
            self.dod = None
            self.dod_hrs = None
        else:
            try:
                dod = pd.Timestamp.to_pydatetime(self.dod)
                diff = dod - start
                self.dod_hrs = diff / pd.Timedelta(hours=1)
            except:
                LOGGER.error(traceback.format_exc())
                LOGGER.error(
                    "Pandas is probably throwing an error it has chosen to disguise as an "
                    "'OverflowError', but which cannot be caught as such. Details above. "
                )
                self.dod_hrs = None
        if pd.isna(self.dob):
            self.dob = None
            self.age = None
        else:
            try:
                dob = pd.Timestamp.to_pydatetime(self.dob)
                diff = start - dob
                self.age = diff / pd.Timedelta.to_pytimedelta(pd.Timedelta(days=365.2425))
            except:
                LOGGER.error(traceback.format_exc())
                LOGGER.error(
                    "Pandas is probably throwing an error it has chosen to disguise as an "
                    "'OverflowError', but which cannot be caught as such. Details above. "
                )
                self.age = None

    def __str__(self) -> str:
        dob = "NA" if self.dob is None else self.dob.strftime("%Y-%m-%d")
        dod = "NA" if self.dod is None else self.dod.strftime("%Y-%m-%d")
        age = f"{self.age:0.0f}" if self.age is not None else "??"
        dod_hrs = "" if self.dod_hrs is None else f" (Deceased at {self.dod_hrs:0.1f} hrs)"
        return f"{self.sid}: {age}-{self.sex} DOB: {dob}, DOD: {dod}{dod_hrs}"

    def info(self) -> str:
        age = f"{self.age:0.0f}" if self.age is not None else "??"
        rid = f"_r{str(self.run_id)}" if hasattr(self, "run_id") else ""
        return f"{self.sid}{rid}: {age}-{self.sex}"

    __repr__ = __str__


if __name__ == "__main__":
    first = Path(
        "/home/derek/Desktop/MIMIC-III_Clinical_Database/data/predecimated/decimation_500/p000188/waves/21611209-175251.parquet"
    )
    sid = "p000188"
    first = Path(
        "/home/derek/Desktop/MIMIC-III_Clinical_Database/data/predecimated/decimation_500/p000377/waves/21680321-154921.parquet"
    )
    sid = "p000377"
    demo = Demographics(sid, first)
    print(demo)
