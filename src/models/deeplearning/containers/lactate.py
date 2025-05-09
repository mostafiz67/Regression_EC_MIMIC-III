from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.interpolate import PchipInterpolator, interp1d
from torch import Tensor

from src.constants import LACT_MAX_MMOL, LACT_T, LACT_VAL


class InterpMethod(Enum):
    previous = "previous"
    linear = "linear"
    pchip = "pchip"
    savgol = "savgol"


class LactateInterpolator:
    def __init__(
        self, shifted_hours: ndarray, target: ndarray, method: InterpMethod = InterpMethod.previous
    ):
        self.interpolator: Callable[[ndarray], ndarray]
        self.method: InterpMethod

        self.method = InterpMethod(method)

        if self.method in [InterpMethod.previous, InterpMethod.linear]:
            kind = self.method.value
            self.interpolator = interp1d(
                shifted_hours,
                target,
                kind=kind,
                copy=False,
                bounds_error=False,
                fill_value="extrapolate",
            )
        elif self.method is InterpMethod.pchip:
            self.interpolator = PchipInterpolator(shifted_hours, target, extrapolate=True)
        else:
            raise NotImplementedError()

    def predict(self, hours: Union[Tensor, ndarray]) -> Tensor:
        if isinstance(hours, ndarray):
            return self.interpolator(hours)
        return Tensor(self.interpolator(hours.numpy()))


class Lactate:
    def __init__(
        self, lact_path: Path, first_wave: Path, interp_method: InterpMethod = InterpMethod.previous
    ) -> None:
        self.path = lact_path
        self.raw = pd.read_parquet(self.path)

        if len(self.raw) < 2:
            raise ValueError(
                f"Lactate data at {self.path} has length {len(self.raw)} and cannot be interpolated."
            )

        t_start = pd.to_datetime(first_wave.stem)
        shifted_hours = ((self.raw[LACT_T] - t_start) / pd.Timedelta(hours=1)).to_numpy()
        self.values: Tensor = Tensor(self.raw[LACT_VAL].to_numpy() / LACT_MAX_MMOL)
        self.raw_times: Series = self.raw[LACT_T]
        self.hours: Tensor = Tensor(shifted_hours)
        self.interpolator = LactateInterpolator(self.hours, self.values, interp_method)

    @staticmethod
    def preprocess(raw: DataFrame) -> DataFrame:
        raise NotImplementedError()
        CLIP_MIN, CLIP_MAX = (0,)
        raw.clip()
