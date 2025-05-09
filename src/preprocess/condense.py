from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
from pandas import DataFrame
from tqdm import tqdm

from src.constants import CHART_T, CHART_VAL, LACT_T, LACT_VAL, WAVE_VAL
from src.preprocess.wave_subject import WaveRawSubject
from src.preprocess.waveform import Waveform

LACT_OUTNAME = "lactate.parquet"
CHART_ABP_OUTNAME = "chart_ABP_mean.parquet"


def condense_data(
    wave_subject: WaveRawSubject,
    lactate: Optional[DataFrame],
    chart_abp: Optional[DataFrame],
    drop_border_nans: bool,
    root_outdir: Path,
    progress: bool = False,
) -> None:
    """Condenses data for `wave_subject` to .parquet files and saves to `root_outdir`

    Parameters
    ----------
    wave_subject: WaveRawSubject
        Subject to condense.

    lactate: Optional[DataFrame]
        If non-empty, dataframe of lactate times and values.

    chart_abp: Optional[DataFrame]
        If non-empty, dataframe of chart ABP times and values.

    drop_border_nans: bool
        If True, remove border (leading and trailing) NaNs in the waves, and correct start
        times. If False, leave border NaN values.

    root_outdir: Path
        Root output directory. If output directory is e.g. `condensed`, then resultant subfolders
        will have the structure:

            condensed/
            |-- p000033
            |----- lactate.parquet
            |----- chart_ABP_mean.parquet
            |----- waves/
            |--------- 22030803-170244.parquet
            |--------- 22030815-092432.parquet
            |--------- ...

    progress: bool = False
        Whether to show a progress bar that advances as each wave is processed.

    Notes
    -----
    This produces one .parquet file for each header group (i.e. contiguous waveform), one .parquet
    file for the lactate data (if present), and one .parquet file for the chart ABP mean data (if
    present).
    """

    outdir = root_outdir / f"{wave_subject.sid}"
    wave_outdir = outdir / "waves"
    if not outdir.exists():
        wave_outdir.mkdir(exist_ok=True, parents=True)

    groups = wave_subject.subject.header_groups

    if lactate is not None:
        if len(lactate) == 0:
            raise ValueError("Lactate is empty. Pass in `lactate=None` instead.")
        lact_times = np.array(lactate[LACT_T])
        lact_vals = np.array(lactate[LACT_VAL])
        lact_table = pa.table({LACT_T: lact_times, LACT_VAL: lact_vals})
        pa.parquet.write_table(lact_table, outdir / LACT_OUTNAME)

    if chart_abp is not None:
        if len(chart_abp) == 0:
            raise ValueError("Chart ABP mean data is empty. Pass in `chart_abp=None` instead.")
        chart_times = np.array(chart_abp[CHART_T])
        chart_vals = np.array(chart_abp[CHART_VAL])
        chart_table = pa.table({CHART_T: chart_times, CHART_VAL: chart_vals})
        pa.parquet.write_table(chart_table, outdir / CHART_ABP_OUTNAME)

    for group in tqdm(groups, desc="Condensing data", total=len(groups), disable=not progress):
        waveform = Waveform(group)
        result = waveform.as_contiguous(drop_border_nans)
        if result is None:
            continue
        df, start = result
        fname = start.strftime("%Y%m%d-%H%M%S")
        wave_table = pa.table({WAVE_VAL: df[WAVE_VAL]})
        pa.parquet.write_table(wave_table, wave_outdir / f"{fname}.parquet")
