import pandas as pd
from pandas import DataFrame

from src.constants import CHART_T, CHART_VAL, LACT_T, LACT_VAL


def subject_lact_df(sid: str, lab_events: DataFrame) -> DataFrame:
    """Returns only the lactate values and times for subject with `sid` from LAB_EVENTS raw data

    Returns
    -------
    df: DataFrame
        Has columns defined by src.constants.LACT_T and src.constants.LACT_VAL containing
        the lactate times and values (unit is mmol/L, see src.constants.LACT_UNIT).
    """
    SID = int(sid.replace("p", "").replace("0", ""))
    lact_sub = lab_events.loc[lab_events["SUBJECT_ID"] == SID]
    lact_sub = lact_sub.sort_values("CHARTTIME")
    lact_times = pd.to_datetime(lact_sub["CHARTTIME"])
    lact_vals = lact_sub["VALUENUM"]
    return DataFrame({LACT_T: lact_times, LACT_VAL: lact_vals})


def subject_chart_ABP_mean_df(sid: str, chart: DataFrame) -> DataFrame:
    """Get ABP mean chart values and times only for subject with `sid` from `chart` CHART_EVENTS
    raw data that has been filtered to include only ABP mean data.

    Returns
    -------
    chart: DataFrame
        Has columns defined by src.constants.CHART_T and src.constants.CHART_VAL containing
        the chart times and (currently) ABP mean values

    Notes
    -----
    We have the filtered chart file only to allow faster local work due to compute / memory
    requirements to sort through full CHART_EVENTS table.
    """
    SID = int(str(sid).replace("p", "").replace("0", ""))
    chart_sub = chart.loc[chart["SUBJECT_ID"] == SID]
    chart_sub = chart_sub.sort_values("CHARTTIME")
    chart_times = pd.to_datetime(chart_sub["CHARTTIME"])
    chart_vals = chart_sub["VALUENUM"]
    return DataFrame({CHART_T: chart_times, CHART_VAL: chart_vals})
