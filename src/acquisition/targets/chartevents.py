# fmt: off
import sys  # isort:skip
from typing import List, Tuple

from src.acquisition.headers.group import HeaderGroup
from src.acquisition.headers.subject import RawSubject

from pathlib import Path  # isort:skip


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import pandas as pd

from src._logging.base import LOGGER

# fmt: on
from src.constants import CLINICAL_DB, DATA


def search_label(label: str) -> Tuple[List[int], List[str]]:
    df = pd.read_csv(CLINICAL_DB / "D_ITEMS.csv.gz", compression="gzip")
    df = df.loc[df["LABEL"].str.contains(label, na=False)]
    ids = list(df["ITEMID"])
    labels = list(df["LABEL"])
    LOGGER.debug(df[["ITEMID", "LABEL"]].to_string())
    return ids, labels


def build_dataframe(id: int, label: str, chunk_size: int = 1000000, verbose: bool = False) -> None:

    if "/" in label:
        label = label.replace("/", " or ")
    OUTPUT_CSV = DATA / f"{label}_ChartEvents.csv"
    OUTPUT_PARQUET = DATA / f"{label}_ChartEvents.parquet"

    LOGGER.debug("Building Dataframe..")

    df = pd.read_parquet(CLINICAL_DB / "CHARTEVENTS_MINIMAL.parquet")
    df = df.loc[df.ITEMID == id]
    df.to_csv(OUTPUT_CSV)
    df.to_parquet(OUTPUT_PARQUET)

    LOGGER.info(f"CSV saved to {OUTPUT_CSV} successfully.")
    LOGGER.info(f"Parquet saved to {OUTPUT_PARQUET} successfully.")


def get_chartevent_info(id: int, label: str) -> pd.DataFrame:
    if "/" in label:
        label = label.replace("/", " or ")
    OUTPUTFILE = DATA / f"{label}_ChartEvents.parquet"
    if not OUTPUTFILE.exists():
        build_dataframe(id, label, verbose=True)
    chart = pd.read_parquet(OUTPUTFILE)
    return chart


def get_chart_events(ids: List[int], labels: List[str]) -> List[pd.DataFrame]:
    chart_events = []
    for chart_target_id, chart_target_label in zip(ids, labels):
        chart_event = get_chartevent_info(chart_target_id, chart_target_label)
        if chart_event.empty:
            continue
        chart_events.append(chart_event)
    return chart_events


def get_chart_timestamp(
    subject: RawSubject, chart: pd.DataFrame
) -> Tuple[List[HeaderGroup], List[pd.Timestamp]]:
    SID = int(str(subject.id).replace("p", "").replace("0", ""))
    chart_sub = chart.loc[chart["SUBJECT_ID"] == SID]
    intersects, chart_times = [], []
    for time in pd.to_datetime(chart_sub["CHARTTIME"]):
        intersect = subject.overlapping_groups(time, allow_gap=True)
        if len(intersect) != 0:
            intersects.append(intersect)
            chart_times.append(time)
    return intersects, chart_times


def get_chart_timestamps(
    subject: RawSubject, chart_events: List[pd.DataFrame]
) -> Tuple[List[List[HeaderGroup]], List[List[pd.Timestamp]]]:
    chart_overlaps, chart_times = [], []

    for chart_event in chart_events:
        chart_overlap, chart_time = get_chart_timestamp(subject, chart_event)
        chart_overlaps.append(chart_overlap)
        chart_times.append(chart_time)

    return chart_overlaps, chart_times
