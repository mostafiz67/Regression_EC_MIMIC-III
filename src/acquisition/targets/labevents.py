from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame

from src._logging.base import LOGGER
from src.acquisition.headers.group import HeaderGroup
from src.acquisition.headers.subject import RawSubject
from src.constants import DATA, MEMORY, RESULTS

ROOT = Path(__file__).resolve().parent

LAB_LEGEND_RAW = DATA / "mimic-iii-clinical-database-1.4/D_LABITEMS.csv.gz"
LAB_LEGEND = RESULTS / "D_LABITEMS.parquet"
POTENTIAL_TARGETS = [
    "Calculated Bicarbonate, Whole Blood",
    "Calculated Total CO2",
    "Carboxyhemoglobin",
    "Chloride, Whole Blood",
    "Free Calcium",
    "Glucose",
    "Hematocrit, Calculated",
    "Hemoglobin",
    "Lactate",
    "Methemoglobin",
    "O2 Flow",
    "Oxygen",
    "Oxygen Saturation",
    "Potassium, Whole Blood",
    "Required O2",
    "Sodium, Whole Blood",
    "Tidal Volume",
    "Ventilation Rate",
    "Ventilator",
    "pCO2",
    "pH",
    "pO2",
    "Estimated Actual Glucose",
    "Absolute Hemoglobin",
    "Ammonia",
    "Barbiturate Screen",
    "Benzodiazepine Screen",
    "Bicarbonate",
    "Bilirubin, Direct",
    "Bilirubin, Indirect",
    "Bilirubin, Total",
    "Calcium, Total",
    "Chloride",
    "Cortisol",
    "Creatinine",
    "Ethanol",
    "Globulin",
    "Glucose",
    "Lactate Dehydrogenase (LD)",
    "Lipase",
    "Methotrexate",
    "Phosphate",
    "Potassium",
    "Sodium",
    "Triglycerides",
    "Urea Nitrogen",
    "Uric Acid",
    "Fetal Hemoglobin",
    "Hematocrit",
    "Hemoglobin",
    "Hemoglobin A2",
    "Hemoglobin C",
    "Hemoglobin F",
    "Hemogloblin A",
    "Hemogloblin S",
    "Heparin",
    "Heparin, LMW",
    "Red Blood Cells",
    "Serum Viscosity",
    "WBC Count",
    "White Blood Cells",
]

TOP_TARGETS = [
    "Glucose",
    "Lactate",
    "Hemoglobin",
    "Hematocrit, Calculated",
    "Ventilation Rate",
    "pCO2",
    "pH",
    "Estimated Actual Glucose",
    "Absolute Hemoglobin",
    "Ammonia",
    "Bicarbonate",
    "Bilirubin, Direct",
    "Bilirubin, Indirect",
    "Bilirubin, Total",
    "Calcium, Total",
    "Chloride",
    "Cortisol",
    "Glucose",
    "Lactate Dehydrogenase (LD)",
    "Phosphate",
    "Potassium",
    "Sodium",
    "Triglycerides",
    "Urea Nitrogen",
    "Uric Acid",
    "Fetal Hemoglobin",
    "Hematocrit",
    "Hemoglobin",
    "Hemoglobin A2",
    "Hemoglobin C",
    "Hemoglobin F",
    "Hemogloblin A",
    "Hemogloblin S",
    "Red Blood Cells",
    "Serum Viscosity",
    "WBC Count",
    "White Blood Cells",
]


def is_junk_target(s: str) -> bool:
    """Flag various obvious junk labels / lab types"""
    JUNK_LABELS = ["specimen", "comments", "void", "hold", "billed", "24", "problem", "other"]
    s = s.lower().strip()
    for label in JUNK_LABELS:
        if label in s:
            return True
    return False


def is_of_interest(s: str) -> bool:
    for label in TOP_TARGETS:
        if label == s:
            return True
    return False


def labevents_legend(force_reprocess: bool = False) -> DataFrame:
    """Load and fix some inconsistencies / NaN problems in the legend, and filter
    to potentially interesting measurements.

    Notes
    -----
    This loads a DataFrame with a752 rows and the (usable) columns:

        - itemid: primary key that appears in the LABEVENTS table
        - label: human readable name / description
        - fluid: source of lab sample
            - 'blood', 'other body fluid', 'ascites', 'csf', 'stool', 'urine',
              'cerebrospinal fluid (csf)', 'joint fluid', 'pleural', 'bone marrow'
        - category:
            - 'blood gas', 'chemistry', 'hematology'
        - loinc_code:
            https://www.mayocliniclabs.com/test-catalog/appendix/loinc-codes.html

    Our predictor waveforms will be generally ABP, PLETH, or *maybe* ECG. There
    is no point predicting things extremely rarely measured / slow to change, or
    that a priori cannot be meaningfully-related to these waves. There are also
    a number of lab measurements that would only be interesting to medical
    experts, so we must dramatically limit the pool of potential targets here.

    I think it thus makes sense to consider only blood and urine as sources of
    prediction targets. Other sources are more medically obscure targets anyway
    (e.g. marrow samples are all "CD10", "CD22" etc). And insofar as we are
    trying to predict *costly* or *invasive* measurements, urine-derived measures
    don't seem like a great target.
    """
    if LAB_LEGEND.exists() and not force_reprocess:
        return pd.read_parquet(LAB_LEGEND)
    df = (
        pd.read_csv(LAB_LEGEND_RAW)
        .rename(columns=lambda s: s.lower())
        .drop(columns=["row_id", "loinc_code"])
    )
    df.fluid = df.fluid.apply(lambda s: s.lower().strip())
    df.category = df.category.apply(lambda s: s.lower().strip())
    df = df.loc[df.fluid == "blood"]  # see notes above
    interesting = df.label.apply(is_of_interest)
    df = df.loc[interesting]
    df.to_parquet(LAB_LEGEND)
    return df


def get_lactate_info() -> pd.DataFrame:
    OUTFILE = RESULTS / "LABEVENTS_lactate_minimal.parquet"
    if OUTFILE.exists():
        return pd.read_parquet(OUTFILE)
    LOGGER.info("Loading lactate data")
    lact = pd.read_csv(RESULTS / "lactate_events.csv").drop(columns=["Unnamed: 0"])
    lact = lact.loc[:, ["SID", "CHARTTIME", "VALUENUM", "FLAG"]]
    LOGGER.info("Converting lactate dates")
    lact["CHARTTIME"] = lact["CHARTTIME"].apply(pd.to_datetime)
    lact.to_parquet(OUTFILE)
    return lact


@MEMORY.cache
def get_contiguous_wave_info() -> pd.DataFrame:
    LOGGER.info("Loading wave data")
    wave = pd.read_csv(ROOT / "contiguous_waveform_data_test.csv").drop(columns=["Unnamed: 0"])
    LOGGER.info("Converting wave dates")
    wave.start = wave.start.apply(pd.to_datetime)
    wave.end = wave.end.apply(pd.to_datetime)
    wave.duration = wave.duration.apply(pd.Timedelta)
    wave = wave.loc[~wave.corrupt_time]
    return wave


def get_lactate_times(
    subject: RawSubject, lact: pd.DataFrame
) -> Tuple[List[HeaderGroup], List[pd.Timestamp], int]:
    SID = int(str(subject.id).replace("p", "").replace("0", ""))
    lact = lact.loc[lact["SUBJECT_ID"] == SID]
    intersects, lact_times = [], []
    count = 0
    for time in pd.to_datetime(lact["CHARTTIME"]):
        intersect = subject.overlapping_groups(time, allow_gap=True)
        if len(intersect) != 0:
            count += 1
            intersects.append(intersect)
            lact_times.append(time)
    return intersects, lact_times, count
