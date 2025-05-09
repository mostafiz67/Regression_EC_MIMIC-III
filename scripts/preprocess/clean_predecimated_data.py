# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from src._logging.base import LOGGER
from src.constants import CHART_T, CHART_VAL, CLEAN_DECIMATED_DATA, LACT_T, LACT_VAL, WAVE_VAL
from src.models.deeplearning.containers.deepsubject import DeepSubject

"""Cleans the decimated predictor (waveform) data by removing border NaNs and interpolating the NaNs
in middle with previous value.

Notes
-----
Using environment-linked ENV_LINKED_IDS variable, i.e. if run on Compute Canada, will load and
process all subjects, but otherwise, will limit to a fast testing subset. To override this behaviour
use `--all`.
"""

ROOT_OUTDIR = None


@dataclass
class Options:
    in_dir: Path
    outdir: Path
    decimation: int


def resolve_path(s: str) -> Path:
    return Path(s).resolve()


def get_options() -> Options:
    parser = ArgumentParser()
    parser.add_argument(
        "--indir",
        "-i",
        type=resolve_path,
        required=True,
        help="Directory containing files to process. Should have folders p000123 and etc.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=resolve_path,
        default=CLEAN_DECIMATED_DATA,
        help="Directory for output files. Will be created if not present.",
    )
    parser.add_argument(
        "--decimation",
        "-d",
        type=int,
        help="Decimation to use",
    )
    args = parser.parse_args()
    opts = Options(in_dir=args.indir, outdir=args.outdir, decimation=args.decimation)
    if not opts.in_dir.exists():
        raise FileNotFoundError(f"Cannot find data to clean in {opts.in_dir}")
    return opts


def clean_subject(subject: DeepSubject, root_outdir: Path) -> Union[str, None]:
    try:
        outdir = root_outdir / f"{subject.id}"
        if not outdir.exists():
            outdir.mkdir(exist_ok=True)

        stem = CLEAN_DECIMATED_DATA / f"{subject.id}"
        lact_outfile = stem / "lactate_df.parquet"
        chart_outfile = stem / "chart_df.parquet"

        for wave in subject.waves:
            if wave.is_empty():
                continue
            else:
                cleaned_wave, wave_start = wave.cleaned(method="previous")
                wave_start = pd.to_datetime(str(wave_start)).strftime("%Y%m%d-%H%M%S")
                wave_table = pa.table({WAVE_VAL: cleaned_wave})
                if wave_table is not None:  # fix the start time of the wave
                    wave_outfile = stem / (wave_start + ".parquet")
                    pa.parquet.write_table(wave_table, wave_outfile)
                else:
                    continue

        lact_table = pa.table(
            {LACT_T: subject.lactate.data[LACT_T], LACT_VAL: subject.lactate.data[LACT_VAL]}
        )
        pa.parquet.write_table(lact_table, lact_outfile)

        chart_table = pa.table(
            {CHART_T: subject.chart.data[CHART_T], CHART_VAL: subject.chart.data[CHART_VAL]}
        )
        pa.parquet.write_table(chart_table, chart_outfile)
        return None
    except Exception:
        return traceback.format_exc()


def run(options: Options) -> None:
    ROOT_OUTDIR = options.outdir
    if not ROOT_OUTDIR.exists():
        LOGGER.info(f"Creating root directory {ROOT_OUTDIR} for clean decimated data..")
        ROOT_OUTDIR.mkdir(exist_ok=True, parents=True)

    subject: DeepSubject
    subjects = DeepSubject.initialize_all_from_directory_unfiltered(
        options.in_dir, options.decimation, predecimated=False
    )
    errors: List[Tuple[str, str]] = []
    pbar = tqdm(subjects, total=len(subjects), desc="Removing NaNs")
    desc = "Removing NaNs for {sid}"
    for subject in pbar:
        pbar.set_description(desc.format(sid=subject.id))
        result = clean_subject(subject, ROOT_OUTDIR)
        if result is not None:
            errors.append((subject.id, result))

    for _, err in errors:
        LOGGER.error(err)
    LOGGER.error(f"Got errors when processing sids {[sid for sid, _ in errors]}. Details above.")


if __name__ == "__main__":
    options = get_options()
    run(options)
