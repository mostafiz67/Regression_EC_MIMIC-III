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
from time import ctime
from typing import List, Optional, Tuple

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from scripts.utils.constants import CHART_FILE_ABP_MEAN, LABEVENTS_FILE
from scripts.utils.dataframes import subject_chart_ABP_mean_df, subject_lact_df
from src._logging.base import LOGGER
from src.acquisition.headers.subject import RawSubject
from src.constants import ON_NIAGARA
from src.preprocess.condense import condense_data
from src.preprocess.wave_subject import WaveRawSubject

Result = Optional[Tuple[str, str]]


@dataclass
class Options:
    in_dir: Path
    outdir: Path
    nanclean: bool
    limit: int


@dataclass
class ParallelArgs:
    options: Options
    subject: RawSubject
    chart: Path = CHART_FILE_ABP_MEAN
    labevents: Path = LABEVENTS_FILE


def resolve_path(s: str) -> Path:
    return Path(s).resolve()


def get_options() -> Options:
    parser = ArgumentParser()
    parser.add_argument(
        "--indir",
        "-i",
        type=resolve_path,
        required=True,
        help=(
            "Directory containing header files to process. Must be either the `physionet.org` "
            "directory created by `wget` download or a parent of that directory."
        ),
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=resolve_path,
        required=True,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--drop-border-nans",
        action="store_true",
        help="Whether or not to also drop border NaNs and correct dates prior to saving.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If given, will not process more than `--limit` subjects. Useful for testing.",
    )
    args = parser.parse_args()
    opts = Options(
        in_dir=args.indir, outdir=args.outdir, nanclean=args.drop_border_nans, limit=args.limit
    )
    if not opts.in_dir.exists():
        raise FileNotFoundError(f"Cannot find data to condense in {opts.in_dir}")
    return opts


def condense_raw_subject(args: ParallelArgs) -> Result:
    """Condenses and saves data to disk in varous `.parquet` files. Helper for parallelization."""
    subject: RawSubject = args.subject
    outdir = args.options.outdir
    chart_abp_full = pd.read_parquet(CHART_FILE_ABP_MEAN)
    lactate_full = pd.read_parquet(LABEVENTS_FILE)
    try:
        wave_subject = WaveRawSubject(subject)
        sid = wave_subject.subject.id
        chart_abp: Optional[DataFrame] = subject_chart_ABP_mean_df(sid, chart_abp_full)
        lactate: Optional[DataFrame] = subject_lact_df(sid, lactate_full)
        chart_abp = None if len(chart_abp) == 0 else chart_abp  # type: ignore
        lactate = None if len(lactate) == 0 else lactate  # type: ignore
        condense_data(
            wave_subject=wave_subject,
            lactate=lactate,
            chart_abp=chart_abp,
            drop_border_nans=args.options.nanclean,
            root_outdir=outdir,
            progress=False,
        )
        return None
    except Exception:
        return (subject.id, traceback.format_exc())


def condense_sequentially(args: List[ParallelArgs]) -> List[Result]:
    results = []
    pbar = tqdm(args, total=len(args), desc="Condensing subjects")
    desc = "Condensing subject {sid}"
    subj: RawSubject
    arg: ParallelArgs
    for arg in pbar:
        pbar.set_description(desc.format(arg.subject.id))
        results.append(condense_raw_subject(arg))
    pbar.close()
    return results


def log_results(results: List[Result]) -> None:
    err_sids = []
    for result in results:
        if result is not None:
            sid, message = result
            err_sids.append(sid)
            LOGGER.error(f"Got error for {sid}:")
            LOGGER.error(message)
    if len(err_sids) > 0:
        LOGGER.error(
            f"Got errors when attempting to condense subjects {err_sids}. "
            f"Details above. Finished at {ctime()}"
        )
    else:
        LOGGER.info(f"Finished condensing subjects at {ctime()}")


if __name__ == "__main__":
    opts = get_options()
    # the line below can be expected to take up to 20 minutes on Niagara with 40 cores, and will
    # require up to 40GB of RAM
    subjects = RawSubject.load_all_from_directory(physionet_root=opts.in_dir, limit=opts.limit)
    args = [ParallelArgs(opts, subject) for subject in subjects]

    LOGGER.info(f"Starting to condense subjects at {ctime()}")
    if ON_NIAGARA:  # use parallelism
        # can't use much workers, rapidly run out of memory... not sure why
        # e.g. with 12 workers below memory usage sits near 90GB quite consistently, can
        # spike to 100GB.  Htop # shows something like 40 cores being used also pretty
        # consistently (not clear why). Probably could safely push `max_workers` to 20-24.
        # 16 workers definitely works, but just barely completes in under 1 hour, and only
        # `load_all_from_directory` is already cached.
        results = process_map(
            condense_raw_subject, args, total=len(args), desc="Condensing", max_workers=20
        )
    else:
        results = condense_sequentially(args)
    log_results(results)
