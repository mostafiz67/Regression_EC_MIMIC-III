# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

"""
Converts existing condensed data in `.parquet` form to a reduced (decimated) copy.
"""

FloatFmt = Literal["f16", "f32", "f64"]

DECIMATION = 25
FLOATFMT = 32
ROOT = Path(__file__).resolve().parent / "condensed data"
OUTDIR = ROOT.parent
if not OUTDIR.exists():
    OUTDIR.mkdir(exist_ok=True, parents=True)


@dataclass
class Options:
    decimation: int
    floatfmt: FloatFmt
    indir: Path
    outdir: Path

    def __init__(self, decimation: int, floatfmt: FloatFmt, outdir: Path, indir: Path) -> None:
        self.decimation: int = decimation
        self.floatfmt: FloatFmt = f"f{floatfmt}"  # type: ignore
        if "16" in self.floatfmt:
            self.ftype = np.float16
        elif "32" in self.floatfmt:
            self.ftype = np.float32
        elif "64" in self.floatfmt:
            self.ftype = np.float64
        else:
            raise ValueError("Invalid float type.")
        self.indir = indir

        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        self.outdir: Path = outdir / f"decimated_{self.decimation}_{self.floatfmt}"


@dataclass
class ParallelArgs:
    options: Options
    subj_dir: Path


def get_options() -> Options:
    parser = ArgumentParser()
    parser.add_argument("--indir", "-i", type=Path, help="Subject source folder")
    parser.add_argument("--outdir", "-o", type=Path, default=OUTDIR)
    parser.add_argument(
        "--decimation", "-d", type=int, default=DECIMATION, help="Amount to decimate"
    )
    parser.add_argument(
        "--floatfmt", "-f", type=int, default=FLOATFMT, help="Float format to convert to"
    )
    args = parser.parse_args()
    return Options(**args.__dict__)


def decimate(args: ParallelArgs) -> None:
    outdir = args.options.outdir / args.subj_dir.name
    if not outdir.exists():
        outdir.mkdir(exist_ok=True, parents=True)
    files = sorted(args.subj_dir.rglob("*.parquet"))

    for file in files:
        if "lactate" in file.name or "chart" in file.name:
            outfile = outdir / file.name
            copyfile(file, outfile)
        else:
            wave_dir = outdir / "waves"
            outfile = wave_dir / file.name
            if not wave_dir.exists():
                wave_dir.mkdir(exist_ok=True, parents=True)
            df = pd.read_parquet(file)
            df = df.iloc[:: args.options.decimation, :].astype(args.options.ftype)
            df.to_parquet(outfile)


if __name__ == "__main__":
    """Returns decimated data for faster coding purposes"""
    options = get_options()
    args = [ParallelArgs(options, p.parent) for p in sorted(options.indir.rglob("lactate.parquet"))]
    process_map(decimate, args)
