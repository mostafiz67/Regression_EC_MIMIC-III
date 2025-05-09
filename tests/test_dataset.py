# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from _pytest.capture import CaptureFixture
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.constants import DATA
from src.models.deeplearning.dataloader import WindowArgs, WindowDataset

# SOURCE = DATA / "decimated_250_f32"
SOURCE = DATA / "subset"


@dataclass
class VerifyArgs:
    dataset: WindowDataset
    args: WindowArgs
    index: int


def verify_batch(verify_args: VerifyArgs) -> None:
    dataset = verify_args.dataset
    args = verify_args.args
    i = verify_args.index
    batch = dataset[i]
    if args.is_validation:
        x, y, y_hours, distances, subject = batch
    else:
        x, y, distances = batch

    W = dataset.predictor_size
    if args.include_predictor_times and args.include_prev_target_as_predictor:
        x_shape = (W, 3)
    elif args.include_predictor_times or args.include_prev_target_as_predictor:
        x_shape = (W, 2)
    else:
        x_shape = (W,)  # type: ignore
    y_shape = (dataset.target_size,)

    assert x.shape == x_shape
    assert y.shape == y_shape
    assert distances.shape == y_shape

    if args.include_predictor_times and args.is_validation:
        x_hours = x[:, 0]
        np.testing.assert_almost_equal(
            60 * (x_hours[-1] - x_hours[0]), args.desired_predictor_window_minutes, decimal=1
        )
        np.testing.assert_almost_equal(y_hours[0], x_hours[-1] + args.lag_minutes / 60, decimal=8)


def test_dataset(capsys: Optional[CaptureFixture]) -> None:
    # sids = np.random.choice(IDS, 5, replace=False)
    # sids = ["p000695", "p014863", "p089544", "p063792", "p078366"]
    sids = 20
    loader_args: List[WindowArgs] = [
        WindowArgs(**args)
        for args in (
            ParameterGrid(
                dict(
                    subjects=[sids],
                    data_source=[SOURCE],
                    desired_predictor_window_minutes=[5, 60],
                    lag_minutes=[1, 5, 120],
                    target_window_minutes=[10, 60],
                    target_window_period_minutes=[0, 5, 10],  # must divide all values above
                    # next three cases=True is more complicated than False, so save time by just
                    # testing these much more complex loader configurations
                    include_prev_target_as_predictor=[True],
                    include_predictor_times=[True],
                    is_validation=[True],
                    decimation=[125, 250],
                )
            )
        )
    ]

    ctxt_manager = nullcontext() if capsys is None else capsys.disabled()
    with ctxt_manager:
        args: WindowArgs
        for args in tqdm(loader_args, desc="Testing Dataset params", leave=True):
            dataset = WindowDataset(**args.__dict__, ignore_loading_errors=True)
            idx = min(int(len(dataset) * 0.05), 10)
            checks = list(range(0, idx + 1)) + list(range(len(dataset) - idx, len(dataset)))
            verify_args = [VerifyArgs(dataset, args, index) for index in checks]
            # process_map(verify_batch, verify_args, desc="Verifying batches")
            for vargs in tqdm(verify_args, desc="Verifying batches"):
                verify_batch(vargs)


if __name__ == "__main__":
    MEMOIZER.clear()
    test_dataset(capsys=None)
