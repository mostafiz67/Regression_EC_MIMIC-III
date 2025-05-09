from warnings import filterwarnings

import numpy as np
from _pytest.capture import CaptureFixture
from tqdm import tqdm

from src.preprocess.rolling_clip import naive_percentile_clip, percentile_clip, slow_percentile_clip

FloatArray = np.ndarray


def test_percentile_clip(capsys: CaptureFixture) -> None:
    filterwarnings("ignore", category=DeprecationWarning)
    a = np.array([1, 2, 3, 4, 20, 5, 6]).astype(np.float64)
    # a = np.array([1, 2, 3, 4, 20, 5, 6])
    # for w in [2, 3, 4, 5, 6]:
    for w in [2, 3, 4, 5, 6]:
        for p in [2.5, 5, 10, 25]:
            np.testing.assert_almost_equal(
                slow_percentile_clip(a, w, p), percentile_clip(a, w, p), err_msg=f"w={w}, p={p}"
            )
    with capsys.disabled():
        for _ in tqdm(range(10)):
            a = np.random.uniform(0, 100, size=np.random.randint(200, 500))
            for w in [2, 3, 4, 5, 100]:
                for p in [2.5, 5, 10, 25]:
                    np.testing.assert_almost_equal(
                        slow_percentile_clip(a, w, p),
                        percentile_clip(a, w, p),
                        err_msg=f"w={w}, p={p}",
                    )


def test_naive_clip(capsys: CaptureFixture) -> None:
    filterwarnings("ignore", category=DeprecationWarning)
    a = np.array([1, 2, 3, 4, 20, 5, 6]).astype(np.float64)
    # a = np.array([1, 2, 3, 4, 20, 5, 6])
    # for w in [2, 3, 4, 5, 6]:
    for w in [2, 3, 4, 5, 6]:
        for p in [2.5, 5, 10, 25]:
            np.testing.assert_almost_equal(
                slow_percentile_clip(a, w, p, "edge"),
                naive_percentile_clip(a, w, p),
                err_msg=f"w={w}, p={p}",
            )
