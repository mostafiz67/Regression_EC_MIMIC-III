# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

from src.models.deeplearning.arguments import Arg, LinearArgs, UnpackableArgs, WindowArgs

if __name__ == "__main__":
    loader_args = WindowArgs(
        **dict(
            subjects=5,
            data_source=Path(".").resolve(),
            desired_predictor_window_minutes=5,
            lag_minutes=5,
            target_window_minutes=10,
            target_window_period_minutes=5,
            include_prev_target_as_predictor=True,
            include_predictor_times=True,
            is_validation=True,
            decimation=125,
        )
    )
    linear_args = LinearArgs(**dict(predictor_shape=(3, 100), target_shape=(50,), dropout=0.10))
    test_args = UnpackableArgs()
    setattr(test_args, "subjects", Arg(subjects=10))
    # print(LinearArgs(**loader_args, predictor_shape=(300, 3), target_shape=2))
    print(loader_args)
    print(loader_args.subjects)
    print({**loader_args.subjects})
    print({**loader_args})
    print({**loader_args.tunable()})
    print({**loader_args.untunable()})

    merged = UnpackableArgs.merge([loader_args, linear_args, test_args])
    print(f"{'  Merged Args  ':=^120}")
    print(test_args)
    print(merged)
    print({**merged.tunable()})
    print({**merged.untunable()})
