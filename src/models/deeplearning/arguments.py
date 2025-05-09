from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort:skip
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

import json
import os
import pickle
import platform
import re
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from pytorch_lightning import LightningModule, Trainer

from src.constants import DEV_RAND_VARIABLE_SIDS, DEV_SIDS, FULL_DATA
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.utils import window_dimensions
from src.preprocess.spikes import SpikeRemoval

"""
We will ultimately be tuning and exploring a variety of model configurations and setting in order to
optimize our performance on the MIMIC data. In order to do this efficiently and without waste given
our limited resources, we need to carefuly set up infrastructure to handle logging of key results
(e.g. model performances, runtimes) along with their respective configurations.
"""

DEFAULT_SOURCE = FULL_DATA
DEFAULT_BATCH = 1024
DEFAULT_WORKERS = 1

Phase = Literal["train", "val", "test", "pred"]
T = TypeVar("T")


class SmoothingMethod(Enum):
    Lowpass = "lowpass"
    Mean = "mean"
    Median = "median"
    Savgol = "savgol"


class Pooling(Enum):
    Max = "max"
    Avg = "avg"


class Arg(Mapping, Generic[T]):
    """Container for arguments and tracking trainability / not.

    Usage: arg = Arg(name=value).tuned()
    To get dict: **arg.
    """

    def __init__(self, **kwargs: Mapping[str, T]) -> None:
        if len(kwargs) != 1:
            raise ValueError("")
        value: T
        name, value = list(kwargs.items())[0]  # type: ignore
        setattr(self, name, value)
        self.dict: Dict[str, T] = {name: value}  # type: ignore
        self.is_tunable: bool = False
        self.value_: T = value

    @property
    def value(self) -> T:
        return self.value_

    @value.setter
    def value(self, val: T) -> None:
        self.value_ = val

    @value.getter
    def value(self) -> T:
        return self.value_

    def as_tunable(self) -> TunableArg:
        return TunableArg[T](**self.dict)  # type: ignore # mypy too dumb to understand this is fine

    def __iter__(self) -> Iterator[str]:
        return iter(self.dict)

    def __len__(self) -> int:
        return len(self.dict)

    def __getitem__(self, item: str) -> T:
        return self.dict[item]

    def __str__(self) -> str:
        name, value = list(self.dict.items())[0]
        val = f'Path("{value}")' if isinstance(value, Path) else value
        return f"{self.__class__.__name__}[ {name}={val} ]"

    __repr__ = __str__


class TunableArg(Arg, Mapping, Generic[T]):
    def __init__(self, **kwargs: Mapping[str, T]) -> None:
        super().__init__(**kwargs)
        self.is_tunable: bool = True


# argparse converters


def to_bool(arg: str) -> bool:
    orig = arg
    arg = arg.lower()
    if arg in ["true", "t", "1"]:
        return True
    if arg in ["false", "f", "0"]:
        return False
    raise ValueError(f"`{orig}` is not valid for a boolean argument.")


def smooth(arg: str) -> Union[int, float]:
    return float(arg) if ("." in str(arg) or "e" in str(arg)) else int(arg)


def subj(arg: str) -> Union[int, Optional[List[str]], Literal["random"]]:
    orig = arg
    arg = arg.lower()
    if "random" in arg:
        return "random"
    if "p" in arg:
        sids = re.sub(r"[\[\]\(\)\s]", "", arg.lower()).split(",")
        return [s for s in sids if len(s) > 0]
    raise ValueError(f"Could not parse `subjects` arg: {orig}")


# https://stackoverflow.com/questions/33402220/custom-double-star-operator-for-a-class
class UnpackableArgs(Mapping):
    TUNABLES: List[str] = []
    DEFAULTS: Dict[str, Any] = {}
    TYPES: Dict[str, Any] = {}

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.dict: Dict[str, Any] = {}
        name: str
        for name, val in kwargs.items():
            arg = Arg[T](**{name: val})
            if name in self.TUNABLES:
                arg = arg.as_tunable()
            setattr(self, name, arg)
            self.dict[name] = arg

    def tunable(self) -> Dict[str, Any]:
        """Return only tunable args"""
        arg: Arg
        d: Dict[str, Any] = dict()
        for key, arg in self.dict.items():
            if arg.is_tunable:
                d.update(**arg)
        return d

    def untunable(self) -> Dict[str, Any]:
        """Return only untunable args"""
        arg: Arg
        d: Dict[str, Any] = dict()
        for key, arg in self.dict.items():
            if not arg.is_tunable:
                d.update(**arg)
        return d

    @classmethod
    def default(cls, **kwargs: Any) -> UnpackableArgs:
        parser = ArgumentParser()
        for argname, default in cls.DEFAULTS.items():
            parser.add_argument(f"--{argname}", type=cls.TYPES[argname], default=default)
        args, _ = parser.parse_known_args()
        # if kwargs includes a value that is itself unpackable:
        pops = []
        for key in kwargs.keys():
            if key not in cls.DEFAULTS:
                pops.append(key)
                continue
            try:
                # ensure we have the right type
                raw = kwargs[key]
                if raw is None:
                    continue
                converter = cls.TYPES[key]
                if converter is to_bool and isinstance(raw, bool):
                    continue
                if converter is smooth and isinstance(raw, SmoothingMethod):
                    continue
                else:
                    kwargs[key] = converter(kwargs[key])
            except TypeError as e:
                raise RuntimeError(
                    f"Could not parse argument `{key}` with value {kwargs[key]}"
                ) from e
        for key in pops:
            kwargs.pop(key, None)

        return cls(
            **{
                **vars(args),
                **kwargs,
            }
        )  # type: ignore

    @classmethod
    def merge(
        cls: Type[UnpackableArgs], unpackable_args: Sequence[UnpackableArgs]
    ) -> UnpackableArgs:
        """Return a new dict with all arguments from `unpackable_args`.
        Each `UnpackableArgs` in `unpackable_args` must be from a distinct type."""
        types = set([arg.__class__.__name__ for arg in unpackable_args])
        if len(types) < len(unpackable_args):
            raise ValueError(
                "Cannot merge multiple copies of the same arguments. Each "
                "element of `unpackable_args` must be a unique subclass of `UnpackableArgs`"
            )
        arg: Arg
        new = cls()
        names = set()
        for unpackable in unpackable_args:
            for name, arg in unpackable.dict.items():
                if name in names:
                    val = arg.dict.pop(name)
                    name = f"{unpackable.__class__.__name__.lower()}_{name}"
                    arg.dict[name] = val
                else:
                    names.add(name)
                setattr(new, name, arg)
        return new

    def to_json(self, path: Path) -> None:
        if not path.parent.exists():
            path.parent.mkdir(exist_ok=True, parents=True)

        with open(path, "w") as handle:
            json.dump({**self}, handle)

    def to_pickle(self, path: Path) -> None:
        if not path.parent.exists():
            path.parent.mkdir(exist_ok=True, parents=True)

        with open(path, "wb") as handle:
            pickle.dump({**self}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, path: Path, **overrides: Any) -> UnpackableArgs:
        if not path.exists():
            return cls.default(**overrides)

        with open(path, "rb") as handle:
            return cls.default(**{**pickle.load(handle), **overrides})

    def __iter__(self) -> Iterator:
        arg: Arg
        d: Dict[str, Any] = dict()
        for key, arg in self.dict.items():
            d.update(**arg)
        return iter(d)

    def __len__(self) -> int:
        return len(self.dict)

    def __getitem__(self, item: Any) -> Any:
        arg = self.dict[item]
        return arg.value

    def __str__(self) -> str:
        fmt = [f"{self.__class__.__name__}("]
        for i, (key, arg) in enumerate(self.dict.items()):
            comma = "," if i != len(self.dict) - 1 else ""
            name, value = list(arg.dict.items())[0]
            arg_str = f"{arg.__class__.__name__:>10}: {name}={value}"
            fmt.append(f"  {arg_str}{comma}")
        fmt.append(")")
        return "\n".join(fmt)

    __repr__ = __str__

    def format(self) -> str:
        return str({**self})


class ProgramArgs(UnpackableArgs):
    """Class for arguments related to environment and files/IO"""

    progress_bar_refresh_rate: Arg[int]
    trainer_args: Arg[Namespace]

    @classmethod
    def default(cls, **kwargs: Any) -> UnpackableArgs:
        """Handle this manually due to interaction with Lightning Parser"""
        cc_cluster = os.environ.get("CC_CLUSTER")
        pbar = cc_cluster is None

        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        trainer_args = parser.parse_known_args()[0]

        args = dict(
            progress_bar_refresh_rate=1,
            enable_progress_bar=pbar,
            trainer_args=trainer_args,
        )

        return cls(**{**args, **kwargs})  # type: ignore


class TrainerArgs(UnpackableArgs):
    """Class for arguments related to trainer and files/IO"""

    progress_bar_refresh_rate: Arg[int]
    default_root_dir: Arg[Path]
    gpus: Arg[int]

    def __init__(
        self,
        model_cls: Type[LightningModule],
        **kwargs: Mapping[str, Any],
    ) -> None:
        super().__init__(**kwargs)  # type: ignore
        self.default_root_dir.value = ROOT / f"dl_logs/{model_cls.__name__}"
        self.gpus.value = 1 if platform.system().lower() == "inux" else 0
        if not self.default_root_dir.value.exists():
            self.default_root_dir.value.mkdir(exist_ok=True, parents=True)


class DataArgs(UnpackableArgs):
    """NOTE: Follow the pattern of this class for creating new sets of arguments.

    subjects: Arg[Optional[Union[int, Sequence[str]]]] = DEV_SIDS[phase]
    data_source: Arg[Path] = DEFAULT_SOURCE
    predecimated: Arg[bool] = True
    target_interpolation: TunableArg[InterpMethod] = InteprMethod.previous
    target_dropout: TunableArg[float] = 0
        How often to randomly zero target information when
        `include_prev_target_as_predictor` is True
    batch_size: TunableArg[int] = DEFAULT_BATCH
    shuffle: Arg[bool] = phase == "train",
    drop_last: Arg[bool] = True
    num_workers: Arg[int] = DEFAULT_WORKERS
    is_validation: Arg[bool] = phase == "pred",
    preshuffle: Arg[bool] = False
    ignore_loading_errors: Arg[bool] = True
    """

    subjects: Arg[Optional[Union[int, Sequence[str]]]]
    data_source: Arg[Path]
    predecimated: Arg[bool]
    target_interpolation: TunableArg[InterpMethod]
    target_dropout: TunableArg[float]
    batch_size: TunableArg[int]
    shuffle: Arg[bool]
    drop_last: Arg[bool]
    num_workers: Arg[int]
    is_validation: Arg[bool]
    preshuffle: Arg[bool]
    ignore_loading_errors: Arg[bool]

    TUNABLES = ["batch_size"]

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

    @property
    def dataset_args(self) -> Dict[str, Any]:
        return dict(
            subjects=self.subjects.value,
            data_source=self.data_source.value,
            predecimated=self.predecimated.value,
            target_interpolation=self.target_interpolation.value,
            target_dropout=self.target_dropout.value,
            is_validation=self.is_validation.value,
            preshuffle=self.preshuffle.value,
            ignore_loading_errors=self.ignore_loading_errors.value,
        )

    @property
    def loader_args(self) -> Dict[str, Any]:
        return dict(
            batch_size=self.batch_size.value,
            shuffle=self.shuffle.value,
            drop_last=self.drop_last.value,
            num_workers=self.num_workers.value,
        )

    @classmethod
    def default(  # type: ignore
        cls,
        phase: Phase,
        source: Path = DEFAULT_SOURCE,
        subjects: Union[Optional[List[str]], Literal["random"]] = None,
        **kwargs: Any,
    ) -> DataArgs:
        """
        data_source=source,
        predecimated=True,
        batch_size=DEFAULT_BATCH,
        drop_last=True,
        num_workers=DEFAULT_WORKERS,
        ignore_loading_errors=True,
        subjects=DEV_SIDS[phase],
        shuffle=phase == "train",
        is_validation=phase == "pred",
        target_interpolation=InterpMethod.previous,
        target_dropout=0,
        preshuffle=False,
        """
        SHARED_LOADER_ARGS = dict(
            data_source=source,
            predecimated=True,
            batch_size=DEFAULT_BATCH,
            drop_last=True,
            num_workers=DEFAULT_WORKERS,
            preshuffle=False,
            ignore_loading_errors=True,
        )
        DEFAULTS = dict(
            **SHARED_LOADER_ARGS,
            **dict(
                subjects=None,
                shuffle=phase == "train",
                is_validation=phase == "pred",
                target_interpolation=InterpMethod.previous,
                target_dropout=0,
            ),
        )
        TYPES = dict(
            data_source=Path,
            predecimated=to_bool,
            batch_size=int,
            drop_last=to_bool,
            num_workers=int,
            preshuffle=to_bool,
            ignore_loading_errors=to_bool,
            subjects=subj,
            shuffle=to_bool,
            is_validation=to_bool,
            target_interpolation=InterpMethod,
            target_dropout=float,
        )

        _phase = str(phase).lower()
        if phase == "predict":
            _phase = "pred"  # type: ignore
        if isinstance(subjects, str) and subjects == "random":
            subjects = DEV_RAND_VARIABLE_SIDS[_phase]
        else:
            subjects = subjects or DEV_SIDS[_phase]

        parser = ArgumentParser()
        for argname, default in DEFAULTS.items():
            parser.add_argument(f"--{argname}", type=TYPES[argname], default=default)
        args, _ = parser.parse_known_args()
        args.subjects = subjects
        return DataArgs(**{**vars(args), **kwargs})  # type: ignore


class PreprocArgs(UnpackableArgs):
    """
    smoothing_method: TunableArg[SmoothingMethod] = None
    smoothing_value: TunableArg[Union[int, float]] = 1.0
    spike_removal: TunableArg[Optional[SpikeRemoval]] = None
    """

    local_smoothing_method: TunableArg[Optional[SmoothingMethod]]
    local_smoothing_value: TunableArg[Union[int, float]]
    spike_removal: TunableArg[Optional[SpikeRemoval]]

    TUNABLES = ["local_smoothing_method", "local_smoothing_value", "spike_removal"]
    # fmt: off
    DEFAULTS = dict(
        local_smoothing_method=None,
        local_smoothing_value=1.0,
        spike_removal=None
    )
    TYPES = dict(
        local_smoothing_method=SmoothingMethod,
        local_smoothing_value=smooth,
        spike_removal=SpikeRemoval,
    )
    # fmt: on

    def __init__(self, **kwargs: Mapping[str, object]) -> None:
        super().__init__(**kwargs)  # type: ignore


class WindowArgs(UnpackableArgs):
    """NOTE: Follow the pattern of this class for creating new sets of arguments.

    data_source: Arg[Path] = None
    desired_predictor_window_minutes: TunableArg[float] = 120
    lag_minutes: TunableArg[float] = 0
    target_window_minutes: TunableArg[int] = 1440
    target_window_period_minutes: TunableArg[int] = 10
    include_prev_target_as_predictor: TunableArg[bool] = True
    include_predictor_times: TunableArg[bool] = False
    decimation: TunableArg[int] = 500
    """

    desired_predictor_window_minutes: TunableArg[float]
    lag_minutes: TunableArg[float]
    target_window_minutes: TunableArg[int]
    target_window_period_minutes: TunableArg[int]
    include_prev_target_as_predictor: TunableArg[bool]
    include_predictor_times: TunableArg[bool]
    decimation: TunableArg[int]

    TUNABLES = [
        "desired_predictor_window_minutes",
        "lag_minutes",
        "target_window_minutes",
        "target_window_period_minutes",
        "include_prev_target_as_predictor",
        "include_predictor_times",
        "decimation",
    ]
    DEFAULTS = dict(
        desired_predictor_window_minutes=120,
        lag_minutes=0,
        target_window_minutes=1440,
        target_window_period_minutes=10,
        include_prev_target_as_predictor=True,
        include_predictor_times=False,
        decimation=500,
    )
    TYPES = dict(
        desired_predictor_window_minutes=float,
        lag_minutes=float,
        target_window_minutes=int,
        target_window_period_minutes=int,
        include_prev_target_as_predictor=to_bool,
        include_predictor_times=to_bool,
        decimation=int,
    )

    def __init__(self, **kwargs: Mapping[str, object]) -> None:
        super().__init__(**kwargs)  # type: ignore

    @property
    def predictor_shape(self) -> Tuple[int, int]:
        """Returns predictor shape as (seq_length, n_channels)"""
        predictor_size = window_dimensions(
            self.desired_predictor_window_minutes.value, self.decimation.value
        )[0]
        predictor_shape = [predictor_size, 1]
        if self.include_predictor_times.value:
            predictor_shape[1] += 1
        if self.include_prev_target_as_predictor.value:
            predictor_shape[1] += 1
        return cast(Tuple[int, int], tuple(predictor_shape))

    @property
    def target_shape(self) -> Tuple[int]:
        """Returns target shape as (seq_length,)"""
        y_W = self.target_window_period_minutes.value
        if not isinstance(y_W, int):
            raise TypeError("`target_window_period_minutes` must be an integer.")
        if self.target_window_period_minutes.value != 0 and (
            self.target_window_minutes.value % y_W != 0
        ):
            raise ValueError(
                "Target window period must divide `target_window_minutes` without remainder."
            )
        target_size = int(self.target_window_minutes.value // y_W + 1 if y_W != 0 else 1)
        target_shape = (target_size,)
        return target_shape


class EvaluationArgs(UnpackableArgs):
    """Program args related to final costly validation step (e.g. plots)

    subjects_per_batch: Arg[int] = Arg[20]
        Limit prediction to `subjects_per_batch` at a time. E.g. If generating
        final plots for 50 subjects, and setting `subjects_per_batch=20`, then
        subject batches will be size [17, 17, 16]. This makes prediction take
        longer, but is necessary to prevent RAM explosion.

    limit_pred_subjects: Arg[int] = Arg[None]
        Only run predictions on `limit_pred_subjects` subjects total.

    estimate_runtime: bool = True
        Whether or not to automate batch size and runtime testing.

    val_interval_hrs: Arg[float] = Arg[1/3]
        How often we want validation loops during training.
    """

    subjects_per_batch: Arg[int]
    limit_pred_subjects: Arg[int]
    estimate_runtime: Arg[bool]
    val_interval_hrs: Arg[float]

    DEFAULTS = dict(
        subjects_per_batch=20,
        limit_pred_subjects=None,
        estimate_runtime=True,
        val_interval_hrs=1 / 3,  # 20 minutes
    )

    TYPES = dict(
        subjects_per_batch=int,
        limit_pred_subjects=int,
        estimate_runtime=to_bool,
        val_interval_hrs=float,
    )


class GenericDeepLearningArgs(UnpackableArgs):
    """
    weight_decay: TunableArg[float] = 1e-5
        L2 regularization for Adam

    lr_init: TunableArg[float] = 1e-3
        Initial learning rate for Adam

    lr_step: TunableArg[int] = 1000
        Reduce learning rate by factor of `lr_decay` every `lr_step` batches

    lr_decay: TunableArg[float] = 0.8
        Reduce learning rate by factor of `lr_decay` every `lr_step` batches
    """

    weight_decay: TunableArg[float]
    lr_init: TunableArg[float]
    lr_step: TunableArg[Optional[float]]
    lr_decay: TunableArg[Optional[float]]

    TUNABLES = [
        "weight_decay",
        "lr_init",
        "lr_step",
        "lr_decay",
    ]
    DEFAULTS = dict(  # type: ignore
        weight_decay=1e-5,
        lr_init=1e-3,
        lr_step=1000,
        lr_decay=0.8,
    )

    TYPES = dict(  # type: ignore
        weight_decay=float,
        lr_init=float,
        lr_step=float,
        lr_decay=float,
    )

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)


class LinearArgs(UnpackableArgs):
    """Arguments for simple linear model"""

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)


class SineLinearArgs(UnpackableArgs):
    """
    sine_features: TunableArg[int] = 64
    trainable: TunableArg[bool] = False
    scale: TunableArg[float] = 1.0
    """

    sine_features: TunableArg[int]
    trainable: TunableArg[bool]
    scale: TunableArg[float]

    TUNABLES = [
        "sine_features",
        "trainable",
        "scale",
    ]
    DEFAULTS = dict(
        sine_features=64,
        trainable=False,
        scale=1.0,
    )
    TYPES = dict(
        sine_features=int,
        trainable=to_bool,
        scale=float,
    )

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)


class MLPArgs(UnpackableArgs):
    """Arguments for simple MLP deep learning model"""


class LstmArgs(UnpackableArgs):
    """
    resize: TunableArg[Optional[int]] = None
        If not None, resize input series length to `resize`.

    num_layers: TunableArg[Literal[1, 2, 4, 6, 8]] = 2
        Number of LSTM layers to stack. Dramatically increases training time.

    num_hidden: TunableArg[Literal[16, 32, 64, 128, 256, 512]] = 32
        Number of hidden layers in each LSTM of the stacked LSTM.

    proj_size: TunableArg[int] = 0
        Projection size (see docs). Reduces model memory footprint as proj_size
        approaches `num_hidden` but also seems to dramatically increase runtime.

    bidirectional: TunableArg[bool] = False
        Whether or not to use a bidirectional LSTM.

    num_linear_layers: TunableArg[Literal[1, 2, 4, 8]] = 1
        Depth of final MLP.

    linear_width: TunableArg[Literal[32, 64, 128, 256, 512]] = 32
        Size of layers (except first and last) in final MLP.

    shared_linear: TunableArg[bool] = False
        Whether or not to use shared weights (Conv1D with kernel_size=1) in
        final MLP layers. Also dramatically reduces memory

    gap: TunableArg[bool] = False
        Whether to reduce LSTM outputs along the sequence (spatial) dimension
        prior to the MLP layer. Current reduction is mean.

    use_full_seq: TunableArg[bool] = False
        If True, use all LSTM outputs in final MLP layer. E.g. uses all of
        `o`, where `o, (h, c) = lstm(x)`. Otherwise, just use the final hidden
        state, e.g. use `o[:, -1, :] == h[-1]`.

    mish: TunableArg[bool] = False
        If True, use Mish activation in final MLP. Otherwise use ReLU.

    dropout: TunableArg[float] = 0
        Dropout to include between LSTM stacks

    time2vec: TunableArg[bool] = False
        Not implemented.
    """

    resize: TunableArg[Optional[int]]
    # lstm depth / breadth
    num_layers: TunableArg[Literal[1, 2, 4, 6, 8]]
    num_hidden: TunableArg[Literal[16, 32, 64, 128, 256, 512]]
    proj_size: TunableArg[int]
    bidirectional: TunableArg[bool]
    # linear depth / breadth
    num_linear_layers: TunableArg[Literal[1, 2, 4, 8]]
    linear_width: TunableArg[Literal[32, 64, 128, 256, 512]]
    shared_linear: TunableArg[bool]
    # architectural
    gap: TunableArg[bool]
    use_full_seq: TunableArg[bool]
    mish: TunableArg[bool]
    dropout: TunableArg[float]
    # time encoding
    time2vec: TunableArg[bool]

    TUNABLES = [
        "resize",
        "num_layers",
        "num_hidden",
        "proj_size",
        "bidirectional",
        "num_linear_layers",
        "linear_width",
        "shared_linear",
        "gap",
        "use_full_seq",
        "mish",
        "dropout",
        "time2vec",
    ]
    DEFAULTS = dict(
        resize=None,
        num_layers=2,
        num_hidden=32,
        proj_size=0,
        bidirectional=False,
        num_linear_layers=1,
        linear_width=32,
        shared_linear=False,
        gap=False,
        use_full_seq=False,
        mish=False,
        dropout=0,
        time2vec=False,
    )

    TYPES = dict(
        resize=int,
        num_layers=int,
        num_hidden=int,
        proj_size=int,
        bidirectional=to_bool,
        num_linear_layers=int,
        linear_width=int,
        shared_linear=to_bool,
        gap=to_bool,
        use_full_seq=to_bool,
        mish=to_bool,
        dropout=float,
        time2vec=to_bool,
    )

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def format(self) -> str:
        d = Namespace(**self)
        s = f""" {f'num_layers={d.num_layers}':<20}{f'num_linear={d.num_linear_layers}':<20}{f'resize={d.resize}':<20}
{f'num_hidden={d.num_hidden}':<20}{f'linear_width={d.linear_width}':<20}{f'gap={d.gap}':<20}
{f'proj_size={d.proj_size}':<20}{f'shared_linear={d.shared_linear}':<20}{f'mish={d.mish}':<20}
{f'bidirectional={d.bidirectional}':<20}{f'use_full_seq={d.use_full_seq}':<20}{f'dropout={d.dropout}':<20}"""  # noqa
        return s


class Conv1dArgs(UnpackableArgs):
    """
    resize: TunableArg[Optional[int]] = None
        If not None, resize input series length to `resize`.

    in_kernel_size: TunableArg[int] = 7
        Size of kernel of input layer.

    in_dilation: TunableArg[int] = 1
        Size of dilation of input layer.

    in_out_ch: TunableArg[int] = 64
        Number of output channels in input layer. Determines width of entire network.

    kernel_size: TunableArg[int] = 3
        Kernel size for non-input convolutions.

    dilation: TunableArg[int] = 1
        Dilation for non-input convolutions.

    depthwise: TunableArg[bool] = False
        If True, set groups=in_channels and do depthwise separable convolutions.
        Otherwise keep groups=1.

    pad: TunableArg[bool] = True
        If True, Conv1D layers other than the input have `padding="same"`. If False,
        only the input layer has `padding="same"`.

    channel_expansion: TunableArg[Literal[1, 2]] = 1
        If `1`, double channels every conv layer. If `2`, double channels every 2 layers.

    max_channels: TunableArg[int] = 1024
        Maximum amount of channels to double to.

    num_conv_layers: TunableArg[int] = 4
        How many conv layers total. Should be even.

    num_linear_layers: TunableArg[int] = 1
        If `gap=True`, this is ignored / set to 1. Otherwise, determines depth
        of final MLP.

    linear_width: TunableArg[int] = 256
        If `gap=True`, this is ignored. Otherwise, determines size of each
        linear layer in final MLP.

    shared_linear: TunableArg[bool] = True
        If True, uses shared weights for final linear layers (e.g. Conv with
        kernel_size=1).

    gap: TunableArg[bool] = True
        If True, uses a global average pooling layer and one final linear
        layer for the output. Otherwise output will be an MLP.

    mish: TunableArg[bool] = False
        If True, use Mish (https://arxiv.org/abs/1908.08681), else use ReLU.

    dropout: TunableArg[Optional[float]] = None
        If not None, include spatial dropout layers every `dropout_frequency` conv layers
        after the final LeakyReLU activation.

    dropout_freq: TunableArg[Literal[1, 2, 4, 8]] = 4

    pooling: TunableArg[bool] = True
        If True, include max pooling every `pooling_freq` conv layers.

    pooling_freq: TunableArg[Literal[1, 2, 4, 8]] = 1
        See above.

    pooling_type: TunableArg[Pooling] = Pooling.Max
        How to pool in pooling layers.
    """

    # input params
    resize: TunableArg[Optional[int]]
    in_kernel_size: TunableArg[int]
    in_dilation: TunableArg[int]
    in_out_ch: TunableArg[int]

    # conv params
    kernel_size: TunableArg[int]
    dilation: TunableArg[int]
    depthwise: TunableArg[bool]
    pad: TunableArg[bool]

    # depth / breadth params
    num_conv_layers: TunableArg[Literal[4, 8, 16, 20, 24, 28, 32]]
    channel_expansion: TunableArg[Literal[1, 2, 4, 8]]
    max_channels: TunableArg[int]
    num_linear_layers: TunableArg[Literal[1, 2, 4, 8]]
    linear_width: TunableArg[Literal[32, 64, 128, 256, 512]]
    shared_linear: TunableArg[bool]

    # architecture params
    gap: TunableArg[bool]
    mish: TunableArg[bool]
    dropout: TunableArg[Optional[float]]
    dropout_freq: TunableArg[Literal[1, 2, 4, 8]]
    pooling: TunableArg[bool]
    pooling_freq: TunableArg[Literal[1, 2, 4, 8]]
    pooling_type: TunableArg[Pooling]

    TUNABLES = [
        "resize",
        "in_kernel_size",
        "in_dilation",
        "in_out_ch",
        "kernel_size",
        "dilation",
        "depthwise",
        "pad",
        "num_conv_layers",
        "channel_expansion",
        "max_channels",
        "num_linear_layers",
        "linear_width",
        "shared_linear",
        "gap",
        "mish",
        "dropout",
        "dropout_freq",
        "pooling",
        "pooling_freq",
        "pooling_type",
    ]
    DEFAULTS = dict(
        resize=None,  #
        in_kernel_size=7,  #
        in_dilation=1,  #
        in_out_ch=64,  #
        kernel_size=3,  #
        dilation=1,  #
        depthwise=False,  #
        pad=True,
        num_conv_layers=4,  #
        channel_expansion=1,  #
        max_channels=1024,  #
        gap=True,  #
        mish=False,  #
        num_linear_layers=1,  #
        linear_width=256,  #
        shared_linear=True,  #
        dropout=None,  #
        dropout_freq=4,  #
        pooling=True,  #
        pooling_freq=1,  #
        pooling_type=Pooling.Max,
    )

    TYPES = dict(
        resize=int,
        in_kernel_size=int,
        in_dilation=int,
        in_out_ch=int,
        kernel_size=int,
        dilation=int,
        depthwise=to_bool,
        pad=to_bool,
        num_conv_layers=int,
        channel_expansion=int,
        max_channels=int,
        gap=to_bool,
        mish=to_bool,
        num_linear_layers=int,
        linear_width=int,
        shared_linear=to_bool,
        dropout=float,
        dropout_freq=int,
        pooling=to_bool,
        pooling_freq=int,
        pooling_type=Pooling,
    )

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def format(self) -> str:
        d = Namespace(**self)
        s = f"""{f'resize={d.resize}':<20}{f'kernel_size={d.kernel_size}':<20}{f'max_channels={d.max_channels}':<20}{f'gap={d.gap}':<20}{f'pad={d.pad}':<20}
{f'in_kernel_size={d.in_kernel_size}':<20}{f'dilation={d.dilation}':<20}{f'dropout={d.dropout}':<20}{f'num_linear={d.num_linear_layers}':<20}
{f'in_dilation={d.in_dilation}':<20}{f'depthwise={d.depthwise}':<20}{f'dropout_freq={d.dropout_freq}':<20}{f'linear_width={d.linear_width}':<20}
{f'in_out_ch={d.in_out_ch}':<20}{f'num_conv_layers={d.num_conv_layers}':<20}{f'pooling={d.pooling}':<20}{f'shared_linear={d.shared_linear}':<20}
{f'mish={d.mish}':<20}{f'channel_exp={d.channel_expansion}':<20}{f'pool_freq={d.pooling_freq}':<20}{f'pool_type={d.pooling_type}':<20}"""  # noqa
        return s


class ConvGamArgs(UnpackableArgs):
    scalar_bias: TunableArg[bool]
    scalar_max: TunableArg[bool]
    g_args: TunableArg[Conv1dArgs]

    TUNABLES = ["scalar_bias", "scalar_max"] + Conv1dArgs.TUNABLES
    DEFAULTS = {
        **dict(
            scalar_bias=False,
            scalar_max=False,
        ),
        **Conv1dArgs.DEFAULTS,
    }
    TYPES = {
        **dict(
            scalar_bias=to_bool,
            scalar_max=to_bool,
        ),
        **Conv1dArgs.TYPES,
    }

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_g_args(self) -> Conv1dArgs:
        d = {**self}
        d.pop("scalar_bias", None)
        d.pop("scalar_max", None)
        d.pop("g_args", None)
        return Conv1dArgs(**d)

    def format(self) -> str:
        d = Namespace(**self)
        s = f"""{f'scalar_bias={d.scalar_bias}':<20}{f'scalar_max={d.scalar_max}':<20}\n"""
        return s + self.get_g_args().format()


class LstmGamArgs(UnpackableArgs):
    scalar_bias: TunableArg[bool]
    scalar_max: TunableArg[bool]
    g_args: TunableArg[LstmArgs]

    TUNABLES = ["scalar_bias", "scalar_max"] + LstmArgs.TUNABLES
    DEFAULTS = {
        **dict(
            scalar_bias=False,
            scalar_max=False,
        ),
        **LstmArgs.DEFAULTS,
    }
    TYPES = {
        **dict(
            scalar_bias=to_bool,
            scalar_max=to_bool,
        ),
        **LstmArgs.TYPES,
    }

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_g_args(self) -> LstmArgs:
        d = {**self}
        d.pop("scalar_bias", None)
        d.pop("scalar_max", None)
        d.pop("g_args", None)
        return LstmArgs(**d)

    def format(self) -> str:
        d = Namespace(**self)
        s = f"""{f'scalar_bias={d.scalar_bias}':<20}{f'scalar_max={d.scalar_max}':<20}\n"""
        return s + self.get_g_args().format()


if __name__ == "__main__":
    loader_args: WindowArgs = WindowArgs(
        **dict(  # type: ignore
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
    # print(LinearArgs(**loader_args, predictor_shape=(300, 3), target_shape=2))
    print(loader_args)
    print(loader_args.decimation)
    print({**loader_args.decimation})
    print({**loader_args})
    print({**loader_args.tunable()})
    print({**loader_args.untunable()})

    loader_args2 = deepcopy(loader_args)
    loader_args.decimation.value = 500
    assert loader_args2.decimation.value == 125

    try:
        preproc = PreprocArgs.default(local_smoothing_method="meank")
        print(preproc)
    except ValueError:
        pass

    try:
        eval_args = EvaluationArgs.default(estimate_runtime="pig")
        print(eval_args)
    except ValueError as e:
        assert "`pig` is not valid for a boolean" in str(e)
