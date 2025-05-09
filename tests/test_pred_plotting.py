# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.environment import ccanada_setup  # isort:skip
ccanada_setup()
# fmt: on

import pickle
import warnings
from math import ceil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from pandas import DataFrame
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src._logging.base import LOGGER
from src.constants import FULL_DATA, INTERP_THRESHOLD, LACT_MAX_MMOL, LOGS
from src.constants import MILLIMOLAR_TO_MGDL as MGDL
from src.constants import PREDICTOR_MEDIAN_OF_IQRS as PRED_IQR
from src.constants import PREDICTOR_MEDIAN_OF_MEDIANS as PRED_MEDIAN
from src.models.deeplearning.arguments import Conv1dArgs, WindowArgs
from src.models.deeplearning.containers.deepsubject import DeepSubject
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.utils import best_rect, window_dimensions
from src.models.deeplearning.validation import ValidationResult, ValidationResults
from src.preprocess.spikes import SpikeRemoval
from src.visualization.utils import add_gradient_patch_legend

# PICKLE = LOGS / "dl_logs/Conv1D/lightning_logs/version_6/results.pickle"
PICKLE = LOGS / "dl_logs/LSTM+prev_lact/lightning_logs/version_66/pred_batches/results07.pickle"
with open(PICKLE, "rb") as handle:
    RESULTS: ValidationResults = pickle.load(handle)

RESULT = RESULTS.results[0]

DPI = 150


def add_to_summary_table(self: ValidationResults) -> None:
    """Create a summary table with subject-level errors, non-interpolated errors, error ranges, etc. Idea
    is this has everything we need to *summarize* results, e.g.

    ~ = interpolated

    sid | ~MAE_mu | ~MAE_sd | ~MAE_max | ~corr(MAE, d) |  MAE_mu | MAE_sd | MAE_max | corr(MAE, d) | n_lact | age | deceased | sex | ... |
    ...

    """
    dfs = []
    with warnings.catch_warnings():
        # if no non-interpolated, we get a bunch of NaN arrays, which we know and expect
        warnings.simplefilter("ignore", RuntimeWarning)
        for result in self.results:
            errs = np.ravel(result.preds - result.targets)
            ds = np.ravel(result.distances)
            idx = ds < INTERP_THRESHOLD

            aes = np.abs(errs)
            naes = np.abs(errs[idx])

            nmae = np.nanmean(naes)
            me = np.mean(errs)
            mae = np.mean(aes)

            nmin, np5, np25, nmed, np75, np95, nmax = (
                np.nanpercentile(naes, [0, 5, 25, 50, 75, 95, 100])
                if len(naes) != 0
                else (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            )
            emin, p5, p25, emed, p75, p95, emax = np.percentile(errs, [0, 5, 25, 50, 75, 95, 100])
            amin, ap5, ap25, amed, ap75, ap95, amax = np.percentile(
                aes, [0, 5, 25, 50, 75, 95, 100]
            )

            naesd = np.std(naes, ddof=1)
            esd = np.std(errs, ddof=1)
            aesd = np.std(aes, ddof=1)

            naecorr = np.corrcoef(ds[idx], aes[idx])[0, 1]
            ecorr = np.corrcoef(ds, errs)[0, 1]
            aecorr = np.corrcoef(ds, aes)[0, 1]

            # ~ indicates interpolated errors, ^ non-interpolated
            dfs.append(
                DataFrame(
                    {
                        "MAE^": nmae,
                        "ME~": me,
                        "MAE~": mae,
                        "r(AE,d)^": naecorr,
                        "r(E,d)~": ecorr,
                        "r(AE,d)~": aecorr,
                        # non-interp AE stats
                        "AE_min^": nmin,
                        "AE_5^": np5,
                        "AE_25^": np25,
                        "AE_50^": nmed,
                        "AE_75^": np75,
                        "AE_95^": np95,
                        "AE_max^": nmax,
                        "AE_sd^": naesd,
                        # signed interpolated error stats
                        "E_min~": emin,
                        "E_5~": p5,
                        "E_25~": p25,
                        "E_50~": emed,
                        "E_75~": p75,
                        "E_95~": p95,
                        "E_max~": emax,
                        "E_sd~": esd,
                        # absolute interpolated error stats
                        "AE_min~": amin,
                        "AE_5~": ap5,
                        "AE_25~": ap25,
                        "AE_50~": amed,
                        "AE_75~": ap75,
                        "AE_95~": ap95,
                        "AE_max~": amax,
                        "AE_sd~": aesd,
                    },
                    index=[result.sid],
                )
            )

    df = pd.concat(dfs, axis=0).sort_values(by=["MAE^", "MAE~"], ascending=False)

    self.logdir = Path(__file__).resolve().parent
    tablefile = self.logdir / "summary.json"
    backup = self.logdir / "summary.backup.json"
    try:
        if tablefile.exists():
            tab = pd.read_json(tablefile)
            tab.to_json(backup)
            df = pd.concat([df, tab], axis=0)
        df.to_json(tablefile)
        LOGGER.info(f"Saved subject error summaries to {tablefile}")
    except ValueError as e:
        if "DataFrame index must be unique" in str(e):
            LOGGER.error(
                "The same subject appears to be in two or more separate predict "
                "batches, and so the summary table cannot be saved."
            )
        else:
            raise RuntimeError(f"Unexpected error: {e}") from e


def plot_preds(
    self: ValidationResult, source: Path, window_args: WindowArgs, label: bool, ax: plt.Axes
):
    decimation = window_args.decimation.value
    subject: DeepSubject = DeepSubject.initialize_from_sids(
        sids=[self.sid],
        source=source,
        predictor_window_size=window_dimensions(
            window_args.desired_predictor_window_minutes.value, decimation
        )[0],
        lag_minutes=window_args.lag_minutes.value,
        target_window_minutes=window_args.target_window_minutes.value,
        decimation=decimation,
        predecimated=True,
        spike_removal=SpikeRemoval.Low,
        interp_method=InterpMethod.linear,
        progress=False,
    )[0]

    # decimate to plot at most roughly 2e5 points per subject
    dec = ceil(self.preds.size / 2e5)
    preds = self.preds[::dec] * LACT_MAX_MMOL * MGDL
    trues = self.targets[::dec] * LACT_MAX_MMOL * MGDL
    y_hours = self.y_hours[::dec]
    pred_max, pred_min = y_hours.max(), y_hours.min()

    ax.set_title(subject.sid, fontsize=8)
    palette = sbn.color_palette("flare", as_cmap=False, n_colors=preds.shape[1])
    # plot preds and trues as given by the pred loader
    for i in range(preds.shape[1]):
        ax.scatter(
            y_hours[:, i],
            preds[:, i],
            color=palette[i],
            s=0.8,
            alpha=0.05,
        )
        ax.scatter(
            y_hours[:, i],
            trues[:, i],
            label="true values (interpolated lactate)" if (i == 0 and label) else None,
            color="black",
            alpha=0.5,
            s=0.5,
        )

    # plot non-interpolated lactate
    lact_hrs = subject.lactate.hours
    idx = (lact_hrs >= pred_min) & (lact_hrs <= pred_max)
    pred_hrs = lact_hrs[idx]
    lact = subject.lactate.values[idx] * LACT_MAX_MMOL * MGDL
    ax.scatter(
        pred_hrs,
        lact,
        color="red",
        marker="X",
        s=10,
        label="non-interpolated lactate" if label else None,
    )

    # plot some interpolated lactate a bit farther forward
    T = np.mean(np.diff(y_hours[:, 0]))
    extend = 0.2 * (pred_max - pred_min)  # extend 20%
    hrs_max = pred_max + extend
    hrs_ext = np.linspace(pred_max, hrs_max, int(np.ceil(extend / T)))
    lact_ext = subject.lactate.interpolator.predict(hrs_ext) * LACT_MAX_MMOL * MGDL
    ax.scatter(
        hrs_ext,
        lact_ext,
        color="black",
        alpha=0.5,
        s=0.5,
    )

    # plot the predictor wave
    wave_vals, wave_hrs = [], []
    for wave in subject.waves:
        hrs = wave.hours
        idx = (hrs >= pred_min) & (hrs <= hrs_max)
        wave_vals.extend(wave.values[idx])
        wave_hrs.extend(hrs[idx])
    wave_vals = np.array(wave_vals)
    wave_hrs = np.array(wave_hrs)
    # de-normalize wave to original mmHg units
    wave_vals *= PRED_IQR
    wave_vals += PRED_MEDIAN

    ax2 = ax.twinx()
    ax2.plot(
        wave_hrs,
        wave_vals,
        color="grey",
        lw=1,
        label="predictor" if label else None,
        alpha=0.25,
    )
    # ensure ABP is in "bottom fifth" of plot and clean, most of the time
    ax2.set_ylim(0, 1000)
    ax2.grid(False)
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    cuts = [50, 100, 150, 300]
    for cut in cuts:
        if preds.max() <= cut:
            ax.set_ylim(0, cut)
            break


def plot_percentiles(
    self: ValidationResult, source: Path, window_args: WindowArgs, label: bool, ax: plt.Axes
):
    decimation = window_args.decimation.value
    subject: DeepSubject = DeepSubject.initialize_from_sids(
        sids=[self.sid],
        source=source,
        predictor_window_size=window_dimensions(
            window_args.desired_predictor_window_minutes.value, decimation
        )[0],
        lag_minutes=window_args.lag_minutes.value,
        target_window_minutes=window_args.target_window_minutes.value,
        decimation=decimation,
        predecimated=True,
        interp_method=InterpMethod.previous,
        progress=False,
    )[0]
    ax.set_title(subject.sid, fontsize=8)

    # idx = np.random.choice(self.preds.shape[0], size=int(1e4))
    preds = self.preds * MGDL
    trues = self.targets * MGDL
    y_hours = self.y_hours
    if preds.ndim == 1:
        preds = np.expand_dims(preds, 1)
        trues = np.expand_dims(trues, 1)
        y_hours = np.expand_dims(y_hours, 1)

    T = window_args.target_window_period_minutes.value  # minutes
    lag = window_args.lag_minutes.value

    pred_hours = [int(lag + T * i) / 60 for i in range(preds.shape[1])]

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc_names = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
    n_percentiles = len(percentiles)
    palette = sbn.color_palette("icefire", as_cmap=False, n_colors=n_percentiles)
    palette[4] = "#ff00e6"

    dense_percentiles = np.linspace(0, 100, 100)
    dense_n_percentiles = len(dense_percentiles)
    dense_palette = sbn.color_palette("icefire", as_cmap=False, n_colors=dense_n_percentiles)

    perc_data = np.empty([preds.shape[1], n_percentiles])
    dense_perc_data = np.empty([preds.shape[1], dense_n_percentiles])
    for i in range(preds.shape[1]):
        # signed errors at distance pred_hours[i]
        errs = preds[:, i] - trues[:, i]
        perc_data[i, :] = np.percentile(errs, percentiles)
        dense_perc_data[i, :] = np.percentile(errs, dense_percentiles)

    # layer out dense percentile underneath with transparency
    for p in range(dense_n_percentiles):
        ax.scatter(
            pred_hours,
            dense_perc_data[:, p],
            color=dense_palette[p],
            s=1.0,
            alpha=0.5,
        )

    for p in range(n_percentiles):
        ax.scatter(
            pred_hours,
            perc_data[:, p],
            color=palette[p],
            s=1.0,
            alpha=0.9,
        )
        ax.plot(
            pred_hours,
            perc_data[:, p],
            color=palette[p],
            lw=2.0,
            alpha=0.9,
            label=perc_names[p] if label else None,
        )


def get_twinx(ax: plt.Axes):
    siblings = ax.get_shared_x_axes().get_siblings(ax)
    for sibling in siblings:
        if sibling.bbox.bounds == ax.bbox.bounds and sibling is not ax:
            return sibling
    return None


def plot_all_preds() -> None:
    nrows, ncols = best_rect(len(RESULTS.results))
    # nrows, ncols = 10, 10
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    # plot_all(RESULT, source=FULL_DATA, window_args=RESULTS.window_args, label=True, ax=ax)
    for i, result in tqdm(
        enumerate(RESULTS.results),
        total=len(RESULTS.results),
        desc="Plotting predictions",
    ):
        # if i > 1:
        #     continue
        plot_preds(
            result, source=FULL_DATA, window_args=RESULTS.window_args, label=i == 0, ax=axes.flat[i]
        )
        if (i + 1 >= ncols) and ((i + 1) % ncols == 0):
            ax: Optional[plt.Axes] = get_twinx(axes.flat[i])
            if ax is not None:
                ax.set_yticks([0, 200, 400, 600, 800, 1000])
                ax.set_yticklabels([0, 200, 400, 600, 800, 1000], fontsize=6)
    # fig.legend()
    fmax = f"{RESULTS.window_args.target_window_minutes.value / 60:0.1f}"
    fmin = f"{RESULTS.window_args.lag_minutes.value / 60:0.1f}"
    add_gradient_patch_legend(
        fig,
        axes.flat[0],
        cmap=sbn.color_palette("flare", as_cmap=True),
        gradient_label=rf"forecast distance $\in$ [{fmin}, {fmax}] hrs",
    )
    samples = RESULTS.train_info.batch_size * RESULTS.train_info.trained_batches
    epochs = RESULTS.train_info.epochs_trained
    batches = RESULTS.train_info.trained_batches
    interp = str(RESULTS.train_loader_args.target_interpolation.value.name)
    fig.suptitle(
        f"Predictions (trained for: {epochs} epochs / {batches} batches / {samples:1.2e} samples)"
        f"\nTraining interpolation: {interp}"
    )
    fig.set_size_inches(w=nrows * 5, h=ncols * 4)
    fig.tight_layout()
    fig.subplots_adjust(left=0.04, bottom=0.05, top=0.92, right=0.96)
    fig.text(x=0.45, y=0.01, s="Time since first measurement (hrs)")
    fig.text(x=0.01, y=0.45, s="lactate (mg/dL)", rotation="vertical")
    fig.text(x=0.988, y=0.45, s="ABP (mmHg)", rotation="vertical")
    model_info = RESULTS.model_args.format()
    fig.text(x=0.005, y=0.945, s=model_info, fontfamily="monospace", fontsize=6)

    outfile = Path(__file__).resolve().parent / "predictions.png"
    print("Saving plot...")
    fig.savefig(outfile, dpi=DPI)
    print(f"Saved plot to {outfile}")
    plt.close()


def plot_all_percentiles() -> None:
    nrows, ncols = best_rect(len(RESULTS.results))
    # nrows, ncols = 10, 10
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    # plot_all(RESULT, source=FULL_DATA, window_args=RESULTS.window_args, label=True, ax=ax)
    for i, result in tqdm(
        enumerate(RESULTS.results),
        total=len(RESULTS.results),
        desc="Plotting prediction percentiles",
    ):
        if i > 0:
            continue
        plot_percentiles(
            result, source=FULL_DATA, window_args=RESULTS.window_args, label=i == 0, ax=axes.flat[i]
        )
    samples = RESULTS.train_info.batch_size * RESULTS.train_info.trained_batches
    epochs = RESULTS.train_info.epochs_trained
    batches = RESULTS.train_info.trained_batches
    interp = str(RESULTS.train_loader_args.target_interpolation.value.name)
    model_info = RESULTS.model_args.format()
    fig.legend()
    fig.set_size_inches(w=nrows * 5, h=ncols * 4)
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, bottom=0.05, top=0.92)
    fig.suptitle(
        f"Predictions (trained for: {epochs} epochs / {batches} batches / {samples:1.2e} samples)"
        f"\nTraining interpolation: {interp}"
    )
    fig.text(x=0.47, y=0.01, s="Prediction distance (hrs)")
    fig.text(x=0.01, y=0.4, s="Signed prediction error percentiles (mg/dL)", rotation="vertical")
    fig.text(x=0.005, y=0.945, s=model_info, fontfamily="monospace", fontsize=6)

    outfile = Path(__file__).resolve().parent / "predictions_percentiles.png"
    print(f"Saving plot...")
    fig.savefig(outfile, dpi=DPI)
    print(f"Saved plot to {outfile}")
    plt.close()


def plot_errors(self: ValidationResults) -> None:
    errors = self.get_error_dfs()[0]
    errors = errors.iloc[::20, :]
    interpolated = self.INTERP_ERR in errors.columns
    sids = errors["sid"].unique()
    n_plots = ceil(len(self.results) / 5)  # don't show more than 10 subjects in a plot
    nrows, ncols = best_rect(n_plots)
    sid_groups = np.array_split(sids, n_plots)  # noqa

    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False)
    for i, (ax, sid_group) in enumerate(zip(axes.flat, sid_groups)):
        subs = errors.query("sid in @sid_group")
        print(f"Plotting subjects with ids: {sid_group}")
        sbn.histplot(
            data=subs,
            x=self.INTERP_ERR if interpolated else self.NONINTERP_ERR,
            hue="sid",
            ax=ax,
            legend=True,
            bins=100,
            binrange=(0, 75),
            stat="probability",
            common_bins=True,
            common_norm=False,
            element="step",
            fill=True,
        )
    axes.flat[0].set_xlim(0, 75)
    axes.flat[0].set_ylim(0, 0.5)
    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")

    inter = "" if interpolated else "Non-Interpolated "
    fig.suptitle(f"Distribution of {inter}Prediction Errors Across Subjects")
    fig.set_size_inches(w=16, h=16)
    info = (
        errors.drop(columns="sid")
        .describe(percentiles=[0.05, 0.95])
        .rename(columns=lambda s: "MAE (mg/dL)")
        .T.to_markdown(tablefmt="simple")
    )
    fig.text(s=info, x=0.33, y=0.94, fontsize=8, fontfamily="monospace")
    fig.text(x=0.43, y=0.01, s="Signed Prediction Error: y_pred - y_true (mg/dL)")
    fig.text(x=0.01, y=0.5, s="Proportion", rotation="vertical")
    fig.subplots_adjust(
        top=0.913, bottom=0.059, left=0.047, right=0.988, hspace=0.065, wspace=0.029
    )
    outfile = (
        Path(__file__).resolve().parent
        / f"{'interpolated' if interpolated else 'non_interpolated'}_error_distributions.pdf"
    )
    fig.savefig(outfile)
    print(f"Distributions saved to {outfile}")
    # plt.show()
    plt.close()


def plot_predictor_error_corrs(
    self: ValidationResult, source: Path, window_args: WindowArgs, label: bool
):
    """Plot the absolute error relative to x properties.

    E.g. predictor values range from x_min to x_max. It is possible the amount of error is a
    function of the extremeness of x.

    Likewise, bas predictions may be correlated with local variability of x, that is, the
    variabilty of the window containing x.
    """
    ax: plt.Axes
    S = 0.5
    fig, axes = plt.subplots(nrows=2, ncols=2)
    decimation = window_args.decimation.value
    subject: DeepSubject = DeepSubject.initialize_from_sids(
        sids=[self.sid],
        source=source,
        predictor_window_size=window_dimensions(
            window_args.desired_predictor_window_minutes.value, decimation
        )[0],
        lag_minutes=window_args.lag_minutes.value,
        target_window_minutes=window_args.target_window_minutes.value,
        decimation=decimation,
        predecimated=True,
        progress=False,
    )[0]

    preds = self.preds
    trues = self.targets
    y_hours = self.y_hours
    if preds.ndim == 1:
        preds = np.expand_dims(preds, 1)
        trues = np.expand_dims(trues, 1)
        y_hours = np.expand_dims(y_hours, 1)

    ax = axes.flat[0]
    ax.set_title(subject.sid)
    # self.x_iqrs.shape = (n_samples, 1)
    # preds.shape = (n_samples, target_size)
    for i in range(preds.shape[1]):
        x_iqr = self.x_iqrs
        err = preds[:, i] - trues[:, i]
        result = linregress(x_iqr.ravel(), err.ravel())
        m = result.slope
        b = result.intercept
        start = x_iqr.min()
        stop = x_iqr.max()
        vals = np.linspace(start, stop, 1000)
        ax.scatter(x_iqr, err, color="black", alpha=0.5, s=S)
        ax.scatter(vals, m * vals + b, color="red", alpha=0.5, s=S)

    ax.set_xlabel("Predictor IQR (mmHg)")
    ax.set_ylabel("Signed Prediction Error (mg/dL)")

    ax = axes.flat[1]
    ax.set_title(subject.sid)
    # self.x_iqrs.shape = (n_samples, 1)
    # preds.shape = (n_samples, target_size)
    for i in range(preds.shape[1]):
        x_sd = self.x_sds
        err = preds[:, i] - trues[:, i]
        result = linregress(x_sd.ravel(), err.ravel())
        m = result.slope
        b = result.intercept
        start = x_sd.min()
        stop = x_sd.max()
        vals = np.linspace(start, stop, 1000)
        ax.scatter(x_sd, err, color="black", alpha=0.5, s=S)
        ax.scatter(vals, m * vals + b, color="red", alpha=0.5, s=S)

    ax.set_xlabel("Predictor SD (mmHg)")
    ax.set_ylabel("Signed Prediction Error (mg/dL)")

    ax = axes.flat[2]
    ax.set_title(subject.sid)
    # self.x_iqrs.shape = (n_samples, 1)
    # preds.shape = (n_samples, target_size)
    for i in range(preds.shape[1]):
        x_mean = self.x_means
        err = preds[:, i] - trues[:, i]
        result = linregress(x_mean.ravel(), err.ravel())
        m = result.slope
        b = result.intercept
        start = x_mean.min()
        stop = x_mean.max()
        vals = np.linspace(start, stop, 1000)
        ax.scatter(x_mean, err, color="black", alpha=0.5, s=S)
        ax.scatter(vals, m * vals + b, color="red", alpha=0.5, s=S)

    ax.set_xlabel("Predictor Max Value (mmHg)")
    ax.set_ylabel("Signed Prediction Error (mg/dL)")

    ax = axes.flat[3]
    ax.set_title(subject.sid)
    # self.x_iqrs.shape = (n_samples, 1)
    # preds.shape = (n_samples, target_size)
    for i in range(preds.shape[1]):
        x_med = self.x_medians
        err = preds[:, i] - trues[:, i]
        result = linregress(x_med.ravel(), err.ravel())
        m = result.slope
        b = result.intercept
        start = x_med.min()
        stop = x_med.max()
        vals = np.linspace(start, stop, 1000)
        ax.scatter(x_med, err, color="black", alpha=0.5, s=S)
        ax.scatter(vals, m * vals + b, color="red", alpha=0.5, s=S)

    ax.set_xlabel("Predictor Median Value (mmHg)")
    ax.set_ylabel("Signed Prediction Error (mg/dL)")

    fig.set_size_inches(w=16, h=16)
    fig.legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)


if __name__ == "__main__":
    sbn.set_style("darkgrid")
    # plot_predictor_error_corrs(
    #     RESULT, source=FULL_DATA, label=True, window_args=RESULTS.window_args
    # )
    # plot_all_percentiles()
    # fig, ax = plt.subplots()
    # plot_preds(RESULT, source=FULL_DATA, label=True, window_args=RESULTS.window_args, ax=ax)
    plot_all_preds()
    # plot_errors(RESULTS)
    # plt.show()
    # add_to_summary_table(RESULTS)
