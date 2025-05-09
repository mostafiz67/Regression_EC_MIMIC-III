from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame
from pytorch_lightning import Trainer
from tqdm import tqdm

from src._logging.base import LOGGER
from src.constants import INTERP_THRESHOLD, LACT_MAX_MMOL
from src.constants import MILLIMOLAR_TO_MGDL as MGDL
from src.constants import PREDICTOR_MEDIAN_OF_IQRS as PRED_IQR
from src.constants import PREDICTOR_MEDIAN_OF_MEDIANS as PRED_MEDIAN
from src.models.deeplearning.arguments import DataArgs, PreprocArgs, UnpackableArgs, WindowArgs
from src.models.deeplearning.containers.deepsubject import DeepSubject
from src.models.deeplearning.containers.lactate import InterpMethod
from src.models.deeplearning.utils import best_rect, window_dimensions
from src.visualization.utils import add_gradient_patch_legend

ON_CCANADA = os.environ.get("CC_CLUSTER") is not None
DPI = 150


@dataclass
class TrainingInfo:
    logdir: Path
    epochs_trained: int
    trained_batches: int
    validated_batches: int
    total_batches: int
    batch_size: int

    def __init__(self, trainer: Trainer, train_loader_args: DataArgs) -> None:
        self.logdir = Path(trainer.logger.experiment.log_dir)  # type: ignore
        self.trained_batches = trainer._fit_loop.epoch_loop.batch_progress.total.completed
        self.validated_batches = (
            trainer._fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.total.completed
        )
        self.total_batches = self.trained_batches + self.validated_batches
        self.epochs_trained = trainer._fit_loop.epoch_progress.total.completed
        self.batch_size = train_loader_args.batch_size.value


@dataclass
class ValidationResult:
    sid: str
    preds: ndarray
    targets: ndarray
    distances: ndarray
    y_hours: ndarray

    def plot_preds(
        self,
        source: Path,
        predecimated: bool,
        window_args: WindowArgs,
        preproc_args: PreprocArgs,
        plot_interp_method: InterpMethod,
        label: bool,
        ax: plt.Axes,
    ) -> None:
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
            predecimated=predecimated,
            spike_removal=preproc_args.spike_removal.value,
            interp_method=plot_interp_method,
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
        self,
        source: Path,
        predecimated: bool,
        window_args: WindowArgs,
        plot_interp_method: InterpMethod,
        label: bool,
        ax: plt.Axes,
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
            predecimated=predecimated,
            spike_removal=None,
            interp_method=plot_interp_method,
            progress=False,
        )[0]
        ax.set_title(subject.sid, fontsize=8)

        preds = self.preds * LACT_MAX_MMOL * MGDL
        trues = self.targets * LACT_MAX_MMOL * MGDL
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

        cuts = [10, 25, 75, 100, 150, 300]
        ax.set_ylim(max(-50, -cuts[-1]), 300)
        for cut in cuts:
            if preds.max() <= cut:
                ax.set_ylim(max(-50, -cut), cut)
                break


def get_twinx(ax: plt.Axes):
    siblings = ax.get_shared_x_axes().get_siblings(ax)
    for sibling in siblings:
        if sibling.bbox.bounds == ax.bbox.bounds and sibling is not ax:
            return sibling
    return None


class ValidationResults:
    INTERP_ERR = "Signed Prediction Error: y_pred - y_true (mg/dL)"
    NONINTERP_ERR = "Non-Interpolated Signed Prediction Error: y_pred - y_true (mg/dL)"

    def __init__(
        self,
        results: List[ValidationResult],
        logdir: Path,
        pred_idx: int,
        train_info: TrainingInfo,
        model_args: UnpackableArgs,
        window_args: WindowArgs,
        train_loader_args: DataArgs,
        pred_loader_args: DataArgs,
        train_metrics: Optional[DataFrame] = None,
        val_metrics: Optional[DataFrame] = None,
        lr_metrics: Optional[DataFrame] = None,
    ) -> None:
        self.results: List[ValidationResult] = results
        self.logdir: Path = logdir
        self.pred_idx: int = pred_idx
        self.train_info = train_info
        self.model_args = model_args
        self.window_args = window_args
        self.train_loader_args = train_loader_args
        self.pred_loader_args = pred_loader_args
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.lr_metrics = lr_metrics

        self.data_source: Path = self.pred_loader_args.data_source.value
        self.decimation: int = self.window_args.decimation.value

    def add_to_summary_table(self: ValidationResults) -> None:
        """Create a summary table with subject-level errors, non-interpolated errors, error ranges, etc"""
        dfs = []
        with warnings.catch_warnings():
            # if no non-interpolated, we get a bunch of NaN arrays, which we know and expect
            warnings.simplefilter("ignore", RuntimeWarning)
            for result in self.results:
                errs = np.ravel(result.preds - result.targets) * LACT_MAX_MMOL * MGDL
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
                emin, p5, p25, emed, p75, p95, emax = np.percentile(
                    errs, [0, 5, 25, 50, 75, 95, 100]
                )
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
        tablefile = self.logdir / "summary.json"
        backup = self.logdir / "summary.backup.json"
        try:
            if tablefile.exists():
                LOGGER.info(f"Found existing stats at {tablefile}. Updating...")
                tab = pd.read_json(tablefile)
                tab.to_json(backup)
                df = pd.concat([df, tab], axis=0)
            else:
                LOGGER.info(f"Creating new error stats table at {tablefile}...")
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

    def plot_all_preds(self) -> None:
        axes: plt.Axes
        ax: plt.Axes
        result: ValidationResult
        interp = self.pred_loader_args.target_interpolation.value

        nrows, ncols = best_rect(len(self.results))
        sbn.set_style("darkgrid")
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=False
        )
        if not hasattr(self, "preproc_args"):
            try:
                preproc_args = PreprocArgs.from_pickle(self.logdir / "hparams/preproc.hparams.pickle")
            except:
                preproc_args = PreprocArgs.default()
        else:
            preproc_args = self.preproc_args
        for i, (result, ax) in tqdm(
            enumerate(zip(self.results, axes.flat)),
            desc="Plotting predictions",
            total=len(self.results),
            disable=ON_CCANADA,
        ):
            result.plot_preds(
                source=self.data_source,
                predecimated=True,
                window_args=self.window_args,
                preproc_args=preproc_args,
                plot_interp_method=interp,
                label=i == 0,
                ax=ax,
            )
            if (i + 1 >= ncols) and ((i + 1) % ncols == 0):
                ax: Optional[plt.Axes] = get_twinx(axes.flat[i])
                if ax is not None:
                    ax.set_yticks([0, 200, 400, 600, 800, 1000])
                    ax.set_yticklabels([0, 200, 400, 600, 800, 1000], fontsize=6)
        fmax = f"{self.window_args.target_window_minutes.value / 60:0.1f}"
        fmin = f"{self.window_args.lag_minutes.value / 60:0.1f}"
        add_gradient_patch_legend(
            fig,
            axes.flat[0],
            cmap=sbn.color_palette("flare", as_cmap=True),
            gradient_label=f"forecast distance in [{fmin}, {fmax}] hrs",
        )
        samples = self.train_info.batch_size * self.train_info.trained_batches
        epochs = self.train_info.epochs_trained
        batches = self.train_info.trained_batches
        interp = str(self.train_loader_args.target_interpolation.value.name)
        fig.suptitle(
            f"Predictions (trained for: {epochs} epochs / {batches} batches / {samples:1.2e} samples)"
            f"\nTraining interpolation: {interp}"
        )
        width = max(nrows * 5, 15)
        fig.set_size_inches(w=width, h=1 + ncols * 4)
        fig.tight_layout()
        fig.subplots_adjust(left=0.04, bottom=0.05, top=0.92, right=0.96)
        fig.text(x=0.45, y=0.01, s="Time since first measurement (hrs)")
        fig.text(x=0.01, y=0.45, s="lactate (mg/dL)", rotation="vertical")
        fig.text(x=0.988, y=0.45, s="ABP (mmHg)", rotation="vertical")
        model_info = self.model_args.format()
        fig.text(x=0.005, y=0.945, s=model_info, fontfamily="monospace", fontsize=6)

        plotdir = self.logdir / "plots"
        outdir = plotdir / "predictions"
        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / f"predictions{self.pred_idx:02d}.png"
        fig.savefig(outfile, dpi=DPI)
        LOGGER.info(f"Plot saved to {outfile}")
        plt.close()

    def plot_all_percentiles(self) -> None:
        axes: plt.Axes
        ax: plt.Axes
        result: ValidationResult
        interp = self.pred_loader_args.target_interpolation.value
        nrows, ncols = best_rect(len(self.results))
        sbn.set_style("darkgrid")
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        for i, result in tqdm(
            enumerate(self.results),
            total=len(self.results),
            desc="Plotting percentiles",
            disable=ON_CCANADA,
        ):
            result.plot_percentiles(
                source=self.data_source,
                predecimated=True,
                window_args=self.window_args,
                plot_interp_method=interp,
                label=i == 0,
                ax=axes.flat[i],
            )
        samples = self.train_info.batch_size * self.train_info.trained_batches
        epochs = self.train_info.epochs_trained
        batches = self.train_info.trained_batches
        interp = str(self.train_loader_args.target_interpolation.value.name)
        model_info = self.model_args.format()
        fig.legend()
        width = max(nrows * 5, 15)
        fig.set_size_inches(w=width, h=1 + ncols * 4)
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, bottom=0.05, top=0.92)
        fig.suptitle(
            f"Predictions (trained for: {epochs} epochs / {batches} batches / {samples:1.2e} samples)"
            f"\nTraining interpolation: {interp}"
        )
        fig.text(x=0.47, y=0.01, s="Prediction distance (hrs)")
        fig.text(
            x=0.01, y=0.4, s="Signed prediction error percentiles (mg/dL)", rotation="vertical"
        )
        fig.text(x=0.005, y=0.945, s=model_info, fontfamily="monospace", fontsize=6)

        plotdir = self.logdir / "plots"
        outdir = plotdir / "percentiles"
        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        outfile = outdir / f"prediction_percentiles{self.pred_idx:02d}.png"
        LOGGER.info("Saving prediction percentiles plot...")
        fig.savefig(outfile, dpi=DPI)
        LOGGER.info(f"Saved plot to {outfile}")
        plt.close()

    def summarize(self, hists: bool = True, show: bool = True) -> None:
        if not hists:
            # self.plot_metrics(block=True, show=show)
            return
        # self.plot_metrics(block=False, show=show)

    def plot_error_distributions(self, interpolated: bool = False) -> None:
        df_ae, df_ae_non_interp = self.get_error_dfs()
        LOGGER.info("Plotting non-interpolated error distributions")
        self.plot_errors(df_ae_non_interp)
        # plotting interpolated errors is slow, shows less than predictions
        # and percentile plots anyway
        if interpolated:
            LOGGER.info(f"Plotting interpolated error distributions")
            self.plot_errors(df_ae)

    def plot_learning_metrics(self, block: bool, show: bool = True) -> None:
        raise NotImplementedError()
        sbn.set_style("darkgrid")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title("Training and Validation Metrics")
        ax.set_xlabel("batch")
        ax.set_ylabel("Metric value")
        ax2 = ax.twinx()
        ax2.set_ylabel("Pearson correlation between MAE and interpolation distance")
        ax2.set_ylim(-1, 1)

        if self.train_metrics is not None:
            train_mae = self.train_metrics["train/mae (mg/dL)"]
            train_steps = self.train_metrics["step"]
            train_mae_noninterp = self.train_metrics["train/mae (non-interp) (mg/dL)"]
            ax.plot(
                train_steps,
                train_mae,
                label="train_mae (mg/dL)",
                color="#212121",
                linestyle="dashed",
            )
            ax.plot(
                train_steps,
                train_mae_noninterp,
                label="train_mae (non-interp) (mg/dL)",
                color="#000",
            )
        if self.val_metrics is not None:
            val_mae = self.val_metrics["val/mae (mg/dL)"]
            val_corr = self.val_metrics["val/spearman(d, mae)"]
            val_mae_noninterp = self.val_metrics["val/mae (non-interp) (mg/dL)"]
            val_steps = self.val_metrics["step"]
            if len(val_steps) > 1:
                ax.plot(
                    val_steps, val_mae, label="val_mae (mg/dL)", color="#1457ff", linestyle="dashed"
                )
                ax2.plot(val_steps, val_corr, label="corr(d, mae)", color="#ffa914")
                ax.plot(
                    val_steps,
                    val_mae_noninterp,
                    label="val_mae (non-interp) (mg/dL)",
                    color="#1457ff",
                )
            elif len(val_steps) == 1:
                ax.scatter(val_steps, val_mae, label="val_mae (mg/dL)", color="#1457ff", s=5)
                ax2.scatter(val_steps, val_corr, label="corr(d, mae)", color="#ffa914", s=5)
                ax.scatter(
                    val_steps,
                    val_mae_noninterp,
                    label="val_mae (non-interp) (mg/dL)",
                    color="#fd4949",
                    s=5,
                )
        fig.text(s=str(self.window_args), x=0.05, y=0.05, fontsize=8, fontfamily="monospace")
        model_args = str(self.model_args)
        model_args_len = len(model_args.split("\n"))
        window_args_len = len(str(self.window_args).split("\n"))
        if model_args_len < window_args_len:
            newline = "\n"  # can't have \ in f-string
            model_args = f"{model_args}{newline * (window_args_len - model_args_len)}"
        fig.text(s=model_args, x=0.3, y=0.05, fontsize=8, fontfamily="monospace")
        fig.subplots_adjust(bottom=0.2)
        fig.set_size_inches(w=14, h=12)
        fig.legend(loc=(0.75, 0.08))
        outfile = self.logdir / "metrics_plot.pdf"
        fig.savefig(outfile)
        LOGGER.info(f"Plot saved to {outfile}")
        if ON_CCANADA:
            plt.close()
            return
        if show:
            plt.show(block=block)
        else:
            plt.close()

    def get_error_dfs(self) -> Tuple[DataFrame, DataFrame]:
        aes = []
        non_interps = []
        for result in self.results:
            ae = np.abs(result.preds - result.targets) * LACT_MAX_MMOL * MGDL
            idx = result.distances < INTERP_THRESHOLD
            ae_non_interp = ae[idx]

            # desc = DataFrame(aes).describe(percentiles=[0.05, 0.95]).drop(columns="count")
            df = DataFrame(ae.ravel(), columns=[self.INTERP_ERR])
            df2 = DataFrame(ae_non_interp.ravel(), columns=[self.NONINTERP_ERR])
            df["sid"] = result.sid
            df2["sid"] = result.sid
            aes.append(df)
            non_interps.append(df2)
        df_ae = pd.concat(aes, axis=0, ignore_index=True)
        df_ae_non_interp = pd.concat(non_interps, axis=0, ignore_index=True)
        return df_ae, df_ae_non_interp

    def plot_errors(self, errors: DataFrame) -> None:
        N_BINS, BIN_MAX, PROB_MAX = 100, 75, 0.5
        interpolated = self.INTERP_ERR in errors.columns
        sids = errors["sid"].unique()
        n_plots = ceil(len(self.results) / 5)  # don't show more than 5 subjects in a plot
        nrows, ncols = best_rect(n_plots)
        sid_groups = np.array_split(sids, n_plots)  # noqa  # is used by `query` below

        sbn.set_style("darkgrid")
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False)
        for ax, sid_group in zip(axes.flat, sid_groups):
            subs = errors.query("sid in @sid_group")
            LOGGER.info(f"Plotting subjects with ids: {sid_group}")
            sbn.histplot(
                data=subs,
                x=self.INTERP_ERR if interpolated else self.NONINTERP_ERR,
                hue="sid",
                ax=ax,
                legend=True,
                bins=N_BINS,
                binrange=(0, BIN_MAX),
                stat="probability",
                common_bins=True,
                common_norm=False,
                element="step",
                fill=True,
            )
        axes.flat[0].set_xlim(0, BIN_MAX)
        axes.flat[0].set_ylim(0, PROB_MAX)
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
        plotdir = self.logdir / "plots"
        outdir = plotdir / "distributions"
        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        outfile = (
            outdir
            / f"{'interpolated' if interpolated else 'non_interpolated'}_error_distributions{self.pred_idx:02d}.pdf"
        )
        fig.savefig(outfile)
        LOGGER.info(f"Plot saved to {outfile}")
        plt.close()
