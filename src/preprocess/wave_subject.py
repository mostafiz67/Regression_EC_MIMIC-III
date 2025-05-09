import gc
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy import interpolate, signal
from scipy.optimize import minimize_scalar

from src._logging.base import LOGGER
from src.acquisition.headers.subject import RawSubject
from src.constants import CHART_T, CHART_VAL, CUTOFF_FREQ, ON_NIAGARA
from src.preprocess.utils import clip_wave, filter_wave, optimal_lag
from src.preprocess.waveform import Waveform


class WaveRawSubject:
    """Wraps an acquisition RawSubject and provides some convenience functions / data access
    related to subject waveforms."""

    def __init__(self, subject: RawSubject) -> None:
        self.subject = subject
        self.sid = self.subject.path.name
        self._waveforms: Optional[List[Waveform]] = None

    @property
    def waveforms(self) -> List[Waveform]:
        if self._waveforms is None:
            self._waveforms = [Waveform(g) for g in self.subject.header_groups]
        return self._waveforms

    def compute_time_discrepancies(
        self,
        chart_df: pd.DataFrame,
        med_filter_size: int = 1,
        perc_window_hrs: float = 0.25,
    ) -> Tuple[List[float], List[float]]:
        """Estimates discrepancies in timstamps by comparing (aligning) raw waveforms with
        `chart_df` waveforms. Raw waveforms are filtered and smoothed to improve alignment.

        Parameters
        ----------
        chart_df: pd.Dataframe
            Contains time values and chart values (in this case ABP mean) w.r.t. the subject.

        med_filter_size: int
            Median filter window size for smoothing.

        perc_window_hrs: float
            Window size (in hrs) for rolling percentile-based clipping.

        Returns
        -------
        discrepancies: List[float]
            Discrepancy estimates for each contiguous filtered wave after comparison with
            CHART_EVENTS

        clip_discrepancies: List[float]
            Discrepancy estimates for each contiguous filtered and clipped wave after
            comparison with CHART_EVENTS
        """
        groups = self.subject.header_groups
        t0 = min([g.master.start for g in groups])
        # Scaling chart time values w.r.t first time point of the subject
        chart_times = np.asarray(
            ((chart_df[CHART_T] - t0) / pd.Timedelta(hours=1)), dtype=np.float64
        )
        chart_vals = np.array(chart_df[CHART_VAL])

        discrepancies, clip_discrepancies = [], []
        for g, group in enumerate(groups):
            waveform = Waveform(group)
            df = waveform.as_contiguous()[1]
            if df is None:
                continue

            LOGGER.info("Interpolating NaNs...")
            df.interpolate("nearest", inplace=True)
            t_wave = np.asarray(((df.t - t0) / pd.Timedelta(hours=1)))
            wave = np.asarray(df.wave, dtype=np.float32)

            LOGGER.info("Bandpass filtering wave...")
            sos = signal.butter(4, CUTOFF_FREQ, output="sos")
            filtered = filter_wave(wave=wave, sos=sos, med_window_size=med_filter_size)

            LOGGER.info("Clipping wave extreme values...")
            copy_filtered_wave = np.copy(filtered)
            enough_wave_data = clip_wave(
                t_wave, wave, copy_filtered_wave, window_hrs=perc_window_hrs, percentile=30
            )
            if not enough_wave_data:
                LOGGER.info(
                    f"Header group {g} for subject {self.sid} is too small "
                    f"to clip given window size of {perc_window_hrs} hrs"
                )
                continue

            y3 = signal.sosfiltfilt(sos, copy_filtered_wave).astype(np.float32)
            del copy_filtered_wave  # copies can be large so this is a precaution
            gc.collect()

            LOGGER.info("Computing discrepancy estimates...")
            interp_args = dict(bounds_error=False, fill_value=0)
            min_args = dict(bounds=(-5, 5), method="Bounded", options=dict(disp=True))
            interpolator = interpolate.interp1d(chart_times, chart_vals ** interp_args)
            chart_interp = interpolator(t_wave)

            LOGGER.info("Computing estimates for wave..")
            compute_lag = lambda lag: optimal_lag(
                t_wave=t_wave, lag=lag, filtered_wave=filtered, chart_interp=chart_interp
            )
            res = minimize_scalar(compute_lag, **min_args)
            if np.isnan(res.fun):
                LOGGER.info(f"Skipped discepancy estimation for subject {self.sid} group {g}")
                continue

            LOGGER.info("Computing estimates for clipped wave..")
            compute_clip_lag = lambda lag: optimal_lag(
                t_wave=t_wave, lag=lag, filtered_wave=y3, chart_interp=chart_interp
            )
            res_clip = minimize_scalar(compute_clip_lag, **min_args)
            if not np.isnan(res_clip.fun):
                discrepancies.append(res.x)
                clip_discrepancies.append(res_clip.x)

        return discrepancies, clip_discrepancies

    def plot_waveform(
        self,
        times: ndarray,
        filtered_wave: ndarray,
        chart_interp: ndarray,
        y3: ndarray,
        lag: int = 1,
        clip_lag: int = 1,
    ) -> None:
        """Plot the contiguous filtered and filtered clipped wave with the best time lag estimation.

        Parameters
        ----------
        times: ndarray
            Wave time values
        filtered wave: ndarray
            Flitered wave values
        chart_interp: ndarray
            Interpolated chart time values
        y3: ndarray
            Clipped filtered values
        lag: int
            Best time lag estimation value for filtered wave
        clip: int
            Best time lag estimation value for filtered and clipped wave

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        if ON_NIAGARA:
            ax[0].plot(times, filtered_wave, color="black", lw=0.5)
            ax[0].set_xlabel("Filtered smoothed wave")
            ax[1].plot(times, y3, color="black", lw=0.5)
            ax[1].set_xlabel("Clipped smoothed wave")
        else:
            fig.suptitle(f"{self.sid}")
            ax[0].set_title("Filtered smoothed wave")
            ax[0].plot(times[::100], filtered_wave[::100], color="black", lw=0.5)
            ax[0].plot(times[::100], chart_interp[::100], color="red")
            ax[1].set_title(f"De-lagged chart values (lag: {lag})")
            ax[1].plot(times[::100], filtered_wave[::100], color="black", lw=0.5)
            ax[1].plot(times[::100] + lag, chart_interp[::100], color="red")
            ax[2].set_title(f"De-lagged clipped chart values (lag: {clip_lag})")
            ax[2].plot(times[::100], y3[::100], color="black", lw=0.5)
            ax[2].plot(times[::100] + clip_lag, chart_interp[::100], color="red")
            ax[2].set_xlabel("hrs")
            # ax[1].plot(times[::100], y3[::100], color="black", lw=0.5)
            # ax[1].plot(times[::100], chart_vals[::100], color="red")
            # ax[1].set_xlabel("Clipped smoothed wave")
            plt.show()
