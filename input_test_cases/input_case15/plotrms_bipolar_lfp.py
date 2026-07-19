"""Bipolar-LFP RMS-vs-radius plot for the param_sweep_lfp.py output.

For each simulated radius, reads the two reciprocity traces
(c1_lfp_at_contact_in_time_<radius>.csv, c2_lfp_at_contact_in_time_<radius>.csv),
takes their difference to get the bipolar LFP, filters it, and plots:
  - PSD per radius (Results dir / PSD_<fileending>.pdf)
  - RMS bipolar LFP vs. radius (Results dir / RMS_LFP_<fileending>.pdf)
  - the bipolar LFP trace of the last radius processed (Results dir / LFP_<fileending>.pdf)

Can be run standalone (edit the defaults in the __main__ block below) or
imported and called from param_sweep_lfp.py with the sweep's own path/range.
"""

from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt


def calculate_lfp_psd(data, fs, window_duration=1.0, overlap_pct=0.5):
    """Power spectral density via Welch's method."""
    nperseg = min(len(data), max(256, int(fs * window_duration)))
    noverlap = int(nperseg * overlap_pct)
    frequencies, psd = signal.welch(
        data, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
    )
    return frequencies, psd


def process_lfp(data, fs):
    """Band-limit the LFP to 1-500 Hz (4th order Butterworth, zero-phase)."""
    nyquist = fs / 2.0
    high_cutoff = min(500, nyquist * 0.95)
    low_cutoff = 1.0  # Hz

    if high_cutoff <= 0:
        return data

    sos_lowpass = butter(4, high_cutoff, btype="lowpass", fs=fs, output="sos")
    final_signal = sosfiltfilt(sos_lowpass, data)

    sos_highpass = butter(4, low_cutoff, btype="highpass", fs=fs, output="sos")
    final_signal = sosfiltfilt(sos_highpass, final_signal)

    return final_signal


def downsample_trace(time, values, source_fs, target_fs):
    if target_fs == source_fs:
        return time, values, source_fs

    source_fs = int(source_fs)
    target_fs = int(target_fs)
    common_divisor = gcd(source_fs, target_fs)
    up = target_fs // common_divisor
    down = source_fs // common_divisor

    resampled_values = signal.resample_poly(values, up, down)
    duration = time[-1] - time[0]
    resampled_time = np.linspace(
        time[0], time[0] + duration, len(resampled_values), endpoint=True
    )

    return resampled_time.astype(np.float32), resampled_values.astype(np.float32), target_fs


def plot_rms_bipolar(results_path, start, stop, step, fs, target_fs=2500, label="DTI"):
    """Generate PSD, RMS-vs-radius, and example-trace plots for one Results dir.

    Parameters
    ----------
    results_path : Path
        Directory containing c1_lfp_at_contact_in_time_<radius>.csv /
        c2_lfp_at_contact_in_time_<radius>.csv, and where the plots are saved.
    start, stop, step : int
        Same sweep range used to generate the data (radius in mm = value / 10).
    fs : float
        Sampling frequency of the raw ossdbs output.
    target_fs : float
        Frequency the traces are downsampled to before filtering/PSD.
    label : str
        Legend label for the RMS-vs-radius curve.
    """
    results_path = Path(results_path)
    fileending = results_path.name.removeprefix("Results_")

    missing = [
        path
        for i in range(start, stop, step)
        for path in (
            results_path / f"c1_lfp_at_contact_in_time_{i / 10}.csv",
            results_path / f"c2_lfp_at_contact_in_time_{i / 10}.csv",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Sweep is incomplete, refusing to plot: {len(missing)} expected file(s) "
            f"missing from {results_path}, e.g. {missing[0].name}"
        )

    n_points = int((stop - start) / step)
    bipolar_lfp_rms = np.zeros(n_points)
    radius = np.arange(start, stop, step) / 10  # convert to mm

    for i in np.arange(start, stop, step):
        r = i / 10
        data0 = np.loadtxt(
            results_path / f"c1_lfp_at_contact_in_time_{r}.csv", skiprows=1, delimiter=","
        )
        time1 = np.float32(data0)[:, 0]  # in seconds
        data1 = np.float32(data0)[:, 1]  # in V
        time1, data1, current_fs = downsample_trace(time1, data1, fs, target_fs)

        data2_raw = np.loadtxt(
            results_path / f"c2_lfp_at_contact_in_time_{r}.csv", skiprows=1, delimiter=","
        )
        data2 = np.float32(data2_raw)[:, 1]  # in V
        _, data2, _ = downsample_trace(np.float32(data2_raw)[:, 0], data2, fs, target_fs)

        data = data1 - data2  # subtract to get bipolar LFP
        data = process_lfp(data, current_fs)
        rms = np.sqrt(np.mean(data**2))
        bipolar_lfp_rms[int((i - start) / step)] = rms

        freqs, psd_values = calculate_lfp_psd(data, current_fs)

        plt.figure(1)
        # convert from V^2/Hz to uV^2/Hz for better visualization
        plt.plot(freqs, psd_values * 1e12, label=f"Radius: {r} mm")
        plt.title("LFP PSD (Welch's Method)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(r"Power/Frequency ($\mu$V^2/Hz)")
        plt.xlim(0, 100)
        plt.legend()
        plt.grid(True)
        plt.savefig(results_path / f"PSD_{fileending}.pdf")

    plt.figure(2)
    plt.plot(radius, bipolar_lfp_rms * 1e6, marker="o", label=label)
    plt.legend()
    plt.xlabel("Radius (mm)")
    plt.ylabel(r"RMS LFP ($\mu$V)")
    plt.title("RMS bipolar LFP vs. Radius")
    plt.grid()
    plt.savefig(results_path / f"RMS_LFP_{fileending}.pdf")

    plt.figure(3)
    plt.plot(time1, data * 1e6)  # convert from V to uV for better visualization
    plt.xlabel("Time (s)")
    plt.ylabel(r"LFP ($\mu$V)")
    plt.title("Bipolar LFP")
    plt.grid()
    plt.savefig(results_path / f"LFP_{fileending}.pdf")


if __name__ == "__main__":
    dt = 0.0004  # sampling interval [s]
    plot_rms_bipolar(
        results_path=Path(__file__).resolve().parent
        / "Results_100_80_sigma20_Cell_with_AIS_test_Encap0_DTI",
        start=10,
        stop=11,
        step=1,
        fs=1 / dt,
        target_fs=2500,
    )
