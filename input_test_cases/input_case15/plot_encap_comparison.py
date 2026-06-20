import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
from scipy import signal
from math import gcd
from pathlib import Path

start = 0
stop = 200
step = 50

dt = 0.0004 # 0.1 ms sampling interval
fs = 1 / dt      # 10,000 Hz
target_fs = 2500  # Sridhar2026 used 2500 Hz
results_path_string = "/home/ulrike/OSS-DBSv2/input_test_cases/input_case10/Results_EncapComparison/"
results_path = Path(results_path_string)
results_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)

def calculate_lfp_psd(data, fs, window_duration=1.0, overlap_pct=0.5):
    """
    Calculates the Power Spectral Density using Welch's method.
    
    Parameters:
    data (np.array): The LFP signal (time series).
    fs (int): Sampling frequency in Hz.
    """
    nperseg = min(len(data), max(256, int(fs * window_duration)))
    noverlap = int(nperseg * overlap_pct)
    # Compute PSD using Welch's method with a Hanning window
    frequencies, psd = signal.welch(data, 
                                    fs=fs, 
                                    window='hann', 
                                    nperseg=nperseg, 
                                    noverlap=noverlap)
    
    return frequencies, psd

def process_lfp(data, fs):
    nyquist = fs / 2.0
    high_cutoff = min(500, nyquist * 0.95)
    if high_cutoff <= 1:
        return data

    # 1. 4th Order Butterworth Bandpass
    sos_band = butter(4, [1, high_cutoff], btype='band', fs=fs, output='sos')
    bandpassed = sosfiltfilt(sos_band, data)
    
    # 2. Notch Filter for Power Line Noise (50 Hz)
    final_signal = bandpassed
    max_notch = int(min(500, nyquist * 0.95))
    for i in range(50, max_notch + 1, 50): # apply notch filters at 50 Hz and its harmonics
        b_notch, a_notch = iirnotch(i, 30, fs)
        sos_notch = tf2sos(b_notch, a_notch)
        final_signal = sosfiltfilt(sos_notch, final_signal)

    # # Get ba coefficients first
    # b_notch, a_notch = iirnotch(50, 30, fs)
    # # Convert ba to sos
    # sos_notch = tf2sos(b_notch, a_notch)
    # # Apply the notch filter
    # final_signal = sosfiltfilt(sos_notch, bandpassed)
    
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
    resampled_time = np.linspace(time[0], time[0] + duration, len(resampled_values), endpoint=True)

    return resampled_time.astype(np.float32), resampled_values.astype(np.float32), target_fs

#####################################################################################################
# main code
#####################################################################################################

bipolar_lfp_rms = np.zeros(int((stop - start) / step))
bipolar_lfp_rms_DTI = np.zeros(int((stop - start) / step))
models = []

rms_values = {
    "myGillies": [],   # myGillies
    "Cell_test_3D": [] # Cell_test_3D
}
encap_values = []
# color_map = {
#     "myGillies": "tab:blue",
#     "Cell_test_3D": "tab:orange"
# }
# marker_map = {
#     0: "s",   # square
#     50: "^",  # triangle
#     100: "o",  # circle/dot
#     150: "D"   # diamond
# }

for j, type in enumerate(rms_values.keys()):
    for i in np.arange(start, stop, step):
        print(f"Processing {type}")
        fileending = f"5000_350_{type}_sigma10_En{i}"
        path = f"/home/ulrike/OSS-DBSv2/input_test_cases/input_case10/Results_{fileending}/"
        models.append(fileending)
        data0 = np.loadtxt(path + f"c1_lfp_at_contact_in_time_1.2.csv", skiprows=1, delimiter=",")
        time1 = np.float32(np.array(data0))[:, 0] # in seconds
        data1 = np.float32(np.array(data0))[:, 1] # in V
        #data1 = process_lfp(data1, fs) # apply filters to the first contact's LFP
        time1, data1, current_fs = downsample_trace(time1, data1, fs, target_fs)

        data2_raw = np.loadtxt(path + f"c2_lfp_at_contact_in_time_1.2.csv", skiprows=1, delimiter=",")
        time2 = np.float32(np.array(data2_raw))[:, 0] # in seconds
        data2 = np.float32(np.array(data2_raw))[:, 1] # in V
        time2, data2, _ = downsample_trace(time2, data2, fs, target_fs)
        #data2 = process_lfp(data2, fs) # apply filters to the second contact's LFP
        data = (data1 - data2) # subtract to get bipolar LFP
        #data = process_lfp(data, current_fs) # apply filters to the bipolar LFP
        rms = np.sqrt(np.mean(data**2))
        rms_values[type].append(rms)
        if type == "myGillies":
            encap_values.append(f"{i}") # in um
        # Calculate PSD
        freqs, psd_values = calculate_lfp_psd(data, current_fs)

        # Plot PSD
        if type == "myGillies":
            plt.figure(1)
            plt.plot(
                freqs, 
                psd_values*1e12,  # convert from V^2/Hz to uV^2/Hz for better visualization
                label=(f"{i} $\mu$m"),
            )
            #plt.title("LFP PSD (Welch's Method)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(r"Power/Frequency ($\mu$V^2/Hz)")
            plt.xlim(0, 100) 
            plt.grid(True)
            plt.legend()
            plt.savefig(results_path_string + f"PSD_myGillies.pdf")
        if type == "Cell_test_3D":
            plt.figure(2)
            plt.plot(
                freqs, 
                psd_values*1e12,  # convert from V^2/Hz to uV^2/Hz for better visualization
                label=(f"{i} $\mu$m"),
            )
            #plt.title("LFP PSD (Welch's Method)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(r"Power/Frequency ($\mu$V^2/Hz)")
            plt.xlim(0, 100) 
            plt.grid(True)
            plt.legend()
            plt.savefig(results_path_string + f"PSD_Cell_test_3D.pdf")
# Plot RMS LFP vs. Model
x = np.arange(len(encap_values))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in rms_values.items():
    if attribute == "myGillies": 
        name = "Type 2"
    else:
        name = "Type 1"
    offset = width * multiplier
    rects = ax.bar(x + offset, np.array(measurement)*1e6, width, label=name)
    ax.bar_label(rects, padding=2, fmt="%.2f")
    multiplier += 1
ax.set_ylim(0, 14)
ax.set_ylabel(r"RMS LFP ($\mu$V)")
ax.set_xlabel(r"Encapsulation layer thickness ($\mu$m)")
#ax.set_title('Model Comparison')
ax.set_xticks(x + width, encap_values)
ax.legend(loc='upper left', ncols=2)
n_bars = len(rms_values)
ax.set_xticks(x + width * (n_bars - 1) / 2, encap_values)
plt.savefig(results_path_string + f"RMS LFP.pdf")
