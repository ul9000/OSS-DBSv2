import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt
from scipy import signal
from scipy.integrate import trapezoid
from math import gcd

start = 5
stop = 20
step = 5
fileending = "10000_350_sigma20_Cell_with_AIS_noEncap_noDTI_test"

dt = 0.0004  # 0.4 ms sampling interval
fs = 1 / dt  # 2500 Hz
target_fs = 2500  # Sridhar2026 used 2500 Hz


def calculate_lfp_psd(data, fs, window_duration=1.0, overlap_pct=0.5):
    """
    Calculates the Power Spectral Density using Welch's method.
    
    Parameters:
    data (np.array): The LFP signal (time series).
    fs (int): Sampling frequency in Hz.
    """
    nperseg = min(len(data), max(256, int(fs * window_duration)))
    print(f"Using nperseg = {nperseg} samples for PSD calculation.")
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

    if high_cutoff <= 0:
        return data

    # 1. 4th Order Butterworth low-pass filter
    sos_lowpass = butter(4, high_cutoff, btype='lowpass', fs=fs, output='sos')
    final_signal = sosfiltfilt(sos_lowpass, data)
    
    # 2. Notch Filter for Power Line Noise (50 Hz)
    max_notch = int(min(500, nyquist * 0.95))
    # for i in range(50, max_notch + 1, 50): # apply notch filters at 50 Hz and its harmonics
    #     b_notch, a_notch = iirnotch(i, 30, fs)
    #     sos_notch = tf2sos(b_notch, a_notch)
    #     final_signal = sosfiltfilt(sos_notch, final_signal)
    
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

path = f"/home/ulrike/OSS-DBSv2/input_test_cases/input_case10/Results_{fileending}/"

bipolar_lfp_rms = np.zeros(int((stop - start) / step))
bipolar_lfp_rms_DTI = np.zeros(int((stop - start) / step))

radius = np.arange(start, stop, step) /10 # convert to mm
for i in np.arange(start, stop, step):
    data0 = np.loadtxt(path + f"c1_lfp_at_contact_in_time_{i/10}.csv", skiprows=1, delimiter=",")
    time1 = np.float32(np.array(data0))[:, 0] # in seconds
    data1 = np.float32(np.array(data0))[:, 1] # in V
    #data1 = process_lfp(data1, fs) # apply filters to the first contact's LFP
    time1, data1, current_fs = downsample_trace(time1, data1, fs, target_fs)

    data2_raw = np.loadtxt(path + f"c2_lfp_at_contact_in_time_{i/10}.csv", skiprows=1, delimiter=",")
    time2 = np.float32(np.array(data2_raw))[:, 0] # in seconds
    data2 = np.float32(np.array(data2_raw))[:, 1] # in V
    time2, data2, _ = downsample_trace(time2, data2, fs, target_fs)
    #data2 = process_lfp(data2, fs) # apply filters to the second contact's LFP
    data = (data1 - data2) # subtract to get bipolar LFP
    print(f"fs = {current_fs}, data length = {len(data)}, duration = {len(data)/current_fs} seconds")
    data = process_lfp(data, current_fs) # apply filters to the bipolar LFP
    rms = np.sqrt(np.mean(data**2))
    bipolar_lfp_rms[int((i - start) / step)] = rms
    # Calculate PSD
    # print(f"fs = {fs}, data length = {len(data)}, duration = {len(data)/fs} seconds")
    freqs, psd_values = calculate_lfp_psd(data, current_fs)

    # # 1. Create the CSV file if it does not exist
    # file_name = "psd.csv"
    # file_exists = os.path.isfile(file_name)

    # # 1. Create the CSV file
    # with open(file_name, mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     # Write header
    #     writer.writerow(["frequency", "psd"])
    #     # Write rows (frequency, psd pairs)
    #     for f, p in zip(freqs, psd_values):
    #         writer.writerow([f, p])

    # total_power = trapezoid(psd_values, freqs)
    # print(total_power)
    # print(np.var(data))

    # Plot PSD
    plt.figure(1)
    plt.plot(freqs, psd_values*1e12, label=f"Radius: {i/10} mm") # convert from V^2/Hz to uV^2/Hz for better visualization
    plt.title("LFP PSD (Welch's Method)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Power/Frequency ($\mu$V^2/Hz)")
    plt.xlim(0, 100) 
    plt.legend()
    plt.grid(True)
    plt.savefig(path + f"PSD_{fileending}.pdf")

plt.figure(2)
plt.plot(radius, bipolar_lfp_rms*1e6, marker="o", label="DTI")
plt.legend()
plt.xlabel("Radius (mm)")
plt.ylabel(r"RMS LFP ($\mu$V)")
plt.title("RMS bipolar LFP vs. Radius")
plt.grid()
plt.savefig(path + f"RMS_LFP_{fileending}.pdf")
#plt.show()

plt.figure(3)
plt.plot(time1, data*1e6) # convert from V to uV for better visualization
#plt.legend()
plt.xlabel("Time (s)")
plt.ylabel(r"LFP ($\mu$V)")
plt.title(f"Bipolar LFP")
plt.grid()
plt.savefig(path + f"LFP_{fileending}.pdf")
#plt.show()
