import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
from scipy import signal
import csv
import os

start = 10
stop = 12
step = 2

dt = 0.0001 # 0.1 ms sampling interval
fs = 1 / dt      # 10,000 Hz

def calculate_lfp_psd(data, fs, window_duration=0.25, overlap_pct=0.5):
    """
    Calculates the Power Spectral Density using Welch's method.
    
    Parameters:
    data (np.array): The LFP signal (time series).
    fs (int): Sampling frequency in Hz.
    """

    nperseg = 5000
    noverlap = nperseg // 2  # 50% overlap
    # Compute PSD using Welch's method with a Hanning window
    frequencies, psd = signal.welch(data, 
                                    fs=fs, 
                                    window='hann', 
                                    nperseg=nperseg, 
                                    noverlap=noverlap)
    
    return frequencies, psd

def process_lfp(data, fs):
    # 1. 4th Order Butterworth Bandpass (2 - 500 Hz)
    sos_band = butter(4, [1, 500], btype='band', fs=fs, output='sos')
    bandpassed = sosfiltfilt(sos_band, data)
    
    # 2. Notch Filter for Power Line Noise (50 Hz)
    for i in range(50,500,50): # apply notch filters at 50 Hz and its harmonics up to 500 Hz
        b_notch, a_notch = iirnotch(i, 30, fs)
        sos_notch = tf2sos(b_notch, a_notch)
        final_signal = sosfiltfilt(sos_notch, bandpassed)
    # # Get ba coefficients first
    # b_notch, a_notch = iirnotch(50, 30, fs)
    # # Convert ba to sos
    # sos_notch = tf2sos(b_notch, a_notch)
    
    # # Apply the notch filter
    # final_signal = sosfiltfilt(sos_notch, bandpassed)
    
    return final_signal

#####################################################################################################
# main code
#####################################################################################################

path = "/home/ulrike/OSS-DBSv2/input_test_cases/input_case10/1_bipolar_lfp_dti_noencap/"

bipolar_lfp_rms = np.zeros(int((stop - start) / step))
bipolar_lfp_rms_DTI = np.zeros(int((stop - start) / step))

radius = np.arange(start, stop, step) /10 # convert to mm
for i in np.arange(start, stop, step):
    data1 = np.loadtxt(path + f"c1_lfp_at_contact_in_time_{i/10}.csv", skiprows=1, delimiter=",")
    data1 = np.float32(np.array(data1))[:, 1] # in uV
    #data1 = process_lfp(data1, fs) # apply filters to the first contact's LFP
    data2 = np.loadtxt(path + f"c2_lfp_at_contact_in_time_{i/10}.csv", skiprows=1, delimiter=",")
    data2 = np.float32(np.array(data2))[:, 1] # in uV
    #data2 = process_lfp(data2, fs) # apply filters to the second contact's LFP
    data = (data1 - data2) # subtract to get bipolar LFP
    data = process_lfp(data, fs) # apply filters to the bipolar LFP
    rms = np.sqrt(np.mean(data**2))
    bipolar_lfp_rms[int((i - start) / step)] = rms
    downsampled_data = signal.resample(data, 10000)  # Downsample 
    # Calculate PSD
    print(f"fs = {fs}, data length = {len(data)}, duration = {len(data)/fs} seconds")
    freqs, psd_values = calculate_lfp_psd(downsampled_data, fs)

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

    # Plot PSD
    plt.figure(1)
    plt.plot(freqs, psd_values) # Use log scale for better visualization of LFP
    plt.title("LFP Power Spectral Density (Welch's Method)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (V^2/Hz)")
    plt.xlim(0, 500)  # Focus on physiological frequencies
    plt.grid(True)
    plt.savefig(path + "1_PSD_bipolar_lfp_dti_noencap.pdf")

plt.figure(2)
plt.plot(radius, bipolar_lfp_rms, marker="o", label="DTI")
plt.legend()
plt.xlabel("Radius (mm)")
plt.ylabel("RMS LFP ($\mu$V)")
plt.title("RMS bipolar LFP vs. Radius, without Encap Layer, 1 Neuron")
plt.grid()
plt.savefig(path + "1_rms_bipolar_lfp_dti_noencap.pdf")
plt.show()