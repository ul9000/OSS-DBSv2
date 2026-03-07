import numpy as np
from scipy.fft import rfft, rfftfreq

from .utilities import adjust_cutoff_frequency

class NeuronCurrentSignal:
    """Class representing neuron current stimulation signals."""

    def __init__(
        self,
        frequency: float,
        cutoff_frequency: float,
        timestep: float,
        simulation_times: list[np.ndarray] = None,

    ):
        self.frequency = frequency
        self.cutoff_frequency = cutoff_frequency
        self.timestep = timestep
        self.simulation_times = simulation_times
        self.fft_freqs = None
        self.fourier_coefficients = None
        self.signal_length = None

    def get_neuron_current_fft_freqs(self, cutoff_frequency: float, timings) -> np.ndarray:
        """FFT spectrum for all neuron current time-domain signals.

        Parameters
        ----------
        cutoff_frequency: float
            Highest considered frequency.

        """
        # # we double the cutoff_frequency to actually sample until there
        # cutoff_frequency = adjust_cutoff_frequency(
        #     2.0 * cutoff_frequency, self.frequency
        # )
        dt = self.timestep
        # required length for frequency
        timesteps = dt * self.frequency
        if np.isclose(dt, 0.0):
            raise ValueError("Choose a timestep dt larger than zero.")
        if timings.shape[0] < timesteps:
            raise ValueError(f"Im.txt has fewer rows ({timings.shape[0]}) than the specified timesteps ({timesteps}).")
        #return fftfreq(len(timings), d=dt)
        return rfftfreq(len(timings), d=dt)
    
    def get_neuron_current_fft_signal(self, time_domain_signal):
        fft_signal = rfft(time_domain_signal)
        return fft_signal  