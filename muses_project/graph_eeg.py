from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
from matplotlib import pyplot as plt
from matplotlib import animation
from playsound import playsound

import numpy as np
import time
MAX_MEMORY = 100


def compute_band_powers(eegdata, fs, band:int=1):
    """Extract the features (band powers) from the EEG.

    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata
        band: If not None, return only a specific band's powers.

    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = winSampleLength // 2 + 1
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f < 4)
    delta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f <= 8))
    theta = np.mean(PSD[ind_theta, :], axis=0)
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    alpha = np.mean(PSD[ind_alpha, :], axis=0)
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    beta = np.mean(PSD[ind_beta, :], axis=0)

    feature_vector = np.array((delta, theta, alpha, beta))
    if band is not None:
        feature_vector = feature_vector[band]
    feature_vector = np.log10(feature_vector)
    return feature_vector

class MuseController:
    def __init__(self):
        self.buffer_length = 5
        self.epoch_length = 1
        self.overlap_length = 0.8
        self.shift_length = self.epoch_length - self.overlap_length
        self.band_powers = np.ones((4, 1))
        self.cooldown_start = 0
        self.inlet = None
        self.time_correction = None
        self.s_rate = None
        self.eeg_buffer = None
        self.done = False

    def connect_to_streams(self):
        print("Connecting to muse-s stream...")
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            print("No muses found!")
            return 1
        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        self.s_rate = int(self.inlet.info().nominal_srate())
        self.eeg_buffer = np.ones((self.s_rate * self.buffer_length, 4))
        return 0

    def acquire_data(self):
        eeg_data, timestamp = self.inlet.pull_chunk(timeout=1, max_samples=int(self.shift_length * self.s_rate))
        eeg_data = np.array(eeg_data)[:,:4]
        cutoff = self.eeg_buffer.shape[0] - eeg_data.shape[0]
        self.eeg_buffer = np.concatenate((eeg_data, self.eeg_buffer[:cutoff]), axis=0)
        epoch = self.eeg_buffer[:int(self.epoch_length * self.s_rate)]
        band_powers = compute_band_powers(epoch, self.s_rate)[:, np.newaxis]
        self.band_powers = np.concatenate((self.band_powers, band_powers), axis=1)
        self.band_powers = self.band_powers[:, -MAX_MEMORY:]

def graph_eeg():
    controller = MuseController()
    if controller.connect_to_streams() == 0:
        t = np.linspace(1, MAX_MEMORY, MAX_MEMORY)
        fig, axes = plt.subplots(nrows=2, ncols=2)
        lines = []
        channels = ["left ear", "left temple", "right temple", "right ear"]
        colors = ["blue", "orange", "green", "red"]
        for i, ax in enumerate(axes.flat):
            line, = ax.plot([], [], color=colors[i])  # Start with empty data
            ax.set_title(f"Theta - {channels[i]}")
            ax.set_xlim([0, MAX_MEMORY])
            ax.set_ylim([-3, 6])  # Use set_ylim instead of duplicate set_xlim
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            lines.append(line)


        def update(frame):
            controller.acquire_data()

            # Handle cooldown logic
            if controller.cooldown_start == 0:
                if controller.band_powers[0].shape[0] > 10:
                    high_theta_count = 0
                    for i in range(10):
                        if controller.band_powers[1][-i] > 1.15:
                            high_theta_count += 1
                    if high_theta_count > 5:
                        # playsound("assets\\sounds\\vine-boom.mp3")
                        controller.cooldown_start = time.time()
            elif time.time() - controller.cooldown_start >= 1:
                controller.cooldown_start = 0

            # Get the actual data length (up to MAX_MEMORY points)
            data_len = min(controller.band_powers.shape[1], MAX_MEMORY)

            # Use only the most recent data points
            data_start = max(0, controller.band_powers.shape[1] - MAX_MEMORY)
            data_end = controller.band_powers.shape[1]

            # Update each line with the data
            for i, line in enumerate(lines):
                lines[i].set_data(t[:data_len], controller.band_powers[i, data_start:data_end])

            return lines

        ani = animation.FuncAnimation(fig, update, cache_frame_data=False, interval=100, blit=True)
        plt.show()
