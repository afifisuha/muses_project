from matplotlib import pyplot as plt
from matplotlib import animation
from muselsl.stream import list_muses
from muselsl.muse import Muse
from functools import partial
from playsound import playsound
from keyboard import on_press
import asyncio
import numpy as np
from scipy.signal import butter, lfilter
from os import path, getcwd
import time
MAX_MEMORY = 2000
EEG_CHANNELS = 4
ACC_CHANNELS = GYRO_CHANNELS = 3

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
        self.time_correction = None
        self.eeg_sample_rate = 256
        self.eeg_buffer = None
        self.eeg_arr = np.zeros(shape=(1,EEG_CHANNELS))
        self.acc_arr = np.zeros(shape=(1,ACC_CHANNELS))
        self.gyro_arr = np.zeros(shape=(1,GYRO_CHANNELS))
        self.switching = False
        self.on = False
        self.muse = None
        self.done = False

    def connect_to_stream(self):
        muses = list_muses()
        address = muses[0]["address"]

        def push(data, _, metric_type):
            arr = getattr(self, metric_type + "_arr")
            # :4 because eeg has a redundant channel
            arr = np.concatenate((arr,  data.T[:, :4]))
            if arr.shape[0] > MAX_MEMORY:
                arr = arr[arr.shape[0] - MAX_MEMORY:]
            setattr(self, metric_type + "_arr", arr)

        push_eeg = partial(push, metric_type="eeg")
        push_acc = partial(push, metric_type="acc")
        push_gyro = partial(push, metric_type="gyro")

        self.muse = Muse(address=address, callback_eeg=push_eeg, callback_acc=push_acc,
                    callback_gyro=push_gyro)

        connected = self.muse.connect(retries=1)
        if connected:
            self.muse.start()

        self.eeg_buffer = np.ones((self.eeg_sample_rate * self.buffer_length, 4))
        return 0


def butter_lowpass(cutoff, fs, order=5) -> (float, float):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data)


def graph_eeg():
    controller = MuseController()
    if controller.connect_to_stream() == 0:
        t = np.linspace(1, MAX_MEMORY, MAX_MEMORY)
        fig, axes = plt.subplots(nrows=3, ncols=4)
        lines = []
        channels = ["Left Ear", "Left Temple", "Right Temple", "Right Ear", "Accelerometer X", "Accelerometer Y",
                    "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z"]
        colors = ["blue", "orange", "green", "red"]
        for i in range(10):
            ax = axes.flat[i]
            line, = ax.plot([], [], color=colors[i % 4])
            ax.set_title(channels[i])
            ax.set_xlim([0, MAX_MEMORY])
            if i < 4:
                ax.set_ylim([-1000, 1000])
            elif i < 7:
                ax.set_ylim([-5, 5])
            else:
                ax.set_ylim([-200, 200])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            lines.append(line)
        fig.tight_layout()

        def blink_detection():
            # Crude Strong Blink Detection
            right_ear_var = np.var(controller.eeg_arr[-350:][:,0])
            left_ear_var = np.var(controller.eeg_arr[-350:][:,3])
            if not controller.switching:
                if right_ear_var > 50000 and left_ear_var > 50000:
                    controller.switching = True
                    controller.on = ~controller.on
                    if controller.on:
                        playsound(r"assets\sounds\turn_on_beep.wav")
                    else:
                        playsound(r"assets\sounds\turn_off_beep.wav")
            elif right_ear_var < 30000 and left_ear_var < 30000:
                print(right_ear_var)
                controller.switching = False

        def update(_):
            for j in range(4):
                lines[j].set_data(t[:controller.eeg_arr.shape[0]], butter_lowpass_filter(controller.eeg_arr[:, j],
                                                                                         16, 256, 5))
            for j in range(4, 7):
                lines[j].set_data(t[:controller.acc_arr.shape[0]], controller.acc_arr[:, j - 4])
            for j in range(7, 10):
                lines[j].set_data(t[:controller.gyro_arr.shape[0]],controller.gyro_arr[:, j - 7])

            blink_detection()

            # Wait for more data.
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.sleep(0.1))

            return lines

        ani = animation.FuncAnimation(fig, update, cache_frame_data=False, interval=100, blit=True)
        plt.show()
        controller.muse.stop()
        controller.muse.disconnect()
