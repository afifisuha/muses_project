from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
from keyboard import on_press
from matplotlib import pyplot as plt
from matplotlib import animation
from playsound import playsound

import numpy as np
import time

def compute_band_powers(eegdata, fs):
    """Extract the features (band powers) from the EEG.

    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata

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
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)

    feature_vector = np.array((meanDelta, meanTheta, meanAlpha, meanBeta))
    feature_vector = np.log10(feature_vector)
    return feature_vector

class MuseController:
    def __init__(self):
        self.buffer_length = 5
        self.epoch_length = 1
        self.overlap_length = 0.8
        self.shift_length = 0.2
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

    def save_buffer_to_file(self):
        with open("buffer_log.txt", 'w') as f:
            buffer = ""
            for i in range(self.eeg_buffer.shape[0]):
                buffer += str(self.eeg_buffer[i]) + "\n"
            f.write(buffer)

    def acquire_data(self):
        eeg_data, timestamp = self.inlet.pull_chunk(timeout=1, max_samples=int(self.shift_length * self.s_rate))
        eeg_data = np.array(eeg_data)[:,:4]
        cutoff = self.eeg_buffer.shape[0] - eeg_data.shape[0]
        self.eeg_buffer = np.concatenate((eeg_data, self.eeg_buffer[:cutoff]), axis=0)
        epoch = self.eeg_buffer[:self.epoch_length * self.s_rate]
        self.band_powers = np.concatenate((self.band_powers, compute_band_powers(epoch, self.s_rate)), axis=1)
        self.band_powers = self.band_powers[:, -1000:]

if __name__ == '__main__':
    controller = MuseController()
    done = False
    def press_event(event):
        global done
        if event.name == 'q':
            done = True
    on_press(press_event)
    if controller.connect_to_streams() == 0:
        # print("Running, press Q to stop")
        # while not done:
        #     controller.acquire_data()
        t = np.linspace(1, 1000, 1000)
        fig, axes = plt.subplots(nrows=2, ncols=2)
        delta_line, = axes[0,0].plot(t[0], controller.band_powers[0])
        axes[0,0].set_title("Delta")
        theta_line, = axes[0,1].plot(t[0], controller.band_powers[1], "tab:orange")
        axes[0,1].set_title("Theta")
        alpha_line, = axes[1,0].plot(t[0], controller.band_powers[2], "tab:green")
        axes[1,0].set_title("Alpha")
        beta_line, = axes[1,1].plot(t[0], controller.band_powers[3], "tab:red")
        axes[1,1].set_title("Beta")
        for ax in axes.flat:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xlim([0, 1000])
            ax.set_ylim([-3, 6])

        def update(frame):
            controller.acquire_data()
            if controller.cooldown_start == 0:
                if controller.band_powers[0].shape[0] > 10:
                    high_theta_count = 0
                    for i in range(10):
                        if controller.band_powers[1][-i] > 1.15:
                            high_theta_count += 1
                    if high_theta_count > 5:
                        playsound("assets\\sounds\\vine-boom.mp3")
                        controller.cooldown_start = time.time()
            elif time.time() - controller.cooldown_start >= 1:
                controller.cooldown_start = 0

            end_frame = controller.band_powers[0].shape[0]
            delta_line.set_xdata(t[:end_frame])
            delta_line.set_ydata(controller.band_powers[0])
            theta_line.set_xdata(t[:end_frame])
            theta_line.set_ydata(controller.band_powers[1])
            alpha_line.set_xdata(t[:end_frame])
            alpha_line.set_ydata(controller.band_powers[2])
            beta_line.set_xdata(t[:end_frame])
            beta_line.set_ydata(controller.band_powers[3])
            return delta_line, theta_line, alpha_line, beta_line
        ani = animation.FuncAnimation(fig, update, cache_frame_data=False, interval=100, blit=True)
        plt.show()