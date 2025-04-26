import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream
import time
from playsound import playsound
import matplotlib.animation as animation


def compute_theta_power(eegdata, fs):
    winSampleLength, nbCh = eegdata.shape
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = winSampleLength // 2 + 1
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    return np.log10(meanTheta)


class EEGController:
    def __init__(self):
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(streams[0])

        self.s_rate = int(self.inlet.info().nominal_srate())
        self.epoch_length = 1
        self.shift_length = 0.2
        self.buffer_length = 5

        self.eeg_buffer = np.zeros((int(self.s_rate * self.buffer_length), 4))
        self.band_powers = np.zeros((4, 1))
        self.cooldown_start = 0

    def acquire_data(self):
        eeg_data, timestamp = self.inlet.pull_chunk(timeout=1, max_samples=int(self.shift_length * self.s_rate))
        eeg_data = np.array(eeg_data)[:, :4]
        cutoff = self.eeg_buffer.shape[0] - eeg_data.shape[0]
        self.eeg_buffer = np.concatenate((eeg_data, self.eeg_buffer[:cutoff]), axis=0)
        epoch = self.eeg_buffer[:self.epoch_length * self.s_rate]

        theta_vector = compute_theta_power(epoch, self.s_rate)
        self.band_powers = np.concatenate((self.band_powers, theta_vector[:, np.newaxis]), axis=1)
        self.band_powers = self.band_powers[:, -1000:]


controller = EEGController()
controller.acquire_data()

t = np.linspace(1, 1000, 1000)
fig, axes = plt.subplots(nrows=2, ncols=2)
lines = []

channel_names = ["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"]
colors = ["blue", "orange", "green", "red"]

for i, ax in enumerate(axes.flat):
    line, = ax.plot(t[0], controller.band_powers[i], color=colors[i])
    ax.set_title(f"Theta - {channel_names[i]}")
    ax.set_xlim([0, 1000])
    ax.set_ylim([-3, 6])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    lines.append(line)


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

    end_frame = controller.band_powers.shape[1]
    for i, line in enumerate(lines):
        line.set_xdata(t[:end_frame])
        line.set_ydata(controller.band_powers[i])

    return lines


ani = animation.FuncAnimation(fig, update, interval=200, blit=True)
plt.show()
