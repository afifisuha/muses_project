from matplotlib import pyplot as plt
from matplotlib import animation
from playsound import playsound
import asyncio
import numpy as np
from scipy.signal import butter, lfilter

from muses_project.controller import ProjectController, MAX_MEMORY


def butter_lowpass(cutoff, fs, order=5) -> (float, float):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data)


def graph_eeg(controller):
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

    def update(_):
        for j in range(4):
            lines[j].set_data(t[:controller.eeg_arr.shape[0]], butter_lowpass_filter(controller.eeg_arr[:, j],
                                                                                     16, 256, 5))
        for j in range(4, 7):
            lines[j].set_data(t[:controller.acc_arr.shape[0]], controller.acc_arr[:, j - 4])
        for j in range(7, 10):
            lines[j].set_data(t[:controller.gyro_arr.shape[0]],controller.gyro_arr[:, j - 7])

        controller.blink_detection()
        controller.movement_detection()
        # Wait for more data.
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.sleep(0.1))

        return lines

    ani = animation.FuncAnimation(fig, update, cache_frame_data=False, interval=100, blit=True)
    plt.show()
