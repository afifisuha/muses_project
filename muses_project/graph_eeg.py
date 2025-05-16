from matplotlib import pyplot as plt
from matplotlib import animation
from muselsl.stream import list_muses
from muselsl.muse import Muse
from functools import partial
from playsound import playsound
import asyncio
import numpy as np
from scipy.signal import butter, lfilter
MAX_MEMORY = 2000
EEG_CHANNELS = 4
ACC_CHANNELS = GYRO_CHANNELS = 3


class MuseController:
    def __init__(self):
        self.eeg_arr = np.zeros(shape=(1,EEG_CHANNELS))
        self.acc_arr = np.zeros(shape=(1,ACC_CHANNELS))
        self.gyro_arr = np.zeros(shape=(1,GYRO_CHANNELS))
        self.switching = False
        self.rotating = False
        self.opening_or_closing = False
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
            right_ear_var = np.var(controller.eeg_arr[:][:,0])
            left_ear_var = np.var(controller.eeg_arr[:][:,3])
            right_ear_recent_var = np.var(controller.eeg_arr[-250:][:,0])
            left_ear_recent_var = np.var(controller.eeg_arr[-250:][:,3])
            if not controller.switching:
                if right_ear_recent_var / (right_ear_var + 1e-5) + left_ear_recent_var / (left_ear_var + 1e-5) > 5:
                    controller.switching = True
                    controller.on = ~controller.on
                    if controller.on:
                        playsound(r"assets\sounds\turn_on_beep.wav")
                        print("On")
                    else:
                        playsound(r"assets\sounds\turn_off_beep.wav")
                        print("Off")
            elif right_ear_recent_var / (right_ear_var + 1e-5) + left_ear_recent_var / (left_ear_var + 1e-5) < 1:
                controller.switching = False

        def movement_detection():
            if not controller.on:
                return
            # Left: Z gyro increases then decreases
            # Up: Y gyro decreases then increases
            z_motion = controller.gyro_arr[-100:,2] / 52
            y_motion = controller.gyro_arr[-100:,1] / 52
            # Enable if last motion was a while ago, rotate if new motion detected.
            if controller.rotating:
                controller.rotating = np.max(np.cumsum(z_motion)) > 5 or np.min(np.cumsum(z_motion)) < -5
            elif np.abs(np.sum(z_motion)) < 5:
                if np.max(np.cumsum(z_motion)) > 20:
                    print(f"Rotate left {np.max(np.cumsum(z_motion)) * 2} degrees")
                    controller.rotating = True
                elif np.min(np.cumsum(z_motion)) < -20:
                    print(f"Rotate right {-np.min(np.cumsum(z_motion)) * 2} degrees")
                    controller.rotating = True
            if controller.opening_or_closing:
                controller.opening_or_closing = np.max(np.cumsum(y_motion)) > 5 or np.min(np.cumsum(y_motion)) < -5
            elif np.abs(np.sum(y_motion)) < 5:
                if np.max(np.cumsum(y_motion)) > 9:
                    print(f"Close {np.max(np.cumsum(y_motion)) * 3} degrees")
                    controller.opening_or_closing = True
                if np.min(np.cumsum(y_motion)) < -9:
                    print(f"Open {-np.min(np.cumsum(y_motion)) * 3} degrees")
                    controller.opening_or_closing = True


        def update(_):
            for j in range(4):
                lines[j].set_data(t[:controller.eeg_arr.shape[0]], butter_lowpass_filter(controller.eeg_arr[:, j],
                                                                                         16, 256, 5))
            for j in range(4, 7):
                lines[j].set_data(t[:controller.acc_arr.shape[0]], controller.acc_arr[:, j - 4])
            for j in range(7, 10):
                lines[j].set_data(t[:controller.gyro_arr.shape[0]],controller.gyro_arr[:, j - 7])

            blink_detection()
            movement_detection()
            # Wait for more data.
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.sleep(0.1))

            return lines

        ani = animation.FuncAnimation(fig, update, cache_frame_data=False, interval=100, blit=True)
        plt.show()
        controller.muse.stop()
        controller.muse.disconnect()
