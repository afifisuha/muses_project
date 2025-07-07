from bleak import BleakClient
from muselsl.stream import list_muses
from muselsl.muse import Muse
from functools import partial
from playsound import playsound
from muses_project.protocol import HAND_MAC, DIRECT_UUID, commands
import numpy as np

MAX_MEMORY = 2000
EEG_CHANNELS = 4
ACC_CHANNELS = GYRO_CHANNELS = 3

class ProjectController:
    def __init__(self):
        self.eeg_arr = np.zeros(shape=(1,EEG_CHANNELS))
        self.acc_arr = np.zeros(shape=(1,ACC_CHANNELS))
        self.gyro_arr = np.zeros(shape=(1,GYRO_CHANNELS))
        self.switching = False
        self.rotating = False
        self.opening_or_closing = False
        self.on = False
        self.muse = None
        self.hand = None
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

    def blink_detection(self):
        # Crude Strong Blink Detection
        right_ear_var = np.var(self.eeg_arr[:][:,0])
        left_ear_var = np.var(self.eeg_arr[:][:,3])
        right_ear_recent_var = np.var(self.eeg_arr[-250:][:,0])
        left_ear_recent_var = np.var(self.eeg_arr[-250:][:,3])
        if not self.switching:
            if right_ear_recent_var / (right_ear_var + 1e-5) + left_ear_recent_var / (left_ear_var + 1e-5) > 5:
                self.switching = True
                self.on = ~self.on
                if self.on:
                    playsound(r"assets\sounds\turn_on_beep.wav")
                    print("On")
                else:
                    playsound(r"assets\sounds\turn_off_beep.wav")
                    print("Off")

        elif right_ear_recent_var / (right_ear_var + 1e-5) + left_ear_recent_var / (left_ear_var + 1e-5) < 1:
            self.switching = False

    def movement_detection(self):
        if not self.on:
            return
        # Left: Z gyro increases then decreases
        # Up: Y gyro decreases then increases
        z_motion = self.gyro_arr[-100:,2] / 52
        y_motion = self.gyro_arr[-100:,1] / 52
        # Enable if last motion was a while ago, rotate if new motion detected.
        if self.rotating:
            self.rotating = np.max(np.cumsum(z_motion)) > 5 or np.min(np.cumsum(z_motion)) < -5
        elif np.abs(np.sum(z_motion)) < 5:
            if np.max(np.cumsum(z_motion)) > 20:
                print(f"Rotate left {np.max(np.cumsum(z_motion)) * 2} degrees")
                self.rotating = True
            elif np.min(np.cumsum(z_motion)) < -20:
                print(f"Rotate right {-np.min(np.cumsum(z_motion)) * 2} degrees")
                self.rotating = True
        if self.opening_or_closing:
            self.opening_or_closing = np.max(np.cumsum(y_motion)) > 5 or np.min(np.cumsum(y_motion)) < -5
        elif np.abs(np.sum(y_motion)) < 5:
            if np.max(np.cumsum(y_motion)) > 9:
                print(f"Close {np.max(np.cumsum(y_motion)) * 3} degrees")
                self.opening_or_closing = True
            if np.min(np.cumsum(y_motion)) < -9:
                print(f"Open {-np.min(np.cumsum(y_motion)) * 3} degrees")
                self.opening_or_closing = True

    async def connect_to_hand(self):
        print("Connecting to hand...")
        self.hand = BleakClient(HAND_MAC)
        await self.hand.connect()
        print("Connected!")

    def disconnect(self):
        self.muse.stop()
        self.muse.disconnect()


    async def move_hand(self, movement:str):
        await self.hand.write_gatt_char(DIRECT_UUID, bytes(commands[movement]))