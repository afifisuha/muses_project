from muselsl import stream
from multiprocessing import Process
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data


def open_stream():
    stream(address=None)
# p = Process(target=open_stream)
# p.start()

def connect_to_streams(self, obj):
    print("Connecting to muse-s stream...")
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        print("No muses found!")


if __name__ == '__main__':
    print("Main code goes here")

