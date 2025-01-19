from kivy.app import App
from kivy.uix.label import Label
from kivy.core.audio import SoundLoader
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.clock import Clock
from muselsl import stream
from multiprocessing import Process
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data


def open_stream():
    stream(address=None)


class MyApp(App):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.sound = SoundLoader.load('assets/sounds/vine-boom.mp3')
        self.p = None
        self.clabel = None
        self.title_label = None
        self.counter = 0

    def build(self):
        parent = Widget()
        sound_btn = Button(text="Play sound", pos=(350, 300))
        search_btn = Button(text="Search for streams", pos=(400, 300))
        sound_btn.bind(on_press=self.play_sound)
        search_btn.bind(on_press=self.connect_to_streams)
        # TODO: Update the label with the EEG data
        self.clabel = Label(text='0', pos=(350, 500))
        self.title_label = Label(text='EEG Data', pos=(350, 400))
        parent.add_widget(self.clabel)
        parent.add_widget(self.title_label)
        parent.add_widget(sound_btn)
        parent.add_widget(search_btn)
        Clock.schedule_interval(lambda dt: self.increment_label(), 1)
        self.p = Process(target=open_stream)
        self.p.start()
        return parent

    def increment_label(self):
        self.counter += 1
        self.clabel.text = str(self.counter)

    def play_sound(self, obj):
        self.sound.play()

    def connect_to_streams(self):
        self.title_label.text = str('Looking for an EEG stream...')
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('Can\'t find EEG stream.')


if __name__ == '__main__':
    stream_app = MyApp().run()

