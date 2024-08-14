from kivy.app import App
from kivy.uix.label import Label
from kivy.core.audio import SoundLoader
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.clock import Clock
from muselsl import stream
from multiprocessing import Process
from time import sleep


def open_stream():
    stream(address=None)


class MyApp(App):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.sound = SoundLoader.load('assets/sounds/vine-boom.mp3')
        self.p = None
        self.clabel = None
        self.counter = 0

    def build(self):
        parent = Widget()
        sound_btn = Button(text="Play sound", pos=(350, 300))
        sound_btn.bind(on_press=self.play_sound)
        # TODO: Update the label with the EEG data
        self.clabel = Label(text='Hello world', pos=(350, 500))
        parent.add_widget(self.clabel)
        parent.add_widget(sound_btn)
        Clock.schedule_interval(lambda dt: self.increment_label(), 1)
        self.p = Process(target=open_stream)
        self.p.start()
        return parent

    def increment_label(self):
        self.counter += 1
        self.clabel.text = str(self.counter)

    def play_sound(self, obj):
        self.sound.play()


if __name__ == '__main__':
    stream_app = MyApp().run()

