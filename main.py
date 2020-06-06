import os
import threading
import time
import tkinter as tk
from io import BytesIO
from tkinter import filedialog
from kivy.clock import Clock, mainthread

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
import numpy as np
import requests
from PIL import Image
from keras.models import load_model

model = load_model('Models/Object_Model.h5')
model100 = load_model('Models/100_Trained_model.h5')
default_directory = os.path.abspath(os.curdir)


labels10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
labels100 = ['beaver', 'dolphin', 'otter', 'seal', ' whale', 'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
             'orchids', ' poppies', ' roses', ' sunflowers', ' tulips', ' bottles', ' bowls', ' cans', ' cups',
             ' plates', ' apples', ' mushrooms', ' oranges', ' pears', ' sweet peppers', ' clock', ' computer keyboard',
             ' lamp', ' telephone', ' television', ' bed', ' chair', ' couch', ' table',
             ' wardrobe', ' bee', ' beetle', ' butterfly', ' caterpillar', ' cockroach', 'bear', ' leopard', ' lion',
             ' tiger', ' wolf', ' bridge',
             ' castle', ' house', ' road', ' skyscraper', ' cloud', ' forest', ' mountain', ' plain', ' sea', ' camel',
             ' cattle', ' chimpanzee',
             ' elephant', ' kangaroo', ' fox', ' porcupine', ' possum', ' raccoon', ' skunk', ' crab', ' lobster',
             ' snail', ' spider', ' worm',
             ' baby', ' boy', ' girl', ' man', ' woman', ' crocodile', ' dinosaur', ' lizard', ' snake', ' turtle',
             ' hamster', ' mouse', ' rabbit',
             ' shrew', ' squirrel', ' maple', ' oak', ' palm', ' pine', ' willow', ' bicycle', ' bus', ' motorcycle',
             ' pickup truck', ' train',
             ' lawn-mower', ' rocket', ' streetcar', ' tank', ' tractor']


class MainWindow(Screen):
    def identify(self):
        tk.Tk().withdraw()
        input_path = filedialog.askopenfilename()
        input_image = Image.open(input_path).resize((32, 32), resample=Image.LANCZOS)
        image_array = np.array(input_image).astype('float32')
        image_array /= 255.0
        image_array = image_array.reshape(1, 32, 32, 3)
        answer = model.predict(image_array)
        self.ans.text = str(labels10[np.argmax(answer)]).title()
        # 100
        # answer100 = model100.predict(image_array)
        # self.ans.text = str(labels100[np.argmax(answer100)]).title()

    def identify_url(self, url_given):
        if url_given:
            input_path = requests.get(self.url_entry.text)
            input_image = Image.open(BytesIO(input_path.content)).resize((32, 32), resample=Image.LANCZOS)
            image_array = np.array(input_image).astype('float32')
            image_array /= 255.0
            image_array = image_array.reshape(1, 32, 32, 3)
            answer = model.predict(image_array)
            self.ans.text = str(labels10[np.argmax(answer)]).title()
            # answer100 = model100.predict(image_array)
            # self.ans.text = str(labels100[np.argmax(answer100)]).title()


class SplashScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.get_screen)

    def get_screen(self, a):
        threading.Thread(target=self.change_screen).start()

    @mainthread
    def change_screen(self):
        time.sleep(2)
        self.manager.current = "Main"


class WindowManager(ScreenManager):
    pass


class Gui(App):
    def build(self):
        return Builder.load_file("gui.kv")


if __name__ == '__main__':
    Gui().run()
