import tensorflow as tf
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray

# %matplotlib inline

import math
import glob
import io
import base64

import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor

import gym
from gym import logger as gymlogger
gymlogger.set_level(40) #error only

''' ipynb only
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
'''

def preprocess(image):
    # resizes image 210*160*3(color) to 84*84*1(mono)
    # also convert from float to int to reduce RAM usage
    return np.uint8(resize(rgb2gray(image), (84, 84), mode='constant', anti_aliasing=True, anti_aliasing_sigma=None) * 255)

def show_video(training = False):
        # helper for displaying videos of episodes on jupyter
        path = 'training' if training else 'test'

        mp4list = glob.glob(path + '/*.mp4')

        if len(mp4list) > 0:
            for mp4 in mp4list:
              video = io.open(mp4, 'r+b').read()
              encoded = base64.b64encode(video)
              ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                          loop controls style="height: 400px;">
                          <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                          </video>'''.format(encoded.decode('ascii'))))
        else:
          print("Could not find " + path + " videos")
