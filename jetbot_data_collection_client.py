#!/usr/bin/env python
# coding: utf-8

# In[65]:


import rpyc
import logging
import time
import cv2
import numpy as np
import base64
from IPython.display import display, Image  # No need for clear_output here
import ipywidgets as widgets
import os
import csv
import datetime
import torchvision.transforms as transforms
from PIL import Image
import random
import config
from jetbot_remote_client import RemoteJetBot, generate_random_actions, record_data


# --- Setup Logging ---
logging.basicConfig(level=logging.WARNING)
jet_logger = logging.getLogger('JetBotClient')

jet_logger.setLevel(logging.WARNING)   # or logging.ERROR
for h in jet_logger.handlers:
    h.setLevel(logging.WARNING)
jet_logger.propagate = False     


def move_to_new_location(jetbot, forward_time=1.0, turn_time=1.0, speed=0.15):
    """Move the robot to a new location between recording sessions.
    This simple routine drives forward and then turns.
    """
    jetbot.set_motors(speed, speed)
    time.sleep(forward_time)
    jetbot.set_motors(speed, -speed)
    time.sleep(turn_time)
    jetbot.set_motors(0, 0)
    time.sleep(0.5)


# In[66]:


# --- Configuration ---
JETBOT_IP = '192.168.68.51'  # Replace with your Jetbot's IP address
IMAGE_SIZE = 224  # Use 224x224 images, don't use constant from config file since there may be resizing, or rename this and put it there
TARGET_FPS = 30
POSSIBLE_SPEEDS = [0.0, 0.15]
MIN_DURATION = 1.0  # Seconds
MAX_DURATION = 2.0  # Seconds
NUM_ACTIONS = 50 #How many total actions to do
NUM_SESSIONS = 1  # Number of times to record


# In[67]:


jetbot = RemoteJetBot(JETBOT_IP)


# In[68]:


try:
    for session_idx in range(NUM_SESSIONS):
        session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_session_dir = os.path.join(config.SESSION_DATA_DIR, f"session_{session_timestamp}")
        print(f"Creating session directory: {current_session_dir}")
        random_actions = generate_random_actions(NUM_ACTIONS, POSSIBLE_SPEEDS, MIN_DURATION, MAX_DURATION)
        # print(random_actions)

        record_data(jetbot, random_actions, TARGET_FPS, current_session_dir)

        if session_idx < NUM_SESSIONS - 1:
            move_to_new_location(jetbot)
finally:
    jetbot.cleanup()  # Stop motors and close connection


# In[ ]:




