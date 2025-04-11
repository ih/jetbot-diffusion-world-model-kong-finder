#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip3 install rpyc')


# In[1]:


get_ipython().system('pip3 show rpyc')


# In[1]:


import rpyc
from rpyc.utils.server import ThreadedServer
from jetbot import Robot, Camera
import logging
import sys
import cv2
import numpy as np
import base64
import torch


# In[2]:


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('JetBotServer')

class JetBotService(rpyc.Service):
    def __init__(self):
        super().__init__()
        self.robot = None
        self.camera = None
        logger.info("JetBot service initialized")
    
    def on_connect(self, conn):
        logger.info("Client connected")
    
    def on_disconnect(self, conn):
        logger.info("Client disconnected")
        if self.camera is not None:
                self.camera.stop()
                self.camera = None
                logger.info("Camera stopped")

        if self.robot:
            # Safety stop
            try:
                self.robot.stop()
            except:
                pass
    
    def exposed_set_motors(self, left_speed, right_speed):
        try:
            logger.debug(f"Received motor command: left={left_speed}, right={right_speed}")
            
            if self.robot is None:
                logger.debug("Initializing robot")
                self.robot = Robot()
            
            # Convert to float and set motors
            self.robot.left_motor.value = float(left_speed)
            self.robot.right_motor.value = float(right_speed)
            
            logger.debug("Motors set successfully")
            return True  # Acknowledge success
            
        except Exception as e:
            logger.error(f"Error setting motors: {str(e)}")
            raise
            
    def exposed_get_camera_frame(self):
        try:
            if self.camera is None:
                self.camera = Camera.instance(width=224, height=224)
                logger.info("Camera initialized")
            
            # Get frame from camera
            frame = self.camera.value
            #return torch.tensor(frame)
            # Convert to jpg for efficient transfer
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Convert to base64 string for transfer
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            return jpg_as_text
            
        except Exception as e:
            logger.error(f"Error getting camera frame: {str(e)}")
            return None


# In[ ]:


logger.info("Starting JetBot server on port 18861...")
server = ThreadedServer(
    JetBotService(),
    port=18861,
    protocol_config={
        'allow_all_attrs': True,
        'sync_request_timeout': 30,
        'allow_pickle': True,
    }
)
server.start()


# In[2]:


# server.py (save this on your JetBot)
import rpyc
from rpyc.utils.server import ThreadedServer

class MyService(rpyc.Service):
    def on_connect(self, conn):
        print("Client connected!")

    def on_disconnect(self, conn):
        print("Client disconnected!")

    def exposed_hello(self, name):
        print(f"Hello, {name}!")
        return f"Greetings, {name} from the JetBot!"

server = ThreadedServer(MyService, port=18861)
server.start()


# In[ ]:




