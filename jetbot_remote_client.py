#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# --- Setup Logging ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('JetBotClient')

# --- Image Transformation ---
# Transformations *before* saving to disk (for consistency with training)
transform = config.TRANSFORM


class RemoteJetBot:
    def __init__(self, ip_address, port=18861):
        logger.info(f"Connecting to JetBot at {ip_address}:{port}")
        try:
            self.conn = rpyc.connect(
                ip_address,
                port,
                config={
                    'sync_request_timeout': 30,
                    'allow_all_attrs': True
                }
            )
            logger.info("Connected successfully!")
            # Initialize video window
            self.image_widget = widgets.Image(
                format='jpeg',
                width=400,
                height=300,
            )
            display(self.image_widget)
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            raise

    def get_frame(self):
        """Get a single frame from the camera and display it"""
        try:
            # Get frame from server
            jpg_as_text = self.conn.root.get_camera_frame()
            if jpg_as_text:
                # Decode base64 string directly to bytes
                jpg_bytes = base64.b64decode(jpg_as_text)
                # Update the image widget
                self.image_widget.value = jpg_bytes

                # Convert to NumPy array (for saving)
                npimg = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                return frame  # Return the frame as a NumPy array
            return None

        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    def set_motors(self, left_speed, right_speed):
        try:
            logger.debug(f"Sending motor command: left={left_speed}, right={right_speed}")
            result = self.conn.root.set_motors(float(left_speed), float(right_speed))
            logger.debug("Command sent successfully")
            return result
        except Exception as e:
            logger.error(f"Error sending motor command: {str(e)}")
            raise

    def cleanup(self):
        try:
            logger.debug("Cleaning up connection")
            if hasattr(self, 'conn'):
                self.set_motors(0, 0)  # Stop motors
                self.conn.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def generate_random_actions(num_actions, possible_speeds, min_duration, max_duration):
    actions = []
    for _ in range(num_actions):
        speed = random.choice(possible_speeds)
        duration = random.uniform(min_duration, max_duration)  # Use uniform for continuous range
        actions.append((speed, duration))
    return actions

def record_data(jetbot, actions, target_fps, session_dir):
    """
    Records data for a single session into a specific directory.

    Args:
        jetbot: The RemoteJetBot object.
        actions: A list of (action, duration) tuples for this session.
        target_fps: The desired frames per second.
        session_dir: The directory to save this session's data.
    """
    session_image_dir = os.path.join(session_dir, 'images')
    session_csv_path = os.path.join(session_dir, 'data.csv')

    # Create session directories if they don't exist
    os.makedirs(session_image_dir, exist_ok=True)

    print(f"Starting data recording for session: {session_dir}")
    with open(session_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'timestamp', 'action']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        print(f"CSV header written to {session_csv_path}")

        target_interval = 1.0 / target_fps
        image_count = 0 # Counter *within* the session

        for action, duration in actions:
            # print(f"  Starting action: {action} for duration: {duration:.2f}s")
            jetbot.set_motors(action, 0)
            start_time = time.time()

            while time.time() - start_time < duration:
                frame_start_time = time.perf_counter()

                frame = jetbot.get_frame()
                if frame is None:
                    print("  Warning: Received None frame. Skipping.")
                    time.sleep(0.01) # Avoid busy-waiting if camera disconnects
                    continue

                # --- Image Processing ---
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                # Keep original PIL image for saving, apply transforms later if needed for training
                # image_tensor = transform(image_pil) # Transform is mainly for training input

                # --- Saving ---
                timestamp = time.time()
                image_filename = f"image_{image_count:05d}.jpg"
                relative_image_path = os.path.join('images', image_filename) # Relative path within session
                absolute_image_path = os.path.join(session_dir, relative_image_path)

                image_pil.save(absolute_image_path) # Save the original PIL image

                writer.writerow({'image_path': relative_image_path, 'timestamp': timestamp, 'action': action})
                image_count += 1

                # --- Frame Rate Control ---
                frame_end_time = time.perf_counter()
                elapsed_time = frame_end_time - frame_start_time
                sleep_time = target_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"  Finished action: {action}")

    print(f"Session recording complete. Total images in session: {image_count}")



# In[5]:


if __name__ == "__main__":
    # --- Configuration ---
    JETBOT_IP = '192.168.68.60'  # Replace with your Jetbot's IP address
    IMAGE_SIZE = 224  # Use 224x224 images, don't use constant from config file since there may be resizing, or rename this and put it there
    TARGET_FPS = 30
    POSSIBLE_SPEEDS = [0.0, 0.1]
    MIN_DURATION = 2.0  # Seconds
    MAX_DURATION = 5.0  # Seconds
    NUM_ACTIONS = 20 #How many total actions to do


    jetbot = RemoteJetBot(JETBOT_IP)
    
    try:
        session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_session_dir = os.path.join(config.SESSION_DATA_DIR, f"session_{session_timestamp}")
        print(f"Creating session directory: {current_session_dir}")
        random_actions = generate_random_actions(NUM_ACTIONS, POSSIBLE_SPEEDS, MIN_DURATION, MAX_DURATION)
        print(random_actions)
    
        # Record data
        record_data(jetbot, random_actions, TARGET_FPS, current_session_dir)
    finally:
        jetbot.cleanup()  # Stop motors and close connection


# In[ ]:




