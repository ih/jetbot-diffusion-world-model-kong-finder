#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ipyevents')


# In[2]:


import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset # Keep Subset
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
import time
import datetime
import csv
import random
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import threading
import io
from ipyevents import Event
# --- Assuming these are accessible ---
# Make sure config.py is in the path or values are defined
import config

from importnb import Notebook

with Notebook():
    # Make sure JetbotDataset class definition is available
    from jetbot_dataset import JetbotDataset, load_train_test_split # Import load_train_test_split if needed


# In[3]:


# --- Configuration ---
# Use paths from config for the combined dataset
AGGREGATE_CSV_PATH = config.CSV_PATH
AGGREGATE_DATA_DIR = config.DATA_DIR
DISPLAY_IMAGE_SIZE = config.IMAGE_SIZE # For display consistency
NUM_PREV_FRAMES = config.NUM_PREV_FRAMES # Keep this if get_data needs it, even if not displaying prev frames
OUTPUT_REWARD_CSV = config.MANUAL_COLLECTED_REWARD_CSV # Changed output filename slightly

# --- Decide which dataset to use ---
USE_SUBSET = "train" # Options: None, "train", "test"
DATASET_SPLIT_FILENAME = config.SPLIT_DATASET_FILENAME # Ensure this is defined in config

# --- Load Full Dataset (Needed for accessing original dataframe regardless of subset) ---
# We won't apply the normalization transform here, just load PIL images
full_dataset_for_metadata = JetbotDataset(
    csv_path=AGGREGATE_CSV_PATH,
    data_dir=AGGREGATE_DATA_DIR,
    image_size=DISPLAY_IMAGE_SIZE,
    num_prev_frames=NUM_PREV_FRAMES,
    transform=None # Load PIL images directly for display
)

# --- Load the Dataset for Labeling (Full or Subset) ---
if USE_SUBSET:
    # Load the split
    train_dataset_subset, test_dataset_subset = load_train_test_split(full_dataset_for_metadata, DATASET_SPLIT_FILENAME)
    if train_dataset_subset is None or test_dataset_subset is None:
        raise FileNotFoundError(f"Dataset split file '{DATASET_SPLIT_FILENAME}' not found or invalid. Cannot use subset.")

    if USE_SUBSET == "train":
        labeling_dataset = train_dataset_subset
        print(f"Using Training Subset ({len(labeling_dataset)} sequences)")
    elif USE_SUBSET == "test":
        labeling_dataset = test_dataset_subset
        print(f"Using Test Subset ({len(labeling_dataset)} sequences)")
    else:
        raise ValueError("Invalid USE_SUBSET value. Choose None, 'train', or 'test'.")
else:
    labeling_dataset = full_dataset_for_metadata
    print(f"Using Full Dataset ({len(labeling_dataset)} sequences)")


# In[4]:


# --- Data Structures for Labeling ---
# Use a dictionary to store rewards {original_dataframe_index: reward}
reward_labels = {}
current_labeling_index = 0 # Index within the len(labeling_dataset)
is_auto_advancing = False # State for auto-advance
auto_advance_timer = None # To hold the Timer object
current_reward_value = 0.0


# In[5]:


# --- UI Widgets ---
# Index slider reflects the current position within the *labeling_dataset*
index_slider = widgets.IntSlider(
    value=current_labeling_index, min=0, max=len(labeling_dataset) - 1 if len(labeling_dataset) > 0 else 0, step=1,
    description='Sequence Index:', continuous_update=False, layout=widgets.Layout(width='80%')
)

reward_display = widgets.FloatText(
    value=current_reward_value, description='Current Reward:', disabled=True,
    layout=widgets.Layout(width='200px')
)

save_button = widgets.Button(description="Save All Rewards", button_style='success')

# Auto-Advance Widgets
speed_slider = widgets.FloatSlider(
    value=0.1, 
    min=0.01, 
    max=0.2,
    step=0.01, 
    description='Delay (s):',
    continuous_update=False, orientation='horizontal', readout=True, readout_format='.2f',
    layout=widgets.Layout(width='50%')
)
start_stop_button = widgets.Button(description="Start Auto", button_style='info', icon='play')

# Single Image Widget for Display
image_widget = widgets.Image(
    format='jpeg',
    width=DISPLAY_IMAGE_SIZE + 50,
    height=DISPLAY_IMAGE_SIZE + 50,
)

# Output areas
info_output = widgets.Output()
status_output = widgets.Output()

keyboard_info = widgets.HTML(value="""
<b>Keyboard Controls:</b><br>
<b>0-9:</b> Set reward (0.0-0.9)<br>
<b>M:</b> Set reward to 1.0<br>
<b>+/- or =/_:</b> Adjust reward by 0.01<br>
<i>Ensure Cell Output has focus for keys to register</i>
""")


# In[6]:


# --- Helper Function to Get Original DataFrame Index ---
def get_original_dataframe_index(current_dataset, current_index_in_dataset):
    """
    Traces back through Subset objects to find the index in the original full dataset's dataframe.
    """
    temp_dataset = current_dataset
    actual_index = current_index_in_dataset

    # Traverse up the dataset hierarchy if it's nested Subsets
    while isinstance(temp_dataset, Subset):
        if actual_index >= len(temp_dataset.indices):
             # This should ideally not happen if index_slider max is set correctly
             print(f"Error: Index {actual_index} out of bounds for subset indices (len {len(temp_dataset.indices)}).")
             return None
        actual_index = temp_dataset.indices[actual_index] # Map to index in parent dataset
        temp_dataset = temp_dataset.dataset       # Move to the parent dataset

    # Now temp_dataset should be the original JetbotDataset instance
    # and actual_index is the index *within that original dataset's valid_indices list*
    if not hasattr(temp_dataset, 'valid_indices') or not hasattr(temp_dataset, 'dataframe'):
        print("Error: Could not trace back to the original JetbotDataset with valid_indices.")
        return None

    if actual_index >= len(temp_dataset.valid_indices):
         print(f"Error: Mapped index {actual_index} out of bounds for original dataset's valid_indices (len {len(temp_dataset.valid_indices)}).")
         return None

    original_df_index = temp_dataset.valid_indices[actual_index]
    return original_df_index


# In[7]:


# --- Callback Functions ---
def get_data_for_labeling_index(dataset_idx):
    """ Safely gets data using the labeling_dataset's __getitem__ """
    # ...(implementation remains the same)...
    if dataset_idx >= len(labeling_dataset):
        print(f"Error: Index {dataset_idx} out of bounds for the current labeling dataset (len {len(labeling_dataset)}).")
        return None, None
    try:
        current_img_pil, action_tensor, _ = labeling_dataset[dataset_idx]
        if isinstance(current_img_pil, torch.Tensor):
             current_img_pil = transforms.ToPILImage()(current_img_pil.cpu())
        return current_img_pil.convert("RGB"), action_tensor.item() # Ensure RGB
    except IndexError:
        print(f"Error: Index {dataset_idx} out of bounds during __getitem__ call.")
        return None, None
    except Exception as e:
        print(f"Error getting data for labeling index {dataset_idx}: {e}")
        return None, None


def pil_to_widget_bytes(pil_image):
    """ Converts PIL Image to bytes suitable for ipywidgets.Image """
    # ...(implementation remains the same)...
    if pil_image is None: return None
    with io.BytesIO() as output_bytes:
        pil_image.save(output_bytes, format="JPEG")
        return output_bytes.getvalue()

def save_current_reward():
    """Saves the reward from the state variable for the currently displayed index."""
    # --- MODIFIED: Reads from variable, not slider ---
    global current_reward_value # Use the state variable
    current_idx_in_labeling_dataset = index_slider.value
    original_df_idx = get_original_dataframe_index(labeling_dataset, current_idx_in_labeling_dataset)

    if original_df_idx is not None:
        # Ensure reward is clamped between 0 and 1 before saving
        reward_to_save = np.clip(current_reward_value, 0.0, 1.0)
        reward_labels[original_df_idx] = reward_to_save
        # Optional: print(f"Debug: Stored reward {reward_to_save:.2f} for Original DF Index: {original_df_idx}")
    else:
         print(f"Warning: Could not determine original dataframe index for labeling index {current_idx_in_labeling_dataset}. Reward not saved.")

def update_display(labeling_idx):
    """Loads and displays frame info ONLY. Does NOT change current reward state."""
    # No longer need global reward_display here as we aren't setting it

    # --- Input validation ---
    if labeling_idx < 0 or labeling_idx >= len(labeling_dataset):
        with info_output: clear_output(wait=True); print(f"Invalid index: {labeling_idx}")
        image_widget.value = b''; return # Clear image and exit

    # --- Load data ---
    current_img_pil, action = get_data_for_labeling_index(labeling_idx)
    original_df_idx = get_original_dataframe_index(labeling_dataset, labeling_idx)

    # --- Clear status ---
    with status_output: clear_output(wait=True)

    # --- Check data validity ---
    if current_img_pil is None or original_df_idx is None:
        with info_output: clear_output(wait=True); print(f"Could not load data/index for labeling index: {labeling_idx}.")
        image_widget.value = b''; return # Clear image and exit

    # --- Determine reward status text ONLY ---
    # Check if a reward exists in memory for this frame, just for display purposes.
    # DO NOT use this to update the current_reward_value or reward_display widget.
    reward_status_text = "No previous reward saved."
    if original_df_idx in reward_labels:
        saved_val = reward_labels[original_df_idx]
        reward_status_text = f"Previously saved: {saved_val:.2f}" # Note the different text

    # --- Update Info Output ---
    # Display frame info and the reward status text.
    # The reward_display widget will continue showing the last value set by the keyboard.
    with info_output:
        clear_output(wait=True) # Clear previous text
        print(f"Labeling Index: {labeling_idx}/{len(labeling_dataset)-1} (Original DF Index: {original_df_idx})")
        print(f"Action leading to this frame: {action:.4f}")
        print(reward_status_text) # Show if a value WAS saved previously

    # --- Display Image (code remains the same) ---
    try:
        if hasattr(current_img_pil, 'resize'):
             display_image = current_img_pil.resize(
                  (DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE),
                  Image.Resampling.LANCZOS # Use a high-quality resampling filter
             )
             image_widget.value = pil_to_widget_bytes(display_image)
        else:
             print("Error: Invalid image object received during display.")
             image_widget.value = b'' # Clear image on error
    except Exception as e:
        with status_output:
             clear_output(wait=True)
             print(f"Error during image processing/display: {e}")
        image_widget.value = b'' # Clear image on error



def stop_auto_advance(change=None):
    """Stops the auto-advance timer and resets UI."""
    # ...(implementation remains the same, no slider refs needed)...
    global is_auto_advancing, auto_advance_timer
    if auto_advance_timer is not None:
        auto_advance_timer.cancel(); auto_advance_timer = None
    was_advancing = is_auto_advancing
    is_auto_advancing = False
    start_stop_button.description = "Start Auto"
    start_stop_button.button_style = 'info'; start_stop_button.icon = 'play'
    index_slider.disabled = False
    if was_advancing:
        with status_output: clear_output(wait=True); print("Auto-advance stopped.")


def auto_advance_step():
    """Performs one step of auto-advance, SAVING reward first."""
    global is_auto_advancing, auto_advance_timer, current_reward_value

    # print(f"DEBUG: auto_advance_step entered. is_auto_advancing={is_auto_advancing}") # Keep for debugging if needed

    # Essential check: If the flag was turned off between scheduling and execution
    if not is_auto_advancing:
        # print("DEBUG: auto_advance_step entered but is_auto_advancing is False. Stopping.") # Keep for debugging if needed
        return

    try: # Wrap core logic in try-except
        # --- ENSURE SAVE CALL IS HERE ---
        # Save reward for the frame that WAS just displayed, based on the current state variable
        # print(f"DEBUG: auto_advance_step saving reward: {current_reward_value:.2f}") # Keep for debugging if needed
        save_current_reward()
        # --- END SAVE CALL ---

        # Move to next index
        current_idx = index_slider.value
        next_idx = current_idx + 1
        # print(f"DEBUG: Advancing index from {current_idx} to {next_idx}") # Keep for debugging if needed

        if next_idx < len(labeling_dataset):
            index_slider.value = next_idx # This triggers on_index_change -> update_display
            # Schedule the next step only if still advancing
            if is_auto_advancing:
                delay = speed_slider.value
                # print(f"DEBUG: Scheduling next step with delay: {delay}s") # Keep for debugging if needed
                auto_advance_timer = threading.Timer(delay, auto_advance_step)
                auto_advance_timer.start()
            # else:
                # print("DEBUG: is_auto_advancing became False before scheduling next step.") # Keep for debugging if needed

        else:
            with status_output: clear_output(wait=True); print("End of dataset.")
            stop_auto_advance()

    except Exception as e:
        print(f"DEBUG: ERROR in auto_advance_step execution: {e}")
        # Attempt to stop gracefully on error
        stop_auto_advance()
        stop_auto_advance()

def toggle_auto_advance(b):
    """Starts or stops the auto-advance feature."""
    # ...(implementation remains the same)...
    print("HEELLLOO")
    global is_auto_advancing, auto_advance_timer
    if is_auto_advancing:
        stop_auto_advance()
    else:
        is_auto_advancing = True
        start_stop_button.description = "Stop Auto"
        start_stop_button.button_style = 'warning'; start_stop_button.icon = 'stop'
        index_slider.disabled = True
        with status_output: clear_output(wait=True); print(f"Auto-advance started...")
        # --- Start the timer loop (saves happen within loop) ---
        delay = speed_slider.value
        auto_advance_timer = threading.Timer(delay, auto_advance_step)
        auto_advance_timer.start()


def on_index_change(change):
    """Called when the index slider value changes."""
    # ...(implementation remains the same, just updates display)...
    if change['type'] == 'change' and change['name'] == 'value':
        # Stop auto advance if user manually changes slider
        #if is_auto_advancing: stop_auto_advance()
        update_display(change['new'])

# --- REMOVED: on_reward_change (slider callback) ---

def on_save_button_clicked(b):
    """Saves the collected reward labels to a CSV file."""
    # ...(implementation remains largely the same)...
    stop_auto_advance()
    with status_output:
        clear_output(wait=True)
        if not reward_labels: print("No rewards assigned yet."); return

        print(f"Preparing to save {len(reward_labels)} labels...")
        items_to_save = []
        original_dataframe = full_dataset_for_metadata.dataframe
        for df_idx in sorted(reward_labels.keys()):
             reward = reward_labels[df_idx]
             try:
                  if df_idx >= len(original_dataframe): continue # Skip invalid index
                  original_row = original_dataframe.iloc[df_idx]
                  items_to_save.append({
                       'dataframe_index': df_idx, 'session_id': original_row['session_id'],
                       'image_path': original_row['image_path'], 'action': original_row['action'],
                       'assigned_reward': reward })
             except Exception as e: print(f"Warn: Error processing DF index {df_idx}: {e}. Skipping.")

        if not items_to_save: print("No valid entries to save."); return
        save_df = pd.DataFrame(items_to_save)
        try:
            save_df.to_csv(OUTPUT_REWARD_CSV, index=False, float_format='%.8f')
            print(f"Saved {len(items_to_save)} labels to {OUTPUT_REWARD_CSV}")
        except Exception as e: print(f"Error saving CSV: {e}")

# --- NEW: Keyboard Event Handler ---
def handle_keydown(event):
    """Handles key presses to set the reward value."""
    global current_reward_value
    key = event.get('key', '')
    # print(f"DEBUG: Keydown event captured: key='{key}'") # Keep for debugging if needed

    reward_changed = False
    try:
        if key.isdigit() and 0 <= int(key) <= 9:
            current_reward_value = float(key) / 10.0
            reward_changed = True
        elif key == 'm' or key == 'M':
            current_reward_value = 1.0
            reward_changed = True
        elif key == '+' or key == '=': # Handle + and =
            current_reward_value += 0.1
            reward_changed = True
        elif key == '-' or key == '_': # Handle - and _
            current_reward_value -= 0.1
            reward_changed = True
    except Exception as e:
        print(f"DEBUG: Error processing key '{key}': {e}")

    if reward_changed:
        # Clamp reward value
        current_reward_value = np.clip(current_reward_value, 0.0, 1.0)
        # Update display ONLY
        reward_display.value = current_reward_value


# In[8]:


# --- Link Widgets ---
index_slider.observe(on_index_change, names='value')
save_button.on_click(on_save_button_clicked)
start_stop_button.on_click(toggle_auto_advance)


# In[9]:


##### --- Arrange Layout ---
auto_advance_controls = widgets.HBox([speed_slider, start_stop_button])
info_controls = widgets.VBox([info_output, reward_display]) # Group info/reward display

# Main UI layout
ui = widgets.VBox([
    widgets.HBox([info_controls, keyboard_info]), # Show controls side-by-side
    image_widget,
    index_slider,
    auto_advance_controls,
    save_button,
    status_output
])

# --- NEW: Setup and Link Keyboard Handler ---
# Create the Event listener, attached to the main UI container
# This makes it more likely to capture events when focus is within the output cell
key_handler = Event(source=ui, watched_events=['keydown'], prevent_default_action=True)
# Register the callback function
key_handler.on_dom_event(handle_keydown)


# --- Initial Display ---
if len(labeling_dataset) > 0:
    update_display(current_labeling_index)
    display(ui) # Display the main UI container
else:
     print("Cannot display UI because the labeling dataset is empty.")

