#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm # Use notebook version for better display (used in predict_rewards)
from collections import deque # For observation history buffer
import glob # For finding checkpoints
from importnb import Notebook # For importing from other notebooks/scripts
import sys # For exiting gracefully
import cv2 # Needed for image decoding
import base64 # For decoding image from rpyc server
import io # For handling image bytes
import logging # For RemoteJetBot logging
import config
import models
from models import SimpleUNetV1, SimpleRewardEstimator
with Notebook():
    from action_conditioned_diffusion_world_model_gemini import linear_beta_schedule, cosine_beta_schedule, get_index_from_list
    from jetbot_remote_client import RemoteJetBot

logging.basicConfig(level=logging.INFO) # Set logging level (INFO, DEBUG, etc.)
logger = logging.getLogger('MPC_Client')


# In[10]:


# --- JetBot Server Connection ---
JETBOT_SERVER_IP = "192.168.68.64" # <<< --- REPLACE WITH YOUR JETBOT'S ACTUAL IP ADDRESS
# Port identified from jetbot_server.py (uses rpyc)
JETBOT_SERVER_PORT = 18861

# --- Device Setup ---
device = torch.device(config.DEVICE if hasattr(config, 'DEVICE') else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# --- Model Paths ---
# World model path will be determined automatically by searching for 'model_best_epoch_*.pth'.
# Reward model path remains the same.
REWARD_MODEL_PATH = os.path.join(config.OUTPUT_DIR, 'reward_estimator', 'reward_estimator_best.pth')

# --- MPC Parameters ---
HORIZON = 5             # Planning horizon (number of steps to look ahead) H
N_ACTIONS = 1           # Dimensionality of the action space (RIGHT MOTOR ONLY)

# --- Discrete Actions ---
DISCRETE_ACTIONS = [0.0, 0.1] # The two possible actions

# --- Real Robot Parameters ---
REAL_ROBOT_FRAME_DELAY = 1.0 / 30.0 # Based on 30 FPS assumption
ACTION_SCALE = 1.0 # Keep for potential future use, but actions are discrete now

# --- Image Preprocessing (Matches config.TRANSFORM) ---
IMAGE_CHANNELS = 3 # Assuming 3 channels based on config.TRANSFORM and models
preprocess = transforms.Compose([
    # Note: Image received from get_frame is BGR numpy array.
    # Convert to RGB, then PIL, then transform.
    transforms.ToPILImage(), # Convert numpy array (RGB) to PIL Image
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalization from config.TRANSFORM
])
# Inverse transform for visualization
denormalize = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
])

# --- Diffusion Parameters (from config.py or defaults) ---
NUM_TIMESTEPS = getattr(config, 'NUM_TIMESTEPS', 10) # Use config value, default 10 based on log
BETA_START = getattr(config, 'BETA_START', 1e-4) #
BETA_END = getattr(config, 'BETA_END', 0.02) #
SCHEDULE_TYPE = getattr(config, 'SCHEDULE_TYPE', 'linear') #
NUM_PREV_FRAMES = config.NUM_PREV_FRAMES # Number of previous frames model expects

# --- Visualization Buffer ---
VISUALIZATION_BUFFER_SIZE = 50 # Store last N frames for visualization on exit

# %% [markdown]
# ## Diffusion Schedule Setup

# %%
# Calculate betas using imported functions
if SCHEDULE_TYPE == 'linear':
    betas = linear_beta_schedule(NUM_TIMESTEPS, BETA_START, BETA_END)
elif SCHEDULE_TYPE == 'cosine':
    betas = cosine_beta_schedule(NUM_TIMESTEPS)
else:
    raise ValueError(f"Unknown beta schedule: {SCHEDULE_TYPE}")

# Pre-calculate diffusion constants
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
# Ensure tensors are on the correct device
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)



# In[11]:


# --- Load World Model (SimpleUNetV1) ---
# Simplified Automatic Checkpoint Loading: Assumes at least one 'best' model exists.
checkpoint_to_load = None
loaded_config_from_checkpoint = None
world_model = None

# 1. Find the latest 'best' checkpoint file
# Use config for checkpoint directory
best_checkpoints = glob.glob(os.path.join(config.CHECKPOINT_DIR, 'model_best_epoch_*.pth'))
# Sort by epoch number (descending) to get the latest best
best_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
checkpoint_to_load = best_checkpoints[0] # Assume the list is not empty and take the first one
print(f"Attempting to load latest best world model checkpoint: {checkpoint_to_load}")

# 2. Load checkpoint metadata and instantiate the correct model
try:
    # Load directly without checking os.path.exists, assuming glob found a valid file
    checkpoint = torch.load(checkpoint_to_load, map_location=device)

    # --- Get Architecture Info from Checkpoint ---
    model_arch_name = None
    num_prev_frames_loaded = config.NUM_PREV_FRAMES # Default to current config
    time_emb_dim_loaded = getattr(config, 'TIME_EMB_DIM', 32) # Default

    if 'config' in checkpoint and checkpoint['config'] is not None: # Check if config exists and is not None
         loaded_config_from_checkpoint = checkpoint['config']
         if 'model_architecture' in loaded_config_from_checkpoint:
             model_arch_name = loaded_config_from_checkpoint['model_architecture']
             print(f"Checkpoint indicates model architecture: {model_arch_name}")
         if 'num_prev_frames' in loaded_config_from_checkpoint:
             num_prev_frames_loaded = loaded_config_from_checkpoint['num_prev_frames']
         if 'time_emb_dim' in loaded_config_from_checkpoint: # Check if time_emb_dim is saved
             time_emb_dim_loaded = loaded_config_from_checkpoint['time_emb_dim']
    else:
         # Fallback if checkpoint lacks architecture info - use current config
         print("Warning: Checkpoint missing config info or config is None. Using current config values.")
         model_arch_name = config.MODEL_ARCHITECTURE # Get from current config

    # --- Instantiate the model ---
    if model_arch_name and hasattr(models, model_arch_name):
         model_class = getattr(models, model_arch_name)

         # Ensure NUM_PREV_FRAMES matches between loaded model and current config expectation
         if num_prev_frames_loaded != config.NUM_PREV_FRAMES:
             print(f"Warning: Mismatch in NUM_PREV_FRAMES between loaded model ({num_prev_frames_loaded}) and current config ({config.NUM_PREV_FRAMES}). Using value from loaded model: {num_prev_frames_loaded}")
             # Update NUM_PREV_FRAMES globally to match the loaded model
             NUM_PREV_FRAMES = num_prev_frames_loaded
         else:
             print(f"Using NUM_PREV_FRAMES = {NUM_PREV_FRAMES}")


         world_model = model_class(
             image_channels=IMAGE_CHANNELS,
             time_emb_dim=time_emb_dim_loaded, # Use loaded or default
             num_prev_frames=NUM_PREV_FRAMES # Use potentially updated value
         ).to(device)
         print(f"Instantiated world model: {model_arch_name}")

         # --- Load State Dict ---
         world_model.load_state_dict(checkpoint['model_state_dict'])
         world_model.eval() # Set to evaluation mode
         print(f"Successfully loaded world model state from epoch {checkpoint.get('epoch', 'N/A')}.")

    else:
         print(f"ERROR: Model class '{model_arch_name}' not found in models.py or could not be determined!")
         world_model = None # Ensure model is None if instantiation failed

except FileNotFoundError:
    # This error might still occur if the file path from glob is somehow invalid later
    print(f"Error: World model checkpoint file not found at path: {checkpoint_to_load}")
    world_model = None
except KeyError as e:
    print(f"Error loading world model checkpoint (KeyError: {e}). Checkpoint structure might be different.")
    world_model = None
except Exception as e:
     print(f"An unexpected error occurred loading world model checkpoint {checkpoint_to_load}: {e}.")
     world_model = None


# Ensure model is loaded before proceeding
if world_model is None:
    print("World model could not be loaded. Exiting.")
    sys.exit(1) # Exit script
else:
    print(f"Using World Model: {world_model.__class__.__name__}")

for param in world_model.parameters():
    param.requires_grad = False


# In[12]:


# --- Load Reward Model (SimpleRewardEstimator) ---
reward_model = SimpleRewardEstimator(
    input_channels=IMAGE_CHANNELS,
    image_size=config.IMAGE_SIZE # From config.py
).to(device)

print(f"Attempting to load reward model from: {REWARD_MODEL_PATH}")
try:
    if not os.path.exists(REWARD_MODEL_PATH):
         print(f"Error: Reward model checkpoint file not found: {REWARD_MODEL_PATH}")
         raise FileNotFoundError(f"Checkpoint not found: {REWARD_MODEL_PATH}")

    reward_model.load_state_dict(torch.load(REWARD_MODEL_PATH, map_location=device))
    reward_model.eval()
    print(f"Reward model (SimpleRewardEstimator) loaded successfully from {REWARD_MODEL_PATH}")
except FileNotFoundError:
    raise
except Exception as e:
    print(f"An unexpected error occurred loading reward model: {e}")
    raise

for param in reward_model.parameters():
    param.requires_grad = False


# In[13]:


remote_robot = None # Initialize variable
try:
    # Use the imported RemoteJetBot class
    remote_robot = RemoteJetBot(JETBOT_SERVER_IP)
    # Connection happens in __init__
    time.sleep(1.0) # Give connection time to stabilize
except Exception as e:
     logger.error(f"Failed to initialize RemoteJetBot: {e}")
     sys.exit(1)


# In[14]:


def get_observation_real():
    """Captures an image via RPyC and preprocesses it."""
    if remote_robot is None or remote_robot.conn is None or remote_robot.conn.closed:
        logger.error("RPyC connection not available for get_observation_real.")
        return None

    try:
        # Use the get_frame method from the imported class
        image_bgr = remote_robot.get_frame() # Returns BGR numpy array or None

        if image_bgr is None or image_bgr.size == 0:
             logger.warning("Received invalid/empty image frame from remote robot.")
             return None

        # Convert BGR numpy array to RGB for preprocessing
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess (includes ToPILImage, Resize, ToTensor, Normalize)
        obs_tensor = preprocess(image_rgb).to(device)
        return obs_tensor

    except Exception as e:
        logger.error(f"Error during get_observation_real: {e}")
        return None


def apply_action_real(right_motor_action):
    """Sends the action to the JetBot via RPyC."""
    if remote_robot is None or remote_robot.conn is None or remote_robot.conn.closed:
        logger.error("RPyC connection not available for apply_action_real.")
        return

    try:
        right_motor_speed = float(right_motor_action) * ACTION_SCALE
        left_motor_speed = 0.0 # Keep left motor off

        # Clamping (optional, server might also clamp)
        right_motor_speed = max(min(right_motor_speed, 1.0), -1.0)

        # Use the set_motors method from the imported class
        success = remote_robot.set_motors(left_motor_speed, right_motor_speed)
        # Optional: Check success flag if needed

        # Wait for the frame delay *after* sending the command
        time.sleep(REAL_ROBOT_FRAME_DELAY)

    except Exception as e:
        logger.error(f"Error during RPyC apply_action_real: {e}")



# In[15]:


def format_prev_frames(obs_buffer):
    """ Concatenates previous frames from buffer for model input. """
    # Use the globally determined NUM_PREV_FRAMES
    if len(obs_buffer) < NUM_PREV_FRAMES:
         # This should ideally not happen after initialization
         print(f"Error: Observation buffer has {len(obs_buffer)} frames, needs {NUM_PREV_FRAMES} for prev_frames.")
         return None # Indicate error

    # Get the NUM_PREV_FRAMES most recent observations *excluding the latest one*
    prev_frames_list = list(obs_buffer)[-(NUM_PREV_FRAMES + 1):-1]
    # Concatenate along the channel dimension (C) -> (C*num_prev, H, W)
    prev_frames_tensor = torch.cat(prev_frames_list, dim=0)
    return prev_frames_tensor


def predict_rewards(world_model, reward_model, initial_obs_buffer_list, action_sequences_batch):
    """
    Predicts the cumulative reward for a batch of action sequences using the
    diffusion world model and reward model. Uses real previous frames history.

    Args:
        world_model: The SimpleUNetV1 model.
        reward_model: The SimpleRewardEstimator model.
        initial_obs_buffer_list (list): List of initial observation tensors,
                                        size NUM_PREV_FRAMES + 1.
        action_sequences_batch (torch.Tensor): Batch of action sequences (B, H, N_ACTIONS=1).

    Returns:
        torch.Tensor: Predicted cumulative rewards for each sequence (B,).
    """
    batch_size, horizon, num_actions = action_sequences_batch.shape
    total_rewards = torch.zeros(batch_size, device=device)

    # Ensure models are in evaluation mode
    world_model.eval()
    reward_model.eval()

    # --- Initialize hypothetical observation buffers for each sample in the batch ---
    hypothetical_buffers = [deque(initial_obs_buffer_list, maxlen=NUM_PREV_FRAMES + 1) for _ in range(batch_size)]

    with torch.no_grad():
        for h_step in range(horizon):
            actions_t = action_sequences_batch[:, h_step, :] # Actions for this horizon step (B, 1)

            # --- Prepare inputs for the batch ---
            batch_prev_frames_list = []
            for i in range(batch_size):
                # Use the globally determined NUM_PREV_FRAMES here
                prev_frames = format_prev_frames(hypothetical_buffers[i])
                if prev_frames is None:
                    print(f"Error: Could not format prev_frames for batch sample {i} at horizon {h_step}")
                    return torch.full((batch_size,), -float('inf'), device=device) # Return very low reward on error
                batch_prev_frames_list.append(prev_frames)

            batch_prev_frames_tensor = torch.stack(batch_prev_frames_list, dim=0)

            # --- Predict next observation using DDPM sampling loop for the batch ---
            x = torch.randn((batch_size, IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE), device=device)

            # Inner loop for diffusion steps
            for i in reversed(range(0, NUM_TIMESTEPS)):
                t_val = i
                t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

                # Use imported get_index_from_list
                alpha_t = get_index_from_list(alphas, t, x.shape)
                alpha_hat_t = get_index_from_list(alphas_cumprod, t, x.shape)
                beta_t = get_index_from_list(betas, t, x.shape)

                predicted_noise = world_model(x=x, timestep=t, action=actions_t, prev_frames=batch_prev_frames_tensor)

                term1 = 1 / torch.sqrt(alpha_t)
                term2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
                x = term1 * (x - term2 * predicted_noise)

                if t_val > 0:
                    noise = torch.randn_like(x)
                    variance = torch.sqrt(beta_t)
                    x = x + variance * noise

            predicted_obs_batch = torch.clamp(x, -1.0, 1.0)

            # --- Predict reward ---
            rewards_t = reward_model(predicted_obs_batch).squeeze(-1)

            if torch.isnan(rewards_t).any() or torch.isinf(rewards_t).any():
                print(f"Warning: NaN/Inf detected in predicted rewards at horizon step {h_step}. Replacing with 0.")
                rewards_t = torch.nan_to_num(rewards_t, nan=0.0, posinf=0.0, neginf=0.0)

            total_rewards += rewards_t

            # --- Update hypothetical buffers ---
            for i in range(batch_size):
                 hypothetical_buffers[i].append(predicted_obs_batch[i])

    return total_rewards


def choose_best_action(world_model, reward_model, current_obs_buffer_list):
    """
    Chooses the best discrete action (0.0 or 0.1) by predicting rewards
    for constant action sequences over the horizon.

    Args:
        world_model: The SimpleUNetV1 model.
        reward_model: The SimpleRewardEstimator model.
        current_obs_buffer_list (list): The current observation buffer list
                                       (size NUM_PREV_FRAMES + 1).

    Returns:
        float: The best discrete action (0.0 or 0.1).
        tuple: Predicted rewards for (action 0.0, action 0.1)
    """
    # Use the globally determined NUM_PREV_FRAMES
    if len(current_obs_buffer_list) != NUM_PREV_FRAMES + 1:
        print(f"Error: Action selection requires a full observation buffer ({NUM_PREV_FRAMES + 1} frames). Got {len(current_obs_buffer_list)}.")
        return DISCRETE_ACTIONS[0], (-float('inf'), -float('inf')) # Return default stop action and invalid rewards

    # --- Create the two constant action sequences ---
    action_seq_0 = torch.full((1, HORIZON, N_ACTIONS), DISCRETE_ACTIONS[0], device=device, dtype=torch.float32)
    action_seq_1 = torch.full((1, HORIZON, N_ACTIONS), DISCRETE_ACTIONS[1], device=device, dtype=torch.float32)

    # --- Predict rewards for both sequences ---
    # Calling predict_rewards twice
    reward_0 = predict_rewards(world_model, reward_model, current_obs_buffer_list, action_seq_0)
    reward_1 = predict_rewards(world_model, reward_model, current_obs_buffer_list, action_seq_1)

    # --- Compare rewards and choose action ---
    reward_0_val = reward_0.item() if torch.isfinite(reward_0).all() else -float('inf')
    reward_1_val = reward_1.item() if torch.isfinite(reward_1).all() else -float('inf')

    # **ADDED PRINT STATEMENT**
    print(f"  Predicted Rewards -> Action {DISCRETE_ACTIONS[0]}: {reward_0_val:.4f} | Action {DISCRETE_ACTIONS[1]}: {reward_1_val:.4f}")

    if reward_1_val > reward_0_val:
        best_action = DISCRETE_ACTIONS[1] # Action 0.1
    else:
        # Default to 0.0 if rewards are equal or 0.1 is not better
        best_action = DISCRETE_ACTIONS[0] # Action 0.0

    # **ADDED PRINT STATEMENT**
    print(f"  ==> Chosen Action: {best_action}")
    return best_action, (reward_0_val, reward_1_val) # Return chosen action and predicted rewards


# In[16]:


print("Starting Continuous MPC Control Loop...")
# --- Initialize Observation Buffers ---
# Use the globally determined NUM_PREV_FRAMES
observation_buffer = deque(maxlen=NUM_PREV_FRAMES + 1)
visualization_buffer = deque(maxlen=VISUALIZATION_BUFFER_SIZE) # For storing recent frames for display

print(f"Collecting {NUM_PREV_FRAMES + 1} initial observations...")
initial_obs_collected = 0
while initial_obs_collected < NUM_PREV_FRAMES + 1:
    obs = get_observation_real()
    if obs is not None:
        observation_buffer.append(obs) # Add tensor (C, H, W)
        visualization_buffer.append(obs.cpu().numpy()) # Add numpy version for vis
        initial_obs_collected += 1
        print(f"Collected initial observation {initial_obs_collected}/{NUM_PREV_FRAMES + 1}")
    else:
        print("Failed to get initial observation, retrying...")
        time.sleep(0.5)
    # Add a small delay to avoid overwhelming the server/network
    time.sleep(0.05) # Shorter delay between initial captures

if len(observation_buffer) != NUM_PREV_FRAMES + 1:
     print("Error: Could not collect enough initial observations. Exiting.")
     if 'remote_robot' in locals() and remote_robot:
         remote_robot.cleanup() # Use the cleanup method
     sys.exit(1) # Exit script
else:
     print("Initial observation buffer filled. Starting continuous control.")


# --- Main Control Loop ---
step_count = 0
start_run_time = time.time()
try:
    while True: # Run indefinitely until interrupted
        step_start_time = time.time()

        # Check RPyC connection before planning/acting
        if remote_robot is None or remote_robot.conn is None or remote_robot.conn.closed:
             logger.error("RPyC connection lost. Stopping control loop.")
             break

        step_count += 1
        print(f"\n--- Step {step_count} ---") # **ADDED PRINT STATEMENT**

        # 1. Plan the best action by comparing discrete options
        print("Planning action...") # **ADDED PRINT STATEMENT**
        plan_start_time = time.time()
        # Pass the current buffer (as a list) to the optimizer/chooser
        # **MODIFIED: Get predicted rewards back**
        action_val, predicted_rewards_tuple = choose_best_action(world_model, reward_model, list(observation_buffer))
        plan_duration = time.time() - plan_start_time
        print(f"Planning finished in {plan_duration:.3f}s") # **ADDED PRINT STATEMENT**

        # 2. Apply the chosen action (right motor only) via RPyC
        print(f"Applying action: {action_val:.1f}") # **ADDED PRINT STATEMENT**
        apply_action_real(action_val)

        # 3. Get the next observation via RPyC
        # print("Getting next observation...") # Optional print
        next_obs_tensor = get_observation_real()
        if next_obs_tensor is None:
            logger.warning("Failed to get observation after action. Continuing...")
            # Decide how to handle: continue, retry, or stop?
            # For now, we continue, but the observation buffer won't update correctly.
            # Consider adding a retry mechanism or stopping the loop.
            time.sleep(0.1) # Add a small delay if observation failed
            continue # Skip buffer update if obs failed
            # break # Option: Stop the loop on observation error
        else:
             # print("Observation received.") # Optional print
             # Add new observation to buffers if successful
             observation_buffer.append(next_obs_tensor)
             visualization_buffer.append(next_obs_tensor.cpu().numpy()) # Store numpy version

        step_duration = time.time() - step_start_time
        # Print step info periodically or based on verbosity setting
        # **MODIFIED: Print every step now for more detail**
        print(f"Step {step_count} Summary | Plan Time: {plan_duration:.3f}s | Step Time: {step_duration:.3f}s | Chosen Action: R={action_val:.1f} | Pred Rewards (0.0, 0.1): ({predicted_rewards_tuple[0]:.3f}, {predicted_rewards_tuple[1]:.3f})")

except KeyboardInterrupt:
    print("\nKeyboardInterrupt received. Stopping control loop.")
finally:
    # --- Cleanup ---
    print("Shutting down remote connection...")
    if 'remote_robot' in locals() and remote_robot:
        remote_robot.cleanup() # Use the cleanup method of RemoteJetBot
    print("Remote connection closed.")
    end_run_time = time.time()
    total_duration = end_run_time - start_run_time
    print(f"\n===== MPC Control Loop Finished =====")
    print(f"Ran for {step_count} steps.")
    if total_duration > 0:
        print(f"Total Duration: {time.strftime('%H:%M:%S', time.gmtime(total_duration))}")


# %% [markdown]
# ## Results Visualization (Optional - Shows last frames)

# %%
# Visualize observations using denormalized images from the visualization buffer
if visualization_buffer:
    print(f"\nVisualizing last {len(visualization_buffer)} captured observations...")
    vis_obs_np = np.array(visualization_buffer) # Shape: (num_steps, C, H, W)
    num_obs_to_show = min(len(vis_obs_np), 10) # Show up to 10 last frames

    fig, axes = plt.subplots(1, num_obs_to_show, figsize=(num_obs_to_show * 2.5, 3))
    if num_obs_to_show == 1: axes = [axes] # Make iterable if only one subplot

    # Get the indices for the last num_obs_to_show frames
    start_vis_index = len(vis_obs_np) - num_obs_to_show

    for i in range(num_obs_to_show):
        obs_index = start_vis_index + i
        obs_tensor = torch.from_numpy(vis_obs_np[obs_index]).float()
        obs_denorm = denormalize(obs_tensor)
        obs_img_display = obs_denorm.permute(1, 2, 0).cpu().numpy()
        obs_img_display = np.clip(obs_img_display, 0, 1)

        axes[i].imshow(obs_img_display)
        # Title relative to the end of the run
        axes[i].set_title(f"Step {step_count - num_obs_to_show + i + 1}")
        axes[i].axis('off')
    plt.suptitle(f"Last {num_obs_to_show} Observations")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("No observation data captured in the visualization buffer.")


# In[ ]:




