#!/usr/bin/env python
# coding: utf-8

# # Reward Estimator Training
# 
# This notebook trains a model to predict the reward associated with a given
# state (represented by the current image frame).
# It uses the reward labels generated interactively via `reward_labeling.ipynb`.
# 

# In[1]:


# Basic Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import copy # For saving best model
from models import SimpleRewardEstimator
# Project specific imports - ensure these files are accessible
import config
from importnb import Notebook
with Notebook():
    from jetbot_dataset import JetbotDataset
    from reward_dataset import RewardDatasetSingleFrame

print(f"Using device: {config.DEVICE}")
print(f"Reward labels expected at: {config.MANUAL_COLLECTED_REWARD_CSV}") # Use the path from config
print(f"Main data CSV: {config.CSV_PATH}")
print(f"Main data dir: {config.DATA_DIR}")


# ## Configuration
# Define training hyperparameters. Some can be reused from `config.py`.

# In[ ]:


N_FRAMES          = 4                     # prev frames to stack (plus the current)
BATCH_SIZE        = 64
LR                = 3e-4
EPOCHS            = 20
IMG_SIZE          = 96                    # matches world-model training
CHECKPOINT_DIR    = Path("outputs/reward_estimator_resnet")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


REWARD_BATCH_SIZE = 16        # Can be different from diffusion model batch size
REWARD_LEARNING_RATE = 1e-4
REWARD_NUM_EPOCHS = 50
REWARD_VALIDATION_SPLIT = 0.1 # Percentage of labeled data to use for validation
REWARD_SEED = 42             # For reproducible train/val split
REWARD_MODEL_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, 'reward_estimator')
REWARD_CHECKPOINT_FILENAME = "reward_estimator_best.pth"

# Ensure output directory exists
os.makedirs(REWARD_MODEL_OUTPUT_DIR, exist_ok=True)

# --- Data Loading Parameters (from config.py) ---
IMAGE_SIZE = config.IMAGE_SIZE
REWARD_CSV_PATH = config.MANUAL_COLLECTED_REWARD_CSV # Path to the labels
MAIN_CSV_PATH = config.CSV_PATH
MAIN_DATA_DIR = config.DATA_DIR
TRANSFORM = config.TRANSFORM # Use the same transform as the diffusion model for consistency



# ## Reward Estimator Model Definition
# Uses a CNN architecture. This is a simple example; you might need a more complex model depending on performance. It takes the concatenated image sequence as input.

# In[3]:


reward_model = SimpleRewardEstimator()
print("Reward Estimator Model Architecture (Single Frame Input):")
print(reward_model)

# Move model to device
reward_model.to(config.DEVICE)


# ## Data Loading and Splitting
# Instantiate the dataset and create DataLoaders for training and validation.
# 

# In[3]:


# Instantiate the dataset
try:
    # MODIFIED: Using RewardDatasetSingleFrame
    reward_dataset = RewardDatasetSingleFrame(
        main_csv_path=MAIN_CSV_PATH,
        reward_csv_path=REWARD_CSV_PATH,
        data_dir=MAIN_DATA_DIR,
        image_size=IMAGE_SIZE,
        transform=TRANSFORM
    )
except Exception as e:
    print(f"Failed to instantiate RewardDatasetSingleFrame: {e}")
    reward_dataset = None # Set to None to prevent further errors

if reward_dataset and len(reward_dataset) > 0:
    # Split into training and validation sets
    total_size = len(reward_dataset)
    val_size = int(REWARD_VALIDATION_SPLIT * total_size)
    train_size = total_size - val_size

    # Ensure sizes are valid
    if train_size <= 0 or val_size <=0:
        print(f"Warning: Dataset size ({total_size}) is too small for validation split ({REWARD_VALIDATION_SPLIT}). Using full dataset for training.")
        train_subset = reward_dataset
        val_loader = None # No validation
    else:
        print(f"Splitting dataset: Train size = {train_size}, Validation size = {val_size}")
        # Use torch.manual_seed for reproducible splits
        torch.manual_seed(REWARD_SEED)
        train_subset, val_subset = random_split(reward_dataset, [train_size, val_size])
        # Create Validation Loader only if val_size > 0
        val_loader = DataLoader(
            val_subset,
            batch_size=REWARD_BATCH_SIZE,
            shuffle=False, # No need to shuffle validation set
            num_workers=0,
            pin_memory=True
        )


    # Create Training Loader
    train_loader = DataLoader(
        train_subset,
        batch_size=REWARD_BATCH_SIZE,
        shuffle=True,
        num_workers=0, # Adjust based on your system
        pin_memory=True
    )


    print("DataLoaders created.")
    # Optional: Display a sample batch
    try:
       sample_batch = next(iter(train_loader))
       print("Sample batch - Input shape:", sample_batch[0].shape)
       print("Sample batch - Reward shape:", sample_batch[1].shape)
    except Exception as e:
        print(f"Could not load a sample batch: {e}")

else:
    print("Skipping DataLoader creation as dataset is empty or failed to load.")
    train_loader = None
    val_loader = None




# In[7]:


pd


# ## Training Setup
# Define the loss function and optimizer.
# 

# In[9]:


if reward_dataset and len(reward_dataset) > 0:
    # Loss Function (Mean Squared Error is common for regression)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(reward_model.parameters(), lr=REWARD_LEARNING_RATE)

    print("Loss function and optimizer defined.")
else:
    print("Skipping training setup.")
    criterion = None
    optimizer = None


# ## Training Loop

# In[10]:


train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

# MODIFIED: Check train_loader exists, val_loader is optional now
if train_loader and criterion and optimizer:
    print("Starting training...")
    for epoch in range(REWARD_NUM_EPOCHS):
        # --- Training Phase ---
        reward_model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{REWARD_NUM_EPOCHS} [Train]", leave=False)

        # MODIFIED: Unpack only image and reward
        for i, (images, rewards) in enumerate(train_pbar):
            images = images.to(config.DEVICE)
            rewards = rewards.to(config.DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # MODIFIED: Pass only images to the model
            predicted_rewards = reward_model(images)

            # Calculate loss
            loss = criterion(predicted_rewards, rewards)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_pbar.close()

        # --- Validation Phase (Optional) ---
        epoch_val_loss = None
        if val_loader: # Only run validation if val_loader exists
            reward_model.eval()
            running_val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{REWARD_NUM_EPOCHS} [Val]", leave=False)

            with torch.no_grad():
                 # MODIFIED: Unpack only image and reward
                for images, rewards in val_pbar:
                    images = images.to(config.DEVICE)
                    rewards = rewards.to(config.DEVICE)

                    # MODIFIED: Pass only images to the model
                    predicted_rewards = reward_model(images)
                    loss = criterion(predicted_rewards, rewards)
                    running_val_loss += loss.item()
                    val_pbar.set_postfix({'loss': loss.item()})

            epoch_val_loss = running_val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            val_pbar.close()

            print(f"Epoch {epoch+1}/{REWARD_NUM_EPOCHS} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

            # --- Save Best Model ---
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = copy.deepcopy(reward_model.state_dict())
                print(f"  -> New best model saved with validation loss: {best_val_loss:.6f}")
                checkpoint_path = os.path.join(REWARD_MODEL_OUTPUT_DIR, REWARD_CHECKPOINT_FILENAME)
                torch.save(best_model_state, checkpoint_path)
                print(f"  -> Best model state saved to {checkpoint_path}")
        else:
             # If no validation, just print training loss
             print(f"Epoch {epoch+1}/{REWARD_NUM_EPOCHS} - Train Loss: {epoch_train_loss:.6f}")
             # Optionally save based on training loss improvement or just save periodically/last
             # Example: Saving based on training loss (less common than validation)
             # if epoch_train_loss < best_val_loss: # Using best_val_loss variable name but tracking train loss
             #     best_val_loss = epoch_train_loss
             #     best_model_state = copy.deepcopy(reward_model.state_dict())
             #     checkpoint_path = os.path.join(REWARD_MODEL_OUTPUT_DIR, REWARD_CHECKPOINT_FILENAME)
             #     torch.save(best_model_state, checkpoint_path)
             #     print(f"  -> Model state saved with training loss: {best_val_loss:.6f} to {checkpoint_path}")


    print("Training finished.")

    # --- Save the final model state ---
    final_checkpoint_path = os.path.join(REWARD_MODEL_OUTPUT_DIR, "reward_estimator_single_frame_final.pth")
    torch.save(reward_model.state_dict(), final_checkpoint_path)
    print(f"Final model state saved to {final_checkpoint_path}")

    # Also save the best model if validation was not used (will be the last state saved based on training loss if that logic was enabled)
    if not val_loader and best_model_state:
         checkpoint_path = os.path.join(REWARD_MODEL_OUTPUT_DIR, REWARD_CHECKPOINT_FILENAME)
         torch.save(best_model_state, checkpoint_path)
         print(f"Final 'best' model state based on training saved to {checkpoint_path}")


else:
    print("Training skipped due to issues in data loading or setup.")


# ## Plotting Results 

# In[11]:


plt.figure(figsize=(10, 5))
if train_losses:
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
if val_losses:
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')

# Add labels/title only if something was plotted
if train_losses or val_losses:
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Reward Estimator Training and Validation Loss (Single Frame)')
    plt.legend()
    plt.grid(True)
    plot_save_path = os.path.join(REWARD_MODEL_OUTPUT_DIR, 'reward_training_loss_plot_single_frame.png')
    plt.savefig(plot_save_path)
    print(f"Loss plot saved to {plot_save_path}")
    plt.show()
else:
    print("No training data to plot.")


# In[ ]:




