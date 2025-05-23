#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
import datetime # For epoch timing and timestamping
from torchvision import transforms
from collections import deque # For moving average
from dataclasses import dataclass # Added
from typing import List, Optional, Dict, Any # Added

# Your project's specific imports
import config # Your config.py
import models # Your models.py (which should import from diamond_models.ipynb)

# Import dataset from your jetbot_dataset.ipynb
from importnb import Notebook
with Notebook():
    from jetbot_dataset import JetbotDataset # Make sure class name matches

from PIL import Image as PILImage
# import matplotlib.pyplot as plt # Already imported

print("Imports successful.")


# In[2]:


print("--- Configuration ---")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# In[3]:


# Denoiser & InnerModel specific
DM_SIGMA_DATA = getattr(config, 'DM_SIGMA_DATA', 0.5)
DM_SIGMA_OFFSET_NOISE = getattr(config, 'DM_SIGMA_OFFSET_NOISE', 0.1)
DM_NOISE_PREVIOUS_OBS = getattr(config, 'DM_NOISE_PREVIOUS_OBS', True)
DM_IMG_CHANNELS = getattr(config, 'DM_IMG_CHANNELS', 3)
DM_NUM_STEPS_CONDITIONING = getattr(config, 'DM_NUM_STEPS_CONDITIONING', config.NUM_PREV_FRAMES)
DM_COND_CHANNELS = getattr(config, 'DM_COND_CHANNELS', 256)
DM_UNET_DEPTHS = getattr(config, 'DM_UNET_DEPTHS', [2, 2, 2, 2])
DM_UNET_CHANNELS = getattr(config, 'DM_UNET_CHANNELS', [128, 256, 512, 1024]) # Using config.py
DM_UNET_ATTN_DEPTHS = getattr(config, 'DM_UNET_ATTN_DEPTHS', [False, False, True, True])
DM_NUM_ACTIONS = getattr(config, 'DM_NUM_ACTIONS', 2)
DM_IS_UPSAMPLER = getattr(config, 'DM_IS_UPSAMPLER', False)
DM_UPSAMPLING_FACTOR = getattr(config, 'DM_UPSAMPLING_FACTOR', None)

# Sampler specific (for inference/visualization)
SAMPLER_NUM_STEPS = getattr(config, 'SAMPLER_NUM_STEPS', 50)
SAMPLER_SIGMA_MIN = getattr(config, 'SAMPLER_SIGMA_MIN', 0.002)
SAMPLER_SIGMA_MAX = getattr(config, 'SAMPLER_SIGMA_MAX', 80.0)
SAMPLER_RHO = getattr(config, 'SAMPLER_RHO', 7.0)
# Additional Karras sampler params from config if they exist, otherwise defaults in dataclass used
SAMPLER_ORDER = getattr(config, 'SAMPLER_ORDER', 1)
SAMPLER_S_CHURN = getattr(config, 'SAMPLER_S_CHURN', 0.0)
SAMPLER_S_TMIN = getattr(config, 'SAMPLER_S_TMIN', 0.0)
SAMPLER_S_TMAX = getattr(config, 'SAMPLER_S_TMAX', float("inf"))
SAMPLER_S_NOISE = getattr(config, 'SAMPLER_S_NOISE', 1.0)


# Training specific
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
NUM_EPOCHS = config.NUM_EPOCHS
SAVE_MODEL_EVERY = config.SAVE_MODEL_EVERY
SAMPLE_EVERY = config.SAMPLE_EVERY
PLOT_EVERY = config.PLOT_EVERY
GRAD_CLIP_VALUE = getattr(config, 'GRAD_CLIP_VALUE', 1.0)

DM_SIGMA_P_MEAN = getattr(config, 'DM_SIGMA_P_MEAN', -1.2) 
DM_SIGMA_P_STD = getattr(config, 'DM_SIGMA_P_STD', 1.2)   
DM_SIGMA_MIN_TRAIN = getattr(config, 'DM_SIGMA_MIN_TRAIN', 0.002) 
DM_SIGMA_MAX_TRAIN = getattr(config, 'DM_SIGMA_MAX_TRAIN', 80.0)  

EARLY_STOPPING_PATIENCE = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)
EARLY_STOPPING_MIN_EPOCHS = getattr(config, 'MIN_EPOCHS', 20) # Renamed from MIN_EPOCHS in config to avoid ambiguity
EARLY_STOPPING_PERCENTAGE = getattr(config, 'EARLY_STOPPING_PERCENTAGE', 0.1) 
TRAIN_MOVING_AVG_WINDOW = getattr(config, 'TRAIN_MOVING_AVG_WINDOW', 10) 
VAL_MOVING_AVG_WINDOW = getattr(config, 'VAL_MOVING_AVG_WINDOW', 5) 

print("Configuration loaded.")


# In[4]:


data_transform = config.TRANSFORM

full_dataset = JetbotDataset(
    csv_path=config.CSV_PATH,
    data_dir=config.DATA_DIR,
    image_size=config.IMAGE_SIZE,
    num_prev_frames=config.NUM_PREV_FRAMES,
    transform=data_transform
)
print(f"Full dataset size: {len(full_dataset)}")

split_file_path = os.path.join(config.OUTPUT_DIR, getattr(config, 'SPLIT_DATASET_FILENAME', 'dataset_split.pth'))
if os.path.exists(split_file_path):
    print(f"Loading dataset split from {split_file_path}")
    split_data = torch.load(split_file_path)
    train_indices, val_indices = split_data['train_indices'], split_data['val_indices']
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
else:
    print("Creating new train/val split...")
    total_size = len(full_dataset)
    train_size = int(total_size * 0.9)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size]) # Using torch.random_split by default
    torch.save({
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
    }, split_file_path)
    print(f"Saved new dataset split to {split_file_path}")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Train Dataloader: {len(train_dataloader)} batches of size {BATCH_SIZE}")
print(f"Validation Dataloader: {len(val_dataloader)} batches of size {BATCH_SIZE}")


# In[5]:


print("--- Initializing Models ---")

# 1. InnerModel (U-Net part of the Denoiser)
try:
    inner_model_config = models.InnerModelConfig( # This is diamond_models.InnerModelConfig
        img_channels=DM_IMG_CHANNELS,
        num_steps_conditioning=DM_NUM_STEPS_CONDITIONING, # This is NUM_PREV_FRAMES
        cond_channels=DM_COND_CHANNELS,
        depths=DM_UNET_DEPTHS,
        channels=DM_UNET_CHANNELS,
        attn_depths=DM_UNET_ATTN_DEPTHS,
        num_actions=DM_NUM_ACTIONS, # From config, e.g., 2 for JetBot
        is_upsampler=DM_IS_UPSAMPLER # Will be set by DenoiserConfig later too
    )
    inner_model_instance = models.InnerModel(inner_model_config).to(DEVICE) # diamond_models.InnerModelImpl
    print("Using InnerModel (Diamond-style U-Net) as the inner model.")
    print(f"InnerModelImpl parameter count: {sum(p.numel() for p in inner_model_instance.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"Could not instantiate InnerModelImpl due to: {e}. Ensure 'InnerModelConfig', 'InnerModelImpl', dependencies, and DM_* config parameters are correct.")
    raise

# 2. Denoiser (using diamond_models.Denoiser)
try:
    denoiser_cfg = models.DenoiserConfig( # Our new dataclass
        inner_model=inner_model_config, # Pass the config, not the instance here if Denoiser instantiates it.
                                        # diamond_models.Denoiser takes an InnerModelConfig for its own InnerModel.
                                        # Re-checking diamond_models.py: Denoiser.__init__(self, cfg: DenoiserConfig)
                                        # cfg.inner_model.is_upsampler = self.is_upsampler
                                        # self.inner_model = InnerModel(cfg.inner_model) <--- Correct, it expects InnerModelConfig in DenoiserConfig
        sigma_data=DM_SIGMA_DATA,
        sigma_offset_noise=DM_SIGMA_OFFSET_NOISE,
        noise_previous_obs=DM_NOISE_PREVIOUS_OBS,
        upsampling_factor=DM_UPSAMPLING_FACTOR
    )
    # Ensure DenoiserConfig's inner_model field matches diamond_models.InnerModelConfig type
    # The `models.InnerModelConfig` is already an alias to `diamond_models.InnerModelConfig`
    denoiser = models.Denoiser(cfg=denoiser_cfg).to(DEVICE) # Pass the config object
    
    # Setup training sigma distribution for the Denoiser
    sigma_dist_train_cfg = models.SigmaDistributionConfig(
        loc=DM_SIGMA_P_MEAN,
        scale=DM_SIGMA_P_STD,
        sigma_min=DM_SIGMA_MIN_TRAIN,
        sigma_max=DM_SIGMA_MAX_TRAIN
    )
    denoiser.setup_training(sigma_dist_train_cfg) # Call setup_training
    print(f"Denoiser model created and training sigma distribution configured. Total parameter count: {sum(p.numel() for p in denoiser.parameters() if p.requires_grad):,}")

except Exception as e:
    print(f"Could not instantiate or configure Denoiser (from diamond_models.py) due to: {e}.")
    raise

# 3. DiffusionSampler (using diamond_models.DiffusionSampler)
try:
    sampler_cfg = models.DiffusionSamplerConfig( # Our new dataclass
        num_steps_denoising=SAMPLER_NUM_STEPS,
        sigma_min=SAMPLER_SIGMA_MIN,
        sigma_max=SAMPLER_SIGMA_MAX,
        rho=SAMPLER_RHO,
        order=SAMPLER_ORDER,
        s_churn=SAMPLER_S_CHURN,
        s_tmin=SAMPLER_S_TMIN,
        s_tmax=SAMPLER_S_TMAX,
        s_noise=SAMPLER_S_NOISE
    )
    diffusion_sampler = models.DiffusionSampler( # This is diamond_models.DiffusionSampler
        denoiser=denoiser, # Pass the denoiser instance
        cfg=sampler_cfg    # Pass the sampler config object
    ) # Sampler itself might not need .to(DEVICE) if it doesn't have parameters
    print("DiffusionSampler created for visualization.")
except Exception as e:
    print(f"Could not instantiate DiffusionSampler (from diamond_models.py) due to: {e}.")
    raise


# In[6]:


print("--- Setting up Optimizer ---")
optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LEARNING_RATE)
lr_scheduler = None # Placeholder
print(f"Optimizer: AdamW with LR={LEARNING_RATE}")


# In[7]:


START_EPOCH = 0
BEST_TRAIN_LOSS_MA_FROM_CKPT = float('inf')
PREVIOUS_BEST_TRAIN_MODEL_PATH = None

# Correctly use LOAD_CHECKPOINT from config.py for the specific path
load_path_config = config.LOAD_CHECKPOINT 
best_train_loss_model_default_path = os.path.join(config.CHECKPOINT_DIR, "denoiser_model_best_train_loss.pth")

load_path = load_path_config
if load_path: # If a specific path is set in config, use it
    print(f"Attempting to load checkpoint from config.LOAD_CHECKPOINT: {load_path}")
elif os.path.exists(best_train_loss_model_default_path): # Else, try the default best
    load_path = best_train_loss_model_default_path
    print(f"No specific checkpoint in config.LOAD_CHECKPOINT. Found existing best_train_loss model: {load_path}")


if load_path and os.path.exists(load_path):
    print(f"Loading checkpoint from: {load_path}")
    try:
        checkpoint = torch.load(load_path, map_location=DEVICE)
        denoiser.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint.get('epoch', 0) + 1
        BEST_TRAIN_LOSS_MA_FROM_CKPT = checkpoint.get('best_train_loss_ma', float('inf'))
        if load_path.endswith("denoiser_model_best_train_loss.pth"): # Ensure we track the correct one
            PREVIOUS_BEST_TRAIN_MODEL_PATH = load_path
        print(f"Resuming training from epoch {START_EPOCH}. Last best train_loss_ma: {BEST_TRAIN_LOSS_MA_FROM_CKPT:.6f}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        START_EPOCH = 0; BEST_TRAIN_LOSS_MA_FROM_CKPT = float('inf')
else:
    if load_path_config: # If a path was specified but not found
        print(f"Specified checkpoint not found: {load_path_config}. Starting from scratch.")
    else: # No checkpoint specified and default best not found
        print("No checkpoint found or specified. Starting from scratch.")


# In[8]:


def tensor_to_pil(tensor_img):
    tensor_img = (tensor_img.clamp(-1, 1) + 1) / 2
    tensor_img = tensor_img.detach().cpu().permute(1, 2, 0).numpy()
    if tensor_img.shape[2] == 1:
        tensor_img = tensor_img.squeeze(2)
    # Ensure array is writeable for PIL
    if not tensor_img.flags.writeable:
        tensor_img = np.ascontiguousarray(tensor_img)
    if tensor_img.dtype != np.uint8: # This check might be problematic if tensor_img is already uint8
        pil_img_array = (tensor_img * 255).astype(np.uint8)
    else:
        pil_img_array = tensor_img # Already uint8
    pil_img = PILImage.fromarray(pil_img_array)
    return pil_img

def save_visualization_samples(generated_tensors, gt_tensors, epoch, save_dir, max_imgs=4, prefix="train_vis"):
    os.makedirs(save_dir, exist_ok=True)
    num_samples_to_show = min(max_imgs, generated_tensors.shape[0], gt_tensors.shape[0])
    if num_samples_to_show == 0: return

    generated_tensors = generated_tensors[:num_samples_to_show]
    gt_tensors = gt_tensors[:num_samples_to_show]

    fig_height = 6
    fig, axs = plt.subplots(2, num_samples_to_show, figsize=(num_samples_to_show * 3, fig_height))
    if num_samples_to_show == 1: axs = np.array(axs).reshape(2,1) # Ensure axs is always 2D array

    for i in range(num_samples_to_show):
        try:
            gen_pil = tensor_to_pil(generated_tensors[i])
            gt_pil = tensor_to_pil(gt_tensors[i])
            axs[0, i].imshow(gt_pil); axs[0, i].set_title(f"GT {i+1}"); axs[0, i].axis('off')
            axs[1, i].imshow(gen_pil); axs[1, i].set_title(f"Gen {i+1}"); axs[1, i].axis('off')
        except Exception as e:
            print(f"Error visualizing image {i}: {e}")
            if num_samples_to_show > 1 : axs[0,i].axis('off'); axs[1,i].axis('off')
            else: fig.clear(); plt.text(0.5, 0.5, "Error displaying image", ha="center", va="center")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:04d}_samples.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization samples to {save_path}")

print("Visualization helpers defined.")


# In[9]:


def train_denoiser_epoch(denoiser_model, train_dl, opt, grad_clip_val, device, epoch_num_for_log=""):
    denoiser_model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_dl, desc=f"Epoch {epoch_num_for_log} [Train]", leave=False)
    
    # Constants for reshaping and preparing Batch object
    num_prev_frames = config.NUM_PREV_FRAMES
    c, h, w = DM_IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE
    
    for batch_idx, (target_img_batch, action_batch, prev_frames_flat_batch) in enumerate(progress_bar):
        opt.zero_grad()
        
        current_batch_size = target_img_batch.shape[0] # Handle last batch if drop_last=False

        # Move data to device
        target_img_batch = target_img_batch.to(device) # (B, C, H, W)
        action_batch = action_batch.to(device)         # (B, 1)
        prev_frames_flat_batch = prev_frames_flat_batch.to(device) # (B, NumPrevFrames*C, H, W)

        # 1. Prepare Batch.obs: (B, NumPrevFrames + 1, C, H, W)
        prev_frames_seq_batch = prev_frames_flat_batch.view(current_batch_size, num_prev_frames, c, h, w)
        batch_obs_tensor = torch.cat((prev_frames_seq_batch, target_img_batch.unsqueeze(1)), dim=1)

        # 2. Prepare Batch.act: (B, NumPrevFrames)
        # Tiling the single action from dataloader to match NUM_PREV_FRAMES.
        # This is a simplification; ideally, dataset provides action history.
        batch_act_tensor = action_batch.repeat(1, num_prev_frames).long() # Ensure actions are long for embedding
        
        # 3. Prepare Batch.mask_padding (Optional, defaults to all valid if None)
        # Denoiser uses mask_padding[:, n+i], so it's (B, T). Let's assume all data is valid.
        batch_mask_padding = torch.ones(current_batch_size, num_prev_frames + 1, device=device, dtype=torch.bool)

        # Create Batch object
        current_batch_obj = models.Batch(
            obs=batch_obs_tensor, 
            act=batch_act_tensor, 
            mask_padding=batch_mask_padding
            # info=None # Not used for this non-upsampler case
        )
        
        # Denoiser.forward expects a Batch object and returns (loss, logs_dict)
        loss, logs = denoiser_model(current_batch_obj) 
        
        loss.backward()
        if grad_clip_val > 0: torch.nn.utils.clip_grad_norm_(denoiser_model.parameters(), grad_clip_val)
        opt.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item(), "DenoisingLoss": logs.get("loss_denoising", "N/A")})
        
    return total_loss / len(train_dl) if len(train_dl) > 0 else 0

@torch.no_grad()
def validate_denoiser_epoch(denoiser_model, val_dl, device, epoch_num_for_log=""):
    denoiser_model.eval()
    total_loss = 0.0
    progress_bar = tqdm(val_dl, desc=f"Epoch {epoch_num_for_log} [Valid]", leave=False)

    num_prev_frames = config.NUM_PREV_FRAMES
    c, h, w = DM_IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE

    for batch_idx, (target_img_batch, action_batch, prev_frames_flat_batch) in enumerate(progress_bar):
        current_batch_size = target_img_batch.shape[0]

        target_img_batch = target_img_batch.to(device)
        action_batch = action_batch.to(device)
        prev_frames_flat_batch = prev_frames_flat_batch.to(device)

        prev_frames_seq_batch = prev_frames_flat_batch.view(current_batch_size, num_prev_frames, c, h, w)
        batch_obs_tensor = torch.cat((prev_frames_seq_batch, target_img_batch.unsqueeze(1)), dim=1)
        batch_act_tensor = action_batch.repeat(1, num_prev_frames).long()
        batch_mask_padding = torch.ones(current_batch_size, num_prev_frames + 1, device=device, dtype=torch.bool)

        current_batch_obj = models.Batch(
            obs=batch_obs_tensor, 
            act=batch_act_tensor, 
            mask_padding=batch_mask_padding
        )
        
        loss, logs = denoiser_model(current_batch_obj)
        total_loss += loss.item()
        progress_bar.set_postfix({"Val Loss": loss.item(), "DenoisingLoss": logs.get("loss_denoising", "N/A")})
        
    return total_loss / len(val_dl) if len(val_dl) > 0 else 0

print("Training and validation epoch functions adapted for Batch object and Denoiser.forward.")


# In[10]:


print("--- Starting Training Process ---")
overall_training_start_time = time.time() 

all_train_losses_for_plot = [] 
all_val_losses_for_plot = []   

train_loss_moving_avg_q = deque(maxlen=TRAIN_MOVING_AVG_WINDOW)
best_train_loss_ma = BEST_TRAIN_LOSS_MA_FROM_CKPT 
epochs_without_improvement_train = 0
previous_best_train_model_path = PREVIOUS_BEST_TRAIN_MODEL_PATH 

val_loss_moving_avg_q = deque(maxlen=VAL_MOVING_AVG_WINDOW)
final_epoch_completed = START_EPOCH -1 # Corrected initialization

for epoch in range(START_EPOCH, NUM_EPOCHS):
    epoch_start_time = time.time()
    current_epoch_num_for_log = epoch + 1
    # final_epoch_completed = epoch # Moved to end of loop for correct value if early stopping

    avg_train_loss = train_denoiser_epoch(
        denoiser, train_dataloader, optimizer,
        GRAD_CLIP_VALUE, DEVICE, current_epoch_num_for_log
    )
    all_train_losses_for_plot.append(avg_train_loss)
    train_loss_moving_avg_q.append(avg_train_loss)
    current_train_moving_avg = sum(train_loss_moving_avg_q) / len(train_loss_moving_avg_q) if train_loss_moving_avg_q else float('inf')

    avg_val_loss = validate_denoiser_epoch(
        denoiser, val_dataloader, DEVICE, current_epoch_num_for_log
    )
    all_val_losses_for_plot.append(avg_val_loss)
    val_loss_moving_avg_q.append(avg_val_loss) 
    current_val_moving_avg = sum(val_loss_moving_avg_q) / len(val_loss_moving_avg_q) if val_loss_moving_avg_q else float('inf')

    epoch_duration_seconds = time.time() - epoch_start_time
    epoch_duration_formatted = str(datetime.timedelta(seconds=epoch_duration_seconds))

    print(f"Epoch {current_epoch_num_for_log}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} (MA: {current_train_moving_avg:.4f}), Val Loss: {avg_val_loss:.4f} (MA: {current_val_moving_avg:.4f}), Duration: {epoch_duration_formatted}")

    if lr_scheduler: lr_scheduler.step(avg_val_loss if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

    should_stop_early = False
    # Early stopping logic (using EARLY_STOPPING_MIN_EPOCHS correctly)
    if current_epoch_num_for_log > EARLY_STOPPING_MIN_EPOCHS: # Check after min epochs completed
        if current_train_moving_avg < best_train_loss_ma : 
            # ... (rest of early stopping logic seems okay, ensure it uses current_epoch_num_for_log correctly)
            improvement_over_absolute_best = (best_train_loss_ma - current_train_moving_avg) / abs(best_train_loss_ma + 1e-9) * 100
            print(f"  Train Loss MA improved to {current_train_moving_avg:.6f} from {best_train_loss_ma:.6f} ({improvement_over_absolute_best:.2f}% improvement).")
            best_train_loss_ma = current_train_moving_avg
            epochs_without_improvement_train = 0
            new_best_model_path = os.path.join(config.CHECKPOINT_DIR, "denoiser_model_best_train_loss.pth")
            torch.save({
                'epoch': epoch, 'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_train_loss, 
                'val_loss': avg_val_loss, 'best_train_loss_ma': best_train_loss_ma
            }, new_best_model_path)
            print(f"  Saved new best model (train loss MA) at epoch {current_epoch_num_for_log}")
            if previous_best_train_model_path and previous_best_train_model_path != new_best_model_path and os.path.exists(previous_best_train_model_path):
                try: os.remove(previous_best_train_model_path); print(f"  Deleted previous best train model: {previous_best_train_model_path}")
                except OSError as e: print(f"  Warning: Could not delete previous best train model '{previous_best_train_model_path}': {e}")
            previous_best_train_model_path = new_best_model_path
        else: 
            epochs_without_improvement_train += 1
            print(f"  No improvement in train loss MA for {epochs_without_improvement_train} epoch(s). Best MA: {best_train_loss_ma:.6f}, Current MA: {current_train_moving_avg:.6f}")
            if epochs_without_improvement_train >= EARLY_STOPPING_PATIENCE:
                # ... (percentage improvement check)
                idx_before_streak_started = len(all_train_losses_for_plot) - epochs_without_improvement_train -1 # Index of the epoch before non-improvement streak
                # Ensure indices are valid
                if idx_before_streak_started >= 0:
                    # Calculate MA from historical_losses_for_ma of length TRAIN_MOVING_AVG_WINDOW ending at idx_before_streak_started
                    historical_window_start = max(0, idx_before_streak_started - TRAIN_MOVING_AVG_WINDOW + 1)
                    historical_losses_for_ma_calc = all_train_losses_for_plot[historical_window_start : idx_before_streak_started + 1]

                    if len(historical_losses_for_ma_calc) >= TRAIN_MOVING_AVG_WINDOW // 2 : # Need at least half window
                        historical_train_ma = sum(historical_losses_for_ma_calc) / len(historical_losses_for_ma_calc)
                        # Improvement is positive if current_train_moving_avg is smaller
                        percentage_improvement_vs_historical = (historical_train_ma - current_train_moving_avg) / abs(historical_train_ma + 1e-9) * 100
                        print(f"  Patience met. Current Train MA: {current_train_moving_avg:.6f}, Historical MA before streak ({len(historical_losses_for_ma_calc)} epochs): {historical_train_ma:.6f}. Improvement: {percentage_improvement_vs_historical:.2f}%")
                        if percentage_improvement_vs_historical < EARLY_STOPPING_PERCENTAGE:
                            should_stop_early = True
                            print(f"Early stopping triggered: Improvement {percentage_improvement_vs_historical:.2f}% < threshold {EARLY_STOPPING_PERCENTAGE}%.")
                    else:
                        print(f"  Patience met, but not enough historical data ({len(historical_losses_for_ma_calc)} points out of {TRAIN_MOVING_AVG_WINDOW}) to reliably calculate percentage improvement for early stopping.")
                else:
                     print(f"  Patience met, but not enough historical data (idx_before_streak_started = {idx_before_streak_started}) to compare.")

    
    if (current_epoch_num_for_log % SAVE_MODEL_EVERY == 0) or (epoch == NUM_EPOCHS - 1):
        is_best_this_epoch = current_train_moving_avg == best_train_loss_ma # Check if current MA is the best overall
        # Avoid saving regular checkpoint if it's also the best_train_loss epoch to prevent duplicate saves
        if not (is_best_this_epoch and os.path.join(config.CHECKPOINT_DIR, "denoiser_model_best_train_loss.pth") == previous_best_train_model_path):
             torch.save({
                'epoch': epoch, 'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_train_loss,
                'val_loss': avg_val_loss, 'best_train_loss_ma': best_train_loss_ma # Save current best_train_loss_ma
            }, os.path.join(config.CHECKPOINT_DIR, f"denoiser_model_epoch_{current_epoch_num_for_log:04d}.pth"))
             print(f"Saved model checkpoint at epoch {current_epoch_num_for_log}")
    
    final_epoch_completed = epoch # Update last completed epoch here
    if should_stop_early: break

    if (current_epoch_num_for_log % SAMPLE_EVERY == 0) or (epoch == NUM_EPOCHS - 1):
        print(f"Epoch {current_epoch_num_for_log}: Generating visualization samples...")
        denoiser.eval()
        try: 
            vis_batch_target_img, vis_batch_act_single, vis_batch_prev_frames_flat = next(iter(val_dataloader))
        except StopIteration: 
            val_dataloader_iter = iter(DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)); 
            vis_batch_target_img, vis_batch_act_single, vis_batch_prev_frames_flat = next(val_dataloader_iter)
        
        n_vis_samples = min(4, vis_batch_target_img.shape[0])
        
        # Prepare inputs for DiffusionSampler.sample(prev_obs, prev_act)
        # 1. prev_obs: Needs to be (B, NumPrevFrames, C, H, W) for the sampler's initial unpack
        prev_frames_flat_for_sampler = vis_batch_prev_frames_flat[:n_vis_samples].to(DEVICE) # Shape (B, NumPrevFrames*C, H, W)
        
        num_prev_frames_const = config.NUM_PREV_FRAMES
        img_channels_const = DM_IMG_CHANNELS
        img_h_const = config.IMAGE_SIZE
        img_w_const = config.IMAGE_SIZE
        
        prev_frames_for_sampler_input_5d = prev_frames_flat_for_sampler.view(
            n_vis_samples, 
            num_prev_frames_const, 
            img_channels_const, 
            img_h_const, 
            img_w_const
        )

        # 2. prev_act: (B, NumPrevFrames)
        action_single_for_sampler = vis_batch_act_single[:n_vis_samples].to(DEVICE) # (B, 1)
        action_sequence_for_sampler = action_single_for_sampler.repeat(1, config.NUM_PREV_FRAMES).long() # (B, NumPrevFrames)
        
        with torch.no_grad():
            generated_output = diffusion_sampler.sample(
                prev_obs=prev_frames_for_sampler_input_5d, # Pass the reshaped 5D tensor
                prev_act=action_sequence_for_sampler 
            )
        generated_sample_tensor = generated_output[0] 
        
        ground_truth_for_vis = vis_batch_target_img[:n_vis_samples].to(DEVICE) 
        
        save_visualization_samples(generated_sample_tensor, ground_truth_for_vis, current_epoch_num_for_log, config.SAMPLE_DIR, prefix="val_vis")
        denoiser.train()

    if (current_epoch_num_for_log % PLOT_EVERY == 0) or (epoch == NUM_EPOCHS - 1) or should_stop_early :
        plt.figure(figsize=(12, 6))
        plt.plot(all_train_losses_for_plot, label="Avg Train Loss")
        plt.plot(all_val_losses_for_plot, label="Avg Validation Loss")
        if len(all_train_losses_for_plot) >= TRAIN_MOVING_AVG_WINDOW:
            train_ma_plot = [sum(all_train_losses_for_plot[i-TRAIN_MOVING_AVG_WINDOW+1:i+1])/TRAIN_MOVING_AVG_WINDOW for i in range(TRAIN_MOVING_AVG_WINDOW-1, len(all_train_losses_for_plot))]
            plt.plot(range(TRAIN_MOVING_AVG_WINDOW-1, len(all_train_losses_for_plot)), train_ma_plot, label=f'Train Loss MA ({TRAIN_MOVING_AVG_WINDOW} epochs)', linestyle=':')
        if len(all_val_losses_for_plot) >= VAL_MOVING_AVG_WINDOW:
            val_ma_plot = [sum(all_val_losses_for_plot[i-VAL_MOVING_AVG_WINDOW+1:i+1])/VAL_MOVING_AVG_WINDOW for i in range(VAL_MOVING_AVG_WINDOW-1, len(all_val_losses_for_plot))]
            plt.plot(range(VAL_MOVING_AVG_WINDOW-1, len(all_val_losses_for_plot)), val_ma_plot, label=f'Val Loss MA ({VAL_MOVING_AVG_WINDOW} epochs)', linestyle='--')
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Progress (Epoch {current_epoch_num_for_log})")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(config.PLOT_DIR, f"loss_plot_epoch_{current_epoch_num_for_log:04d}.png"))
        plt.close()
        print(f"Saved loss plot up to epoch {current_epoch_num_for_log}")

overall_training_end_time = time.time()
total_training_duration_seconds = overall_training_end_time - overall_training_start_time
total_training_duration_formatted = str(datetime.timedelta(seconds=total_training_duration_seconds))

# final_epoch_completed is the last epoch index that ran (0-indexed)
print(f"--- Training Complete (Stopped after epoch {final_epoch_completed + 1}) ---") 
print(f"Total training duration: {total_training_duration_formatted}") 

# Final Plot
plt.figure(figsize=(12, 6))
plt.plot(all_train_losses_for_plot, label="Avg Train Loss")
plt.plot(all_val_losses_for_plot, label="Avg Validation Loss")
if len(all_train_losses_for_plot) >= TRAIN_MOVING_AVG_WINDOW:
    train_ma_plot = [sum(all_train_losses_for_plot[i-TRAIN_MOVING_AVG_WINDOW+1:i+1])/TRAIN_MOVING_AVG_WINDOW for i in range(TRAIN_MOVING_AVG_WINDOW-1, len(all_train_losses_for_plot))]
    plt.plot(range(TRAIN_MOVING_AVG_WINDOW-1, len(all_train_losses_for_plot)), train_ma_plot, label=f'Train Loss MA ({TRAIN_MOVING_AVG_WINDOW} epochs)', linestyle=':')
if len(all_val_losses_for_plot) >= VAL_MOVING_AVG_WINDOW:
    val_ma_plot = [sum(all_val_losses_for_plot[i-VAL_MOVING_AVG_WINDOW+1:i+1])/VAL_MOVING_AVG_WINDOW for i in range(VAL_MOVING_AVG_WINDOW-1, len(all_val_losses_for_plot))]
    plt.plot(range(VAL_MOVING_AVG_WINDOW-1, len(all_val_losses_for_plot)), val_ma_plot, label=f'Val Loss MA ({VAL_MOVING_AVG_WINDOW} epochs)', linestyle='--')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Denoiser Final Training & Validation Loss (Up to Epoch {final_epoch_completed + 1})")
plt.legend(); plt.grid(True)
final_loss_plot_path = os.path.join(config.PLOT_DIR, "denoiser_final_loss_plot.png")
plt.savefig(final_loss_plot_path)
# plt.show() # Usually not needed in script, but can be uncommented for interactive
print(f"Final loss plot saved to {final_loss_plot_path}")

