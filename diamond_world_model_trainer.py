#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# 
# 
# get_ipython().system('pip install wandb')

# 
# 
# get_ipython().system('pip install --upgrade typing_extensions')


# In[2]:


import math
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
from dataclasses import dataclass 
from typing import List, Optional, Dict, Any 
import random
from torch.optim.lr_scheduler import LambdaLR
import copy # Make sure to import copy at the top of the file

import wandb # Will be initialized in _main_training

# Your project's specific imports
import config # Your config.py
import models # Your models.py (which should import from diamond_models.ipynb)

# Import dataset from your jetbot_dataset.ipynb
from importnb import Notebook
with Notebook():
    from jetbot_dataset import JetbotDataset, filter_dataset_by_action 

from PIL import Image as PILImage

print("Imports successful.")

# DEVICE will be set in _main_training or used directly from config by other functions
# Global config constants that might be used by imported functions like train_diamond_model
# if they don't re-fetch from config themselves (they mostly do, but being safe).
DM_IMG_CHANNELS = getattr(config, 'DM_IMG_CHANNELS', 3)
DM_NUM_ACTIONS = getattr(config, 'DM_NUM_ACTIONS', 2)


# In[3]:


def split_dataset():
    full_dataset = JetbotDataset(
        csv_path=config.CSV_PATH,
        data_dir=config.DATA_DIR,
        image_size=config.IMAGE_SIZE,
        num_prev_frames=config.NUM_PREV_FRAMES,
        transform=config.TRANSFORM
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

    return train_dataset, val_dataset


# In[4]:


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

def save_visualization_samples(generated_tensor, gt_current_tensor, gt_prev_frames_sequence, epoch, save_dir, prefix="val_vis"):
    """
    Saves a visualization comparing a single generated image, its corresponding GT current image,
    and the sequence of GT previous frames.
    - generated_tensor, gt_current_tensor: [C, H, W]
    - gt_prev_frames_sequence: [NumPrev, C, H, W]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    generated_tensor = generated_tensor.detach().cpu()
    gt_current_tensor = gt_current_tensor.detach().cpu()
    gt_prev_frames_sequence = gt_prev_frames_sequence.detach().cpu()

    num_prev_frames = config.NUM_PREV_FRAMES # Get from global config

    num_cols = num_prev_frames + 1  # N previous frames + 1 current GT
    # Create a 2 rows, num_cols columns subplot
    fig, axs = plt.subplots(2, num_cols, figsize=(num_cols * 3, 6), squeeze=False) # squeeze=False ensures axs is always 2D

    try:
        # Top row: Previous GT frames and Current GT frame
        for i in range(num_prev_frames):
            axs[0, i].imshow(tensor_to_pil(gt_prev_frames_sequence[i]))
            axs[0, i].set_title(f"GT Prev {i+1}")
            axs[0, i].axis('off')
            axs[1, i].axis('off') # Keep bottom row empty under previous GT frames

        axs[0, num_prev_frames].imshow(tensor_to_pil(gt_current_tensor))
        axs[0, num_prev_frames].set_title("GT Current")
        axs[0, num_prev_frames].axis('off')

        # Bottom row, last column: Generated frame (aligned under Current GT)
        axs[1, num_prev_frames].imshow(tensor_to_pil(generated_tensor))
        axs[1, num_prev_frames].set_title("Generated")
        axs[1, num_prev_frames].axis('off')

    except Exception as e:
        print(f"Error visualizing image for prefix {prefix}, epoch {epoch}: {e}")
        # Clear figure and display error text
        for r in range(axs.shape[0]):
            for c in range(axs.shape[1]):
                axs[r,c].axis('off')
        fig.clear() 
        plt.text(0.5, 0.5, "Error displaying image", ha="center", va="center", transform=fig.transFigure)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:04d}.png") 
    plt.savefig(save_path)
    plt.close(fig)
    return save_path
    
def prepare_single_sample_for_sampler(sample_data, device):
    target_img, action_single, prev_frames_flat_unbatched = sample_data # prev_frames_flat_unbatched is [NumPrev*C, H, W]
    
    # Add batch dimension (B=1) and move to device
    gt_current_frame_batch = target_img.unsqueeze(0).to(device) # Shape: [1, C, H, W]
    action_single_batch = action_single.unsqueeze(0).to(device) # Shape: [1, 1]
    # prev_frames_flat_for_sampler_input needs to be [B, NumPrev*C, H, W] for the view later if used directly by sampler
    # but for DIAMOND sampler, prev_obs is [B, NumPrevFrames, C, H, W]
    
    num_prev_frames_const = config.NUM_PREV_FRAMES
    img_channels_const = DM_IMG_CHANNELS # Assumes DM_IMG_CHANNELS is globally available or from config
    img_h_const = config.IMAGE_SIZE
    img_w_const = config.IMAGE_SIZE

    # Reshape prev_frames_flat_unbatched for sampler input [1, NumPrev, C, H, W]
    prev_obs_for_sampler_input_5d = prev_frames_flat_unbatched.view(
        num_prev_frames_const,
        img_channels_const,
        img_h_const,
        img_w_const
    ).unsqueeze(0).to(device) # Add batch dim and send to device

    action_sequence_for_sampler = action_single_batch.repeat(1, config.NUM_PREV_FRAMES).long()
    
    # For visualization, we want the GT previous frames, unbatched and sequenced: [NumPrev, C, H, W]
    gt_prev_frames_seq_for_vis = prev_frames_flat_unbatched.view(
        num_prev_frames_const,
        img_channels_const,
        img_h_const,
        img_w_const
    ) # This is already on CPU if sample_data came directly from dataset before .to(device)
      # It will be detached and moved to CPU again in save_visualization_samples
    
    return prev_obs_for_sampler_input_5d, action_sequence_for_sampler, gt_current_frame_batch, gt_prev_frames_seq_for_vis

print("Visualization helpers defined.")


# In[5]:


def train_denoiser_epoch(
    denoiser_model,
    train_dl,
    opt,
    scheduler,
    grad_clip_val,
    device,
    epoch_num_for_log,
    num_train_batches_total,
    num_val_batches_total,
    train_step_start=0,
):
    """Run one training epoch for the denoiser model."""
    denoiser_model.train()
    total_loss = 0.0

    progress_bar = tqdm(range(len(train_dl)), desc=f"Epoch {epoch_num_for_log} [Train]", leave=False)

    num_prev_frames = config.NUM_PREV_FRAMES
    c, h, w = config.DM_IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE
    accumulation_steps = config.ACCUMULATION_STEPS

    opt.zero_grad()

    perf_table = None
    if wandb.run:
        perf_table = wandb.Table(
            columns=["step", "data_fetch_sec", "batch_prep_sec", "fw_bw_sec", "opt_sched_sec"]
        )

    train_iter = iter(train_dl)

    for batch_idx in progress_bar:
        step_time_start = time.perf_counter()

        # ----- Data fetch -----
        fetch_start = time.perf_counter()
        batch = next(train_iter)
        data_fetch_duration = time.perf_counter() - fetch_start

        # ----- Batch preparation -----
        prep_start = time.perf_counter()
        if isinstance(batch, models.Batch):
            current_batch_obj = batch.to(device)
        else:
            target_img_batch, action_batch, prev_frames_flat_batch = batch
            current_batch_size = target_img_batch.shape[0]
            target_img_batch = target_img_batch.to(device)
            action_batch = action_batch.to(device)
            prev_frames_flat_batch = prev_frames_flat_batch.to(device)

            prev_frames_seq_batch = prev_frames_flat_batch.view(
                current_batch_size, num_prev_frames, c, h, w
            )
            batch_obs_tensor = torch.cat(
                (prev_frames_seq_batch, target_img_batch.unsqueeze(1)), dim=1
            )
            batch_act_tensor = action_batch.repeat(1, num_prev_frames).long()
            batch_mask_padding = torch.ones(
                current_batch_size, num_prev_frames + 1, device=device, dtype=torch.bool
            )

            current_batch_obj = models.Batch(
                obs=batch_obs_tensor,
                act=batch_act_tensor,
                mask_padding=batch_mask_padding,
                info=[{}] * current_batch_size,
            )
        batch_prep_duration = time.perf_counter() - prep_start

        # ----- Forward + backward -----
        fw_bw_start = time.perf_counter()
        loss, logs = denoiser_model(current_batch_obj)
        loss = loss / accumulation_steps
        loss.backward()
        fw_bw_duration = time.perf_counter() - fw_bw_start

        # ----- Optimizer / scheduler -----
        opt_start = time.perf_counter()
        if (batch_idx + 1) % accumulation_steps == 0:
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(denoiser_model.parameters(), grad_clip_val)
            opt.step()
            scheduler.step()
            opt.zero_grad()
        opt_sched_duration = time.perf_counter() - opt_start

        step_duration = time.perf_counter() - step_time_start

        print(f"Batch {batch_idx} of size {len(batch)} took {step_duration} seconds")
        wandb.log({"step_duration": step_duration, "batch_idx": batch_idx})

        if perf_table is not None:
            perf_table.add_data(
                train_step_start + batch_idx,
                data_fetch_duration,
                batch_prep_duration,
                fw_bw_duration,
                opt_sched_duration,
            )

        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({"Loss": loss.item() * accumulation_steps, "LR": scheduler.get_last_lr()[0]})

        if batch_idx % 10 == 0:
            train_step = train_step_start + batch_idx
            wandb.log(
                {
                    "train_batch_loss": loss.item() * accumulation_steps,
                    "train_batch_denoising_loss": logs.get("loss_denoising"),
                    "train_step": train_step,
                }
            )

    if perf_table is not None:
        wandb.log({"train_epoch_perf": perf_table})

    avg_loss = total_loss / len(train_dl) if len(train_dl) > 0 else 0.0
    final_step = train_step_start + len(train_dl)
    return avg_loss, final_step

@torch.no_grad()
def validate_denoiser_epoch(denoiser_model, val_dl, device, epoch_num_for_log, num_train_batches_total, num_val_batches_total, val_step_start=0):
    denoiser_model.eval()
    total_loss = 0.0
    progress_bar = tqdm(val_dl, desc=f"Epoch {epoch_num_for_log} [Valid]", leave=False)
    num_prev_frames = config.NUM_PREV_FRAMES
    c, h, w = config.DM_IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE

    for batch_idx, batch in enumerate(progress_bar):
        if isinstance(batch, models.Batch):
            current_batch_obj = batch.to(device)
        else:
            target_img_batch, action_batch, prev_frames_flat_batch = batch
            current_batch_size = target_img_batch.shape[0]
            target_img_batch = target_img_batch.to(device)
            action_batch = action_batch.to(device)
            prev_frames_flat_batch = prev_frames_flat_batch.to(device)
            prev_frames_seq_batch = prev_frames_flat_batch.view(current_batch_size, num_prev_frames, c, h, w)
            batch_obs_tensor = torch.cat((prev_frames_seq_batch, target_img_batch.unsqueeze(1)), dim=1)
            batch_act_tensor = action_batch.repeat(1, num_prev_frames).long()
            batch_mask_padding = torch.ones(current_batch_size, num_prev_frames + 1, device=device, dtype=torch.bool)
        
            # Corrected Batch instantiation
            current_batch_obj = models.Batch(obs=batch_obs_tensor, act=batch_act_tensor, mask_padding=batch_mask_padding, info=[{}] * current_batch_size)
        
        loss, logs = denoiser_model(current_batch_obj)
        total_loss += loss.item()
        progress_bar.set_postfix({"Val Loss": loss.item()})
        
        # Restored wandb logging
        if batch_idx % 10 == 0:
            val_step = val_step_start + batch_idx
            wandb.log({
                "val_batch_loss": loss.item(),
                "val_batch_denoising_loss": logs.get("loss_denoising"),
                "val_step": val_step,
            })
             
    avg_loss = total_loss / len(val_dl) if len(val_dl) > 0 else 0.0
    final_step = val_step_start + len(val_dl)
    return avg_loss, final_step


print("Training and validation epoch functions adapted for Batch object and Denoiser.forward.")


def train_diamond_model(train_loader, val_loader, fresh_dataset_size, start_checkpoint=None):
    """
    Train a denoiser model with robust, step-based early stopping.
    """
    
    device = config.DEVICE
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True
    # --- Model and Optimizer Setup (remains the same) ---
    inner_cfg = models.InnerModelConfig(
        img_channels=config.DM_IMG_CHANNELS,
        num_steps_conditioning=config.NUM_PREV_FRAMES,
        cond_channels=config.DM_COND_CHANNELS,
        depths=config.DM_UNET_DEPTHS,
        channels=config.DM_UNET_CHANNELS,
        attn_depths=config.DM_UNET_ATTN_DEPTHS,
        num_actions=config.DM_NUM_ACTIONS,
        is_upsampler=config.DM_IS_UPSAMPLER,
    )
    denoiser_cfg = models.DenoiserConfig(
        inner_model=inner_cfg,
        sigma_data=config.DM_SIGMA_DATA,
        sigma_offset_noise=config.DM_SIGMA_OFFSET_NOISE,
        noise_previous_obs=config.DM_NOISE_PREVIOUS_OBS,
        upsampling_factor=config.DM_UPSAMPLING_FACTOR,
    )
    denoiser = models.Denoiser(cfg=denoiser_cfg).to(device)
    sigma_cfg = models.SigmaDistributionConfig(
        loc=config.DM_SIGMA_P_MEAN,
        scale=config.DM_SIGMA_P_STD,
        sigma_min=config.DM_SIGMA_MIN_TRAIN,
        sigma_max=config.DM_SIGMA_MAX_TRAIN,
    )
    denoiser.setup_training(sigma_cfg)

    start_step_offset = -1
    if start_checkpoint and os.path.exists(start_checkpoint):
        state = torch.load(start_checkpoint, map_location=device)
        if 'model_state_dict' in state:
            denoiser.load_state_dict(state['model_state_dict'])
        start_step_offset = state.get('step', -1)
        print(f"Loaded from checkpoint {start_checkpoint}")
    else:
        print("Starting from fresh model")

    opt = torch.optim.AdamW(
        denoiser.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.LEARNING_RATE_WEIGHT_DECAY,
        eps=config.LEARNING_RATE_EPS,
    )

    def lr_lambda(step: int):
        warmup = config.LEARNING_RATE_WARMUP_STEPS
        return float(step) / float(max(1, warmup)) if step < warmup else 1.0
    scheduler = LambdaLR(opt, lr_lambda)
    
    # --- Robust Early Stopping & Checkpointing Setup ---
    best_val_loss = float('inf')
    steps_since_last_improvement = 0
    best_model_state_dict = None

    def round_up_to_multiple(x: int, base:int ) -> int:
        """Smallest multiple of `base` ≥ x."""
        return int(((x + base - 1) // base) * base)

    alpha                 = config.MIX_ALPHA                             # e.g. 0.2
    raw_steps_per_fresh_data_epoch = fresh_dataset_size / (config.BATCH_SIZE * alpha)
    steps_per_fresh_data_epoch = round_up_to_multiple(raw_steps_per_fresh_data_epoch, config.ACCUMULATION_STEPS)

    num_steps = round_up_to_multiple(steps_per_fresh_data_epoch * config.NUM_EPOCHS, config.ACCUMULATION_STEPS)
    patience_steps = round_up_to_multiple(steps_per_fresh_data_epoch * config.EARLY_STOPPING_PATIENCE, config.ACCUMULATION_STEPS)
    
    print(f"Incremental training for at least {patience_steps} and at most {num_steps} with {steps_per_fresh_data_epoch} number of steps per fresh data epoch")

    validate_every = steps_per_fresh_data_epoch    
    # Divergence Guard Setup
    divergence_patience = getattr(config, 'TRAIN_DIVERGE_PATIENCE_CHECKS', 3)
    divergence_threshold = getattr(config, 'TRAIN_DIVERGE_THRESHOLD', 0.05)
    last_train_loss = float('inf')
    divergence_counter = 0

    # --- Training Loop ---
    val_step_count = 0
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(num_steps), desc="Incremental Training Steps")

    # Sampler for visualization (similar to _main_training)
    sampler_cfg_vis = models.DiffusionSamplerConfig(
        num_steps_denoising=config.SAMPLER_NUM_STEPS,
        sigma_min=config.SAMPLER_SIGMA_MIN,
        sigma_max=config.SAMPLER_SIGMA_MAX,
        rho=config.SAMPLER_RHO,
        order=config.SAMPLER_ORDER,
        s_churn=config.SAMPLER_S_CHURN,
        s_tmin=config.SAMPLER_S_TMIN,
        s_tmax=config.SAMPLER_S_TMAX,
        s_noise=config.SAMPLER_S_NOISE
    )
    diffusion_sampler_vis = models.DiffusionSampler(denoiser=denoiser, cfg=sampler_cfg_vis)

    # Prepare filtered validation subsets for visualization (similar to _main_training)
    val_stopped_subset_inc = []
    val_moving_subset_inc = []
    if hasattr(val_loader, 'dataset') and len(val_loader.dataset) > 0:
        val_dataset_for_filter = val_loader.dataset
        val_stopped_subset_inc = filter_dataset_by_action(val_dataset_for_filter, target_actions=0.0)
        moving_action_val_vis_inc = getattr(config, 'MOVING_ACTION_VALUE_FOR_VIS', 0.13)
        val_moving_subset_inc = filter_dataset_by_action(val_dataset_for_filter, target_actions=moving_action_val_vis_inc)
    
    perf_table = None
    if wandb.run:
        perf_table = wandb.Table(
            columns=["step", "data_fetch_sec", "batch_prep_sec", "fw_bw_sec", "opt_sched_sec"]
        )

    for step in pbar:
        step_time_start = time.perf_counter()

        # ----- Data fetch -----
        fetch_start = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        data_fetch_duration = time.perf_counter() - fetch_start

        # ----- Batch preparation -----
        prep_start = time.perf_counter()

        # --- Standard Training Step (remains the same) ---

        # (Code to prepare batch object `current_batch_obj` remains the same)
        if isinstance(batch, models.Batch):
            current_batch_obj = batch.to(device)
        else:
            # Unpack the batch from the DataLoader
            target_img_batch, action_batch, prev_frames_flat_batch = batch

            # Move tensors to the correct device
            target_img_batch = target_img_batch.to(device)
            action_batch = action_batch.to(device)
            prev_frames_flat_batch = prev_frames_flat_batch.to(device)

            # Reconstruct the logic from train_denoiser_epoch to create the Batch object
            current_batch_size = target_img_batch.shape[0]
            num_prev_frames = config.NUM_PREV_FRAMES
            c, h, w = config.DM_IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE

            prev_frames_seq_batch = prev_frames_flat_batch.view(current_batch_size, num_prev_frames, c, h, w)
            batch_obs_tensor = torch.cat((prev_frames_seq_batch, target_img_batch.unsqueeze(1)), dim=1)
            batch_act_tensor = action_batch.repeat(1, num_prev_frames).long()  # Ensure this matches the expected action format for the model
            batch_mask_padding = torch.ones(current_batch_size, num_prev_frames + 1, device=device, dtype=torch.bool)

            current_batch_obj = models.Batch(
                obs=batch_obs_tensor,
                act=batch_act_tensor,
                mask_padding=batch_mask_padding,
                info=[{}] * current_batch_size
            )

        batch_prep_duration = time.perf_counter() - prep_start

        # ----- Forward + backward -----
        fw_bw_start = time.perf_counter()
        denoiser.train()
        loss, logs = denoiser(current_batch_obj)
        train_loss_val = loss.item()
        loss = loss / config.ACCUMULATION_STEPS
        loss.backward()
        fw_bw_duration = time.perf_counter() - fw_bw_start

        # ----- Optimizer / scheduler -----
        opt_start = time.perf_counter()
        if (step + 1) % config.ACCUMULATION_STEPS == 0:
            if config.GRAD_CLIP_VALUE > 0:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), config.GRAD_CLIP_VALUE)
            opt.step()
            scheduler.step()
            opt.zero_grad()
        opt_sched_duration = time.perf_counter() - opt_start

        step_duration = time.perf_counter() - step_time_start

        # Logging to wandb (more frequently for steps)
        train_step_count = start_step_offset + step + 1 # Define train_step_count here
        if wandb.run and (step + 1) % 10 == 0: # Log every 10 steps
            wandb.log({
                "incremental_step_train_loss": train_loss_val,
                "incremental_step_denoising_loss": logs.get("loss_denoising"),
                "incremental_step_learning_rate": scheduler.get_last_lr()[0],
                "incremental_step_data_fetch_sec": data_fetch_duration,
                "incremental_step_duration_sec": step_duration,
                "train_step": train_step_count,
            })

        if perf_table is not None:
            perf_table.add_data(
                step + 1,
                data_fetch_duration,
                batch_prep_duration,
                fw_bw_duration,
                opt_sched_duration,
            )
        
        # --- Validation, Early Stopping, and Divergence Check ---
        if (step + 1) % validate_every == 0 or (step + 1) == num_steps:
            val_step_start = val_step_count
            val_time_start = time.time()
            current_val_loss, val_step_count = validate_denoiser_epoch(
                denoiser, val_loader, device, step + 1, 0, 0, val_step_start=val_step_start
            )
            val_duration = time.time() - val_time_start
            print(f'Validation at step {step+1} took {val_duration:.2f}s')

            # Log validation loss
            if wandb.run:
                wandb.log({
                    "incremental_eval_val_loss": current_val_loss,
                    "incremental_validation_duration_sec": val_duration,
                    "val_step": val_step_start
                })
            
            # Image Sampling (similar to _main_training, simplified for step-based)
            # Tied to validation frequency for now.
            if wandb.run and hasattr(config, 'SAMPLE_EVERY') and (step + 1) % config.SAMPLE_EVERY == 0 :
                sample_time_start = time.time()
                denoiser.eval()
                vis_wandb_log_data_inc = {}
                fixed_sample_idx_inc = getattr(config, 'FIXED_VIS_SAMPLE_IDX', 0)

                if hasattr(val_loader, 'dataset') and fixed_sample_idx_inc < len(val_loader.dataset):
                    fixed_sample_data_inc = val_loader.dataset[fixed_sample_idx_inc]
                    # Ensure sample_data is a tuple (img, act, prev_frames_flat)
                    if not (isinstance(fixed_sample_data_inc, tuple) and len(fixed_sample_data_inc) == 3):
                         # Try to get it from .dataset if val_loader.dataset is a Subset
                        if isinstance(val_loader.dataset, torch.utils.data.Subset):
                            original_dataset = val_loader.dataset.dataset
                            original_idx = val_loader.dataset.indices[fixed_sample_idx_inc]
                            fixed_sample_data_inc = original_dataset[original_idx]
                        else:
                            print(f"Skipping fixed sample visualization: data format error or direct access failed.")
                            fixed_sample_data_inc = None 
                            
                    if fixed_sample_data_inc:
                        prev_obs_fixed_inc, prev_act_fixed_inc, gt_fixed_batch_inc, gt_prev_frames_fixed_seq_inc = prepare_single_sample_for_sampler(fixed_sample_data_inc, device)
                        with torch.no_grad():
                            generated_output_tuple_fixed_inc = diffusion_sampler_vis.sample(prev_obs=prev_obs_fixed_inc, prev_act=prev_act_fixed_inc)
                        if generated_output_tuple_fixed_inc:
                            generated_image_to_save_fixed_inc = generated_output_tuple_fixed_inc[0][0]
                            gt_image_to_save_fixed_inc = gt_fixed_batch_inc[0]
                            vis_path_fixed_inc = save_visualization_samples(
                                generated_image_to_save_fixed_inc, gt_image_to_save_fixed_inc, gt_prev_frames_fixed_seq_inc,
                                step + 1, config.SAMPLE_DIR, prefix=f"inc_vis_fixed_step{step+1}"
                            )
                            vis_wandb_log_data_inc[f"incremental_samples/fixed_idx_{fixed_sample_idx_inc}"] = wandb.Image(vis_path_fixed_inc, caption=f"Step {step+1} Fixed Sample")

                # Simplified: Add one random sample from val_stopped_subset_inc if available
                if len(val_stopped_subset_inc) > 0:
                    stopped_sample_data_inc = val_stopped_subset_inc[random.randint(0, len(val_stopped_subset_inc) - 1)]
                    prev_obs_stop, prev_act_stop, gt_batch_stop, gt_prev_seq_stop = prepare_single_sample_for_sampler(stopped_sample_data_inc, device)
                    with torch.no_grad():
                        gen_out_stop = diffusion_sampler_vis.sample(prev_obs=prev_obs_stop, prev_act=prev_act_stop)
                    if gen_out_stop:
                        vis_path_stop = save_visualization_samples(gen_out_stop[0][0], gt_batch_stop[0], gt_prev_seq_stop, step+1, config.SAMPLE_DIR, prefix=f"inc_vis_stopped_step{step+1}")
                        vis_wandb_log_data_inc["incremental_samples/random_stopped"] = wandb.Image(vis_path_stop, caption=f"Step {step+1} Random Stopped")

                # Simplified: Add one random sample from val_moving_subset_inc_subset_inc if available
                if len(val_moving_subset_inc) > 0:
                    moving_sample_data_inc = val_moving_subset_inc[random.randint(0, len(val_moving_subset_inc) - 1)]
                    prev_obs_move, prev_act_move, gt_batch_move, gt_prev_seq_move = prepare_single_sample_for_sampler(moving_sample_data_inc, device)
                    with torch.no_grad():
                        gen_out_move = diffusion_sampler_vis.sample(prev_obs=prev_obs_move, prev_act=prev_act_move)
                    if gen_out_move:
                        vis_path_move = save_visualization_samples(gen_out_move[0][0], gt_batch_move[0], gt_prev_seq_move, step+1, config.SAMPLE_DIR, prefix=f"inc_vis_moving_step{step+1}")
                        vis_wandb_log_data_inc["incremental_samples/random_moving"] = wandb.Image(vis_path_move, caption=f"Step {step+1} Random Moving")
                
                if vis_wandb_log_data_inc:
                    vis_wandb_log_data_inc["train_step"] = train_step_count # Use the same train_step_count
                    sample_duration = time.time() - sample_time_start
                    print(f'Sampling at step {step+1} took {sample_duration:.2f}s')
                    vis_wandb_log_data_inc["incremental_sampling_duration_sec"] = sample_duration
                    wandb.log(vis_wandb_log_data_inc)
                denoiser.train() # Set back to train mode

            # Check for improvement
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                steps_since_last_improvement = 0
                best_model_state_dict = copy.deepcopy(denoiser.state_dict())
                pbar.set_description(f"New best val_loss: {best_val_loss:.4f}")
            else:
                steps_since_last_improvement += validate_every
                pbar.set_description(f"Best val_loss: {best_val_loss:.4f} Steps since last improvement: {steps_since_last_improvement}")
            
            # Check for training loss divergence
            if train_loss_val > last_train_loss * (1 + divergence_threshold):
                divergence_counter += 1
            else:
                divergence_counter = 0 # Reset if loss is stable
            last_train_loss = train_loss_val

            # Check stopping conditions
            if steps_since_last_improvement >= patience_steps:
                print(f"🛑 Early stopping triggered: No improvement in {patience_steps} steps.")
                if wandb.run: wandb.log({"early_stop_reason": "patience_met", "early_stop_step": step + 1})
                break
            
            if divergence_counter >= divergence_patience:
                print(f"🛑 Early stopping triggered: Training loss diverged for {divergence_patience} checks.")
                if wandb.run: wandb.log({"early_stop_reason": "loss_diverged", "early_stop_step": step + 1})
                break

        pbar.set_postfix({"Train Loss": f"{train_loss_val:.4f}", "Best Val": f"{best_val_loss:.4f}", "Steps w/o Improve": f"{steps_since_last_improvement}", "Fetch": f"{data_fetch_duration:.2f}s", "Step Time": f"{step_duration:.2f}s"})

    pbar.close()

    if perf_table is not None:
        wandb.log({"incremental_perf": perf_table})

    # --- Restore Best Model and Save ---
    if best_model_state_dict:
        print(f"✅ Restoring model to best validation loss: {best_val_loss:.4f}")
        denoiser.load_state_dict(best_model_state_dict)
    
    # Save the final, best model for promotion testing
    final_best_path = os.path.join(config.CHECKPOINT_DIR, "tmp_incremental_best.pth")
    torch.save({"model_state_dict": denoiser.state_dict(), 'step': step + 1, 'val_loss': best_val_loss}, final_best_path)

    return final_best_path


# In[6]:


def _main_training(finetune_checkpoint: str | None = None):
    print("--- Main Training Execution --- ")

    print("--- Configuration ---")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {DEVICE}")

    # Denoiser & InnerModel specific
    DM_SIGMA_DATA = getattr(config, 'DM_SIGMA_DATA', 0.5)
    DM_SIGMA_OFFSET_NOISE = getattr(config, 'DM_SIGMA_OFFSET_NOISE', 0.1)
    DM_NOISE_PREVIOUS_OBS = getattr(config, 'DM_NOISE_PREVIOUS_OBS', True)
    # DM_IMG_CHANNELS is global for prepare_single_sample_for_sampler
    DM_NUM_STEPS_CONDITIONING = getattr(config, 'DM_NUM_STEPS_CONDITIONING', config.NUM_PREV_FRAMES)
    DM_COND_CHANNELS = getattr(config, 'DM_COND_CHANNELS', 256)
    DM_UNET_DEPTHS = getattr(config, 'DM_UNET_DEPTHS', [2, 2, 2, 2])
    DM_UNET_CHANNELS = getattr(config, 'DM_UNET_CHANNELS', [128, 256, 512, 1024])
    DM_UNET_ATTN_DEPTHS = getattr(config, 'DM_UNET_ATTN_DEPTHS', [False, False, True, True])
    # DM_NUM_ACTIONS is global for prepare_single_sample_for_sampler
    DM_IS_UPSAMPLER = getattr(config, 'DM_IS_UPSAMPLER', False)
    DM_UPSAMPLING_FACTOR = getattr(config, 'DM_UPSAMPLING_FACTOR', None)

    # Sampler specific (for inference/visualization)
    SAMPLER_NUM_STEPS = getattr(config, 'SAMPLER_NUM_STEPS', 50)
    SAMPLER_SIGMA_MIN = getattr(config, 'SAMPLER_SIGMA_MIN', 0.002)
    SAMPLER_SIGMA_MAX = getattr(config, 'SAMPLER_SIGMA_MAX', 80.0)
    SAMPLER_RHO = getattr(config, 'SAMPLER_RHO', 7.0)
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
    EARLY_STOPPING_MIN_EPOCHS = getattr(config, 'MIN_EPOCHS', 20)
    EARLY_STOPPING_PERCENTAGE = getattr(config, 'EARLY_STOPPING_PERCENTAGE', 0.1)
    TRAIN_MOVING_AVG_WINDOW = getattr(config, 'TRAIN_MOVING_AVG_WINDOW', 10)
    VAL_MOVING_AVG_WINDOW = getattr(config, 'VAL_MOVING_AVG_WINDOW', 5)
    print("Configuration loaded for _main_training.")

    wandb_config = {
        'DM_SIGMA_DATA': DM_SIGMA_DATA,
        'DM_SIGMA_OFFSET_NOISE': DM_SIGMA_OFFSET_NOISE,
        'DM_NOISE_PREVIOUS_OBS': DM_NOISE_PREVIOUS_OBS,
        'DM_IMG_CHANNELS': DM_IMG_CHANNELS,
        'DM_NUM_STEPS_CONDITIONING': DM_NUM_STEPS_CONDITIONING,
        'DM_COND_CHANNELS': DM_COND_CHANNELS,
        'DM_UNET_DEPTHS': DM_UNET_DEPTHS,
        'DM_UNET_CHANNELS': DM_UNET_CHANNELS,
        'DM_UNET_ATTN_DEPTHS': DM_UNET_ATTN_DEPTHS,
        'DM_NUM_ACTIONS': DM_NUM_ACTIONS,
        'DM_IS_UPSAMPLER': DM_IS_UPSAMPLER,
        'DM_UPSAMPLING_FACTOR': DM_UPSAMPLING_FACTOR,
        'SAMPLER_NUM_STEPS': SAMPLER_NUM_STEPS,
        'SAMPLER_SIGMA_MIN': SAMPLER_SIGMA_MIN,
        'SAMPLER_SIGMA_MAX': SAMPLER_SIGMA_MAX,
        'SAMPLER_RHO': SAMPLER_RHO,
        'SAMPLER_ORDER': SAMPLER_ORDER,
        'SAMPLER_S_CHURN': SAMPLER_S_CHURN,
        'SAMPLER_S_TMIN': SAMPLER_S_TMIN,
        'SAMPLER_S_TMAX': SAMPLER_S_TMAX,
        'SAMPLER_S_NOISE': SAMPLER_S_NOISE,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'GRAD_CLIP_VALUE': GRAD_CLIP_VALUE,
        'DM_SIGMA_P_MEAN': DM_SIGMA_P_MEAN,
        'DM_SIGMA_P_STD': DM_SIGMA_P_STD,
        'DM_SIGMA_MIN_TRAIN': DM_SIGMA_MIN_TRAIN,
        'DM_SIGMA_MAX_TRAIN': DM_SIGMA_MAX_TRAIN,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'EARLY_STOPPING_MIN_EPOCHS': EARLY_STOPPING_MIN_EPOCHS,
        'EARLY_STOPPING_PERCENTAGE': EARLY_STOPPING_PERCENTAGE,
        'TRAIN_MOVING_AVG_WINDOW': TRAIN_MOVING_AVG_WINDOW,
        'VAL_MOVING_AVG_WINDOW': VAL_MOVING_AVG_WINDOW,
        'IMAGE_SIZE': config.IMAGE_SIZE,
        'NUM_PREV_FRAMES': config.NUM_PREV_FRAMES,
        'PROJECT_NAME': getattr(config, 'PROJECT_NAME', 'jetbot-diamond-world-model'),
        'FIXED_VIS_SAMPLE_IDX': getattr(config, 'FIXED_VIS_SAMPLE_IDX', 0),
        'MOVING_ACTION_VALUE_FOR_VIS': getattr(config, 'MOVING_ACTION_VALUE_FOR_VIS', 0.13)
    }
    wandb.init(project=wandb_config['PROJECT_NAME'], config=wandb_config)
    print("Wandb initialized for _main_training.")

    print("--- Initializing Models for _main_training ---")
    try:
        inner_model_config = models.InnerModelConfig(
            img_channels=DM_IMG_CHANNELS,
            num_steps_conditioning=DM_NUM_STEPS_CONDITIONING,
            cond_channels=DM_COND_CHANNELS,
            depths=DM_UNET_DEPTHS,
            channels=DM_UNET_CHANNELS,
            attn_depths=DM_UNET_ATTN_DEPTHS,
            num_actions=DM_NUM_ACTIONS,
            is_upsampler=DM_IS_UPSAMPLER
        )
        # inner_model_instance = models.InnerModel(inner_model_config).to(DEVICE) # Not strictly needed if only denoiser is used
        # print(f"InnerModelImpl parameter count: {sum(p.numel() for p in inner_model_instance.parameters() if p.requires_grad):,}")

        denoiser_cfg = models.DenoiserConfig(
            inner_model=inner_model_config, 
            sigma_data=DM_SIGMA_DATA,
            sigma_offset_noise=DM_SIGMA_OFFSET_NOISE,
            noise_previous_obs=DM_NOISE_PREVIOUS_OBS,
            upsampling_factor=DM_UPSAMPLING_FACTOR
        )
        denoiser = models.Denoiser(cfg=denoiser_cfg).to(DEVICE)
        sigma_dist_train_cfg = models.SigmaDistributionConfig(
            loc=DM_SIGMA_P_MEAN, scale=DM_SIGMA_P_STD,
            sigma_min=DM_SIGMA_MIN_TRAIN, sigma_max=DM_SIGMA_MAX_TRAIN
        )
        denoiser.setup_training(sigma_dist_train_cfg)
        print(f"Denoiser model created for _main_training. Total parameter count: {sum(p.numel() for p in denoiser.parameters() if p.requires_grad):,}")

        sampler_cfg = models.DiffusionSamplerConfig(
            num_steps_denoising=SAMPLER_NUM_STEPS, sigma_min=SAMPLER_SIGMA_MIN,
            sigma_max=SAMPLER_SIGMA_MAX, rho=SAMPLER_RHO, order=SAMPLER_ORDER,
            s_churn=SAMPLER_S_CHURN, s_tmin=SAMPLER_S_TMIN,
            s_tmax=SAMPLER_S_TMAX, s_noise=SAMPLER_S_NOISE
        )
        diffusion_sampler = models.DiffusionSampler(denoiser=denoiser, cfg=sampler_cfg)
        print("DiffusionSampler created for visualization in _main_training.")
    except Exception as e:
        print(f"Error initializing models in _main_training: {e}")
        raise

    print("--- Setting up Optimizer and Scheduler for _main_training ---")
    lr_for_optimizer = LEARNING_RATE
    if finetune_checkpoint:
        print(f"Finetuning from checkpoint: {finetune_checkpoint}")
        if os.path.exists(finetune_checkpoint):
            ckpt = torch.load(finetune_checkpoint, map_location=DEVICE)
            if "model_state_dict" in ckpt:
                denoiser.load_state_dict(ckpt["model_state_dict"])
                lr_for_optimizer = LEARNING_RATE / 5
                print(f"Loaded weights for fine-tuning. LR set to {lr_for_optimizer}")
            else:
                print(f"Warning: no model_state_dict in {finetune_checkpoint}. Starting fresh")
        else:
            print(f"Warning: finetune checkpoint {finetune_checkpoint} not found. Starting fresh")
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=lr_for_optimizer,
        weight_decay=config.LEARNING_RATE_WEIGHT_DECAY, eps=config.LEARNING_RATE_EPS
    )
    print(f"Optimizer: AdamW with LR={lr_for_optimizer}")
    def lr_lambda_main(current_step: int):
        if current_step < config.LEARNING_RATE_WARMUP_STEPS:
            return float(current_step) / float(max(1, config.LEARNING_RATE_WARMUP_STEPS))
        return 1.0
    lr_scheduler = LambdaLR(optimizer, lr_lambda_main)
    print(f"LR Scheduler: LambdaLR with {config.LEARNING_RATE_WARMUP_STEPS} warmup steps.")
    wandb.watch(denoiser, log="all", log_freq=100)
    print("Wandb watching denoiser model.")

    START_EPOCH = 0
    BEST_VAL_LOSS_MA_FROM_CKPT = float('inf')
    PREVIOUS_BEST_VAL_MODEL_PATH = None

    load_path_config_main = None if finetune_checkpoint else config.LOAD_CHECKPOINT
    best_val_loss_model_default_path_main = os.path.join(config.CHECKPOINT_DIR, "denoiser_model_best_val_loss.pth")
    load_path_main = None
    if not finetune_checkpoint:
        load_path_main = load_path_config_main
        if load_path_main:
            print(f"Attempting to load checkpoint from config.LOAD_CHECKPOINT: {load_path_main}")
        elif os.path.exists(best_val_loss_model_default_path_main):
            load_path_main = best_val_loss_model_default_path_main
            print(f"Using existing best_val_loss model: {load_path_main}")

        if load_path_main and os.path.exists(load_path_main):
            print(f"Loading checkpoint for _main_training from: {load_path_main}")
            try:
                checkpoint = torch.load(load_path_main, map_location=DEVICE)
                denoiser.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                START_EPOCH = checkpoint.get("epoch", 0) + 1
                BEST_VAL_LOSS_MA_FROM_CKPT = checkpoint.get("best_val_loss_ma", float("inf"))
                if load_path_main.endswith("denoiser_model_best_val_loss.pth"):
                    PREVIOUS_BEST_VAL_MODEL_PATH = load_path_main
                print(f"Resuming _main_training from epoch {START_EPOCH}.")
            except Exception as e:
                print(f"Error loading checkpoint in _main_training: {e}. Starting fresh.")
                START_EPOCH = 0
        else:
            print("No checkpoint found or specified for _main_training. Starting fresh.")
    train_dataset, val_dataset = split_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    
    val_stopped_subset, val_moving_subset = [], []
    if len(val_dataset) > 0:
        print("Preparing filtered validation subsets for visualization...")
        val_stopped_subset = filter_dataset_by_action(val_dataset, target_actions=0.0)
        moving_action_val = wandb_config['MOVING_ACTION_VALUE_FOR_VIS']
        val_moving_subset = filter_dataset_by_action(val_dataset, target_actions=moving_action_val)
        print(f"Found {len(val_stopped_subset)} stopped and {len(val_moving_subset)} moving samples.")
    else:
        from torch.utils.data import Subset # Ensure Subset is available if val_dataset is empty
        val_stopped_subset = Subset(val_dataset, [])
        val_moving_subset = Subset(val_dataset, [])
    
    print("--- Starting Training Process in _main_training ---")
    overall_training_start_time = time.time()
    all_train_losses_for_plot, all_val_losses_for_plot = [], []
    train_loss_moving_avg_q = deque(maxlen=TRAIN_MOVING_AVG_WINDOW)
    val_loss_moving_avg_q = deque(maxlen=VAL_MOVING_AVG_WINDOW)
    best_val_loss_ma = BEST_VAL_LOSS_MA_FROM_CKPT
    epochs_without_improvement_val = 0
    previous_best_val_model_path = PREVIOUS_BEST_VAL_MODEL_PATH
    final_epoch_completed = START_EPOCH - 1
    num_train_batches = len(train_dataloader)
    num_val_batches = len(val_dataloader)
    
    train_step_count = 0
    val_step_count = 0
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        epoch_start_time = time.time()
        current_epoch_num_for_log = epoch + 1
        avg_train_loss, train_step_count = train_denoiser_epoch(
            denoiser_model=denoiser, train_dl=train_dataloader, opt=optimizer,
            scheduler=lr_scheduler, grad_clip_val=GRAD_CLIP_VALUE, device=DEVICE,
            epoch_num_for_log=current_epoch_num_for_log,
            num_train_batches_total=num_train_batches, num_val_batches_total=num_val_batches,
            train_step_start=train_step_count
        )
        
        all_train_losses_for_plot.append(avg_train_loss)
        train_loss_moving_avg_q.append(avg_train_loss)
        current_train_moving_avg = sum(train_loss_moving_avg_q) / len(train_loss_moving_avg_q) if train_loss_moving_avg_q else float('inf')
    
        val_time_start = time.time()
        avg_val_loss, val_step_count = validate_denoiser_epoch(
            denoiser_model=denoiser, 
            val_dl=val_dataloader, 
            device=DEVICE, 
            epoch_num_for_log=current_epoch_num_for_log,
            num_train_batches_total=num_train_batches, 
            num_val_batches_total=num_val_batches,
            val_step_start=val_step_count      
        )
        val_duration = time.time() - val_time_start
        print(f'Validation for epoch {current_epoch_num_for_log} took {val_duration:.2f}s')
        all_val_losses_for_plot.append(avg_val_loss)
        val_loss_moving_avg_q.append(avg_val_loss) 
        current_val_moving_avg = sum(val_loss_moving_avg_q) / len(val_loss_moving_avg_q) if val_loss_moving_avg_q else float('inf')
    
        epoch_duration_seconds = time.time() - epoch_start_time
        epoch_duration_formatted = str(datetime.timedelta(seconds=epoch_duration_seconds))
    
        print(f"Epoch {current_epoch_num_for_log}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} (MA: {current_train_moving_avg:.4f}), Val Loss: {avg_val_loss:.4f} (MA: {current_val_moving_avg:.4f}), Duration: {epoch_duration_formatted}")
    
        ### WANDB: Log epoch-level metrics ###
        
        wandb_log_data = {
            "epoch": current_epoch_num_for_log,
            "avg_train_loss": avg_train_loss,
            "train_loss_ma": current_train_moving_avg,
            "avg_val_loss": avg_val_loss,
            "val_loss_ma": current_val_moving_avg,
            "best_val_loss_ma_so_far": best_val_loss_ma, # Log best val loss MA so far
            "epoch_duration_sec": epoch_duration_seconds,
            "validation_duration_sec": val_duration,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        if lr_scheduler: lr_scheduler.step(avg_val_loss if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)
    
        # Save model based on Validation Loss MA
        if current_val_moving_avg < best_val_loss_ma:
            improvement_val_over_absolute_best = (best_val_loss_ma - current_val_moving_avg) / abs(best_val_loss_ma + 1e-9) * 100
            print(f"  Val Loss MA improved to {current_val_moving_avg:.6f} from {best_val_loss_ma:.6f} ({improvement_val_over_absolute_best:.2f}% improvement).")
            best_val_loss_ma = current_val_moving_avg
            new_best_val_model_path = os.path.join(config.CHECKPOINT_DIR, "denoiser_model_best_val_loss.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss_ma': best_val_loss_ma
            }, new_best_val_model_path)
            print(f"  Saved new best model (val loss MA) at epoch {current_epoch_num_for_log}")
            if previous_best_val_model_path and previous_best_val_model_path != new_best_val_model_path and os.path.exists(previous_best_val_model_path):
                try:
                    os.remove(previous_best_val_model_path)
                    print(f"  Deleted previous best val model: {previous_best_val_model_path}")
                except OSError as e:
                    print(f"  Warning: Could not delete previous best val model '{previous_best_val_model_path}': {e}")
            previous_best_val_model_path = new_best_val_model_path
    
        should_stop_early = False
        # Early stopping based on validation loss moving average
        if current_epoch_num_for_log > EARLY_STOPPING_MIN_EPOCHS:
            if current_val_moving_avg < best_val_loss_ma:
                epochs_without_improvement_val = 0
            else:
                epochs_without_improvement_val += 1
                print(f"  No improvement in val loss MA for {epochs_without_improvement_val} epoch(s). Best MA: {best_val_loss_ma:.6f}, Current MA: {current_val_moving_avg:.6f}")
                if epochs_without_improvement_val >= EARLY_STOPPING_PATIENCE:
                    should_stop_early = True
                    print("Early stopping triggered due to validation loss stagnation.")
        if (current_epoch_num_for_log % SAVE_MODEL_EVERY == 0) or (epoch == NUM_EPOCHS - 1):
            is_best_this_epoch = current_val_moving_avg == best_val_loss_ma
            # Avoid saving regular checkpoint if it's also the best_val_loss epoch to prevent duplicate saves
            if not (is_best_this_epoch and os.path.join(config.CHECKPOINT_DIR, "denoiser_model_best_val_loss.pth") == previous_best_val_model_path):
                 torch.save({
                    'epoch': epoch, 'model_state_dict': denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss_ma': best_val_loss_ma
                }, os.path.join(config.CHECKPOINT_DIR, f"denoiser_model_epoch_{current_epoch_num_for_log:04d}.pth"))
                 print(f"Saved model checkpoint at epoch {current_epoch_num_for_log}")
        
        final_epoch_completed = epoch # Update last completed epoch here
        if should_stop_early: break
    
        if (current_epoch_num_for_log % SAMPLE_EVERY == 0) or (epoch == NUM_EPOCHS - 1) or should_stop_early:
            print(f"Epoch {current_epoch_num_for_log}: Generating multiple visualization samples...")
            sample_time_start = time.time()
            denoiser.eval()
            vis_wandb_log_data = {} # Accumulate images here for a single wandb.log call
    
            # --- 1. Fixed Sample ---
            fixed_sample_idx = wandb_config.get('FIXED_VIS_SAMPLE_IDX', 0)
            if fixed_sample_idx < len(val_dataset):
                print(f"  Generating fixed sample (index {fixed_sample_idx} from val_dataset)...")
                fixed_sample_data = val_dataset[fixed_sample_idx]
                prev_obs_fixed, prev_act_fixed, gt_fixed_batch, gt_prev_frames_fixed_seq = prepare_single_sample_for_sampler(fixed_sample_data, DEVICE) # gt_fixed_batch is [1,C,H,W]
                with torch.no_grad():
                    generated_output_tuple_fixed = diffusion_sampler.sample(prev_obs=prev_obs_fixed, prev_act=prev_act_fixed)
                
                if generated_output_tuple_fixed:
                    generated_image_batch_fixed = generated_output_tuple_fixed[0] # This is [1, C, H, W]
                    if generated_image_batch_fixed.ndim == 4 and generated_image_batch_fixed.shape[0] == 1:
                        generated_image_to_save_fixed = generated_image_batch_fixed[0] # Extract single image: [C, H, W]
                    else:
                        generated_image_to_save_fixed = generated_image_batch_fixed # Fallback, though should be 4D
        
                    gt_image_to_save_fixed = gt_fixed_batch[0] # Extract single GT image: [C, H, W]
        
                    vis_path_fixed = save_visualization_samples(
                        generated_image_to_save_fixed, # Should be [C,H,W]
                        gt_image_to_save_fixed,        # Should be [C,H,W]
                        gt_prev_frames_fixed_seq,
                        current_epoch_num_for_log,
                        config.SAMPLE_DIR,
                        prefix=f"val_vis_fixed_idx{fixed_sample_idx}"
                    )
                    if vis_path_fixed and wandb.run:
                        vis_wandb_log_data[f"validation_samples/fixed_idx_{fixed_sample_idx}"] = wandb.Image(vis_path_fixed, caption=f"Epoch {current_epoch_num_for_log} Fixed Sample (Val Idx {fixed_sample_idx})")
                else:
                    print("  Warning: Sampler did not return output for fixed sample.")
            else:
                print(f"  Warning: FIXED_SAMPLE_IDX {fixed_sample_idx} is out of bounds for val_dataset (size {len(val_dataset)}). Skipping fixed sample.")
        
            # --- 2. Random Stopped Sample (Action 0.0) ---
            if len(val_stopped_subset) > 0:
                print("  Generating random stopped sample...")
                random_stopped_idx_in_subset = random.randint(0, len(val_stopped_subset) - 1)
                stopped_sample_data = val_stopped_subset[random_stopped_idx_in_subset]
                prev_obs_stopped, prev_act_stopped, gt_stopped_batch, gt_prev_frames_stopped_seq = prepare_single_sample_for_sampler(stopped_sample_data, DEVICE) # gt_stopped_batch is [1,C,H,W]
                with torch.no_grad():
                    generated_output_tuple_stopped = diffusion_sampler.sample(prev_obs=prev_obs_stopped, prev_act=prev_act_stopped)
                
                if generated_output_tuple_stopped:
                    generated_image_batch_stopped = generated_output_tuple_stopped[0] # This is [1, C, H, W]
                    if generated_image_batch_stopped.ndim == 4 and generated_image_batch_stopped.shape[0] == 1:
                        generated_image_to_save_stopped = generated_image_batch_stopped[0] # Extract single image: [C, H, W]
                    else:
                        generated_image_to_save_stopped = generated_image_batch_stopped
        
                    gt_image_to_save_stopped = gt_stopped_batch[0] # Extract single GT image: [C, H, W]
        
                    vis_path_stopped = save_visualization_samples(
                        generated_image_to_save_stopped, # Should be [C,H,W]
                        gt_image_to_save_stopped,        # Should be [C,H,W]
                        gt_prev_frames_stopped_seq,
                        current_epoch_num_for_log,
                        config.SAMPLE_DIR,
                        prefix="val_vis_stopped_random"
                    )
                    if vis_path_stopped and wandb.run:
                        vis_wandb_log_data["validation_samples/random_stopped"] = wandb.Image(vis_path_stopped, caption=f"Epoch {current_epoch_num_for_log} Random Stopped Sample")
                else:
                    print("  Warning: Sampler did not return output for stopped sample.")
            else:
                print("  Warning: No stopped (action 0.0) samples found in validation set. Skipping random stopped sample.")
        
            # --- 3. Random Moving Sample ---
            moving_action_val_vis = wandb_config.get('MOVING_ACTION_VALUE_FOR_VIS', 0.1)
            if len(val_moving_subset) > 0:
                print(f"  Generating random moving sample (action {moving_action_val_vis})...")
                random_moving_idx_in_subset = random.randint(0, len(val_moving_subset) - 1)
                moving_sample_data = val_moving_subset[random_moving_idx_in_subset]
                prev_obs_moving, prev_act_moving, gt_moving_batch, gt_prev_frames_moving_seq = prepare_single_sample_for_sampler(moving_sample_data, DEVICE) # gt_moving_batch is [1,C,H,W]
                with torch.no_grad():
                    generated_output_tuple_moving = diffusion_sampler.sample(prev_obs=prev_obs_moving, prev_act=prev_act_moving)
        
                if generated_output_tuple_moving:
                    generated_image_batch_moving = generated_output_tuple_moving[0] # This is [1, C, H, W]
                    if generated_image_batch_moving.ndim == 4 and generated_image_batch_moving.shape[0] == 1:
                        generated_image_to_save_moving = generated_image_batch_moving[0] # Extract single image: [C, H, W]
                    else:
                        generated_image_to_save_moving = generated_image_batch_moving
        
                    gt_image_to_save_moving = gt_moving_batch[0] # Extract single GT image: [C, H, W]
        
                    vis_path_moving = save_visualization_samples(
                        generated_image_to_save_moving, # Should be [C,H,W]
                        gt_image_to_save_moving,        # Should be [C,H,W]
                        gt_prev_frames_moving_seq,
                        current_epoch_num_for_log,
                        config.SAMPLE_DIR,
                        prefix=f"val_vis_moving_act{str(moving_action_val_vis).replace('.', 'p')}_random"
                    )
                    if vis_path_moving and wandb.run:
                        vis_wandb_log_data[f"validation_samples/random_moving_act{str(moving_action_val_vis).replace('.', 'p')}"] = wandb.Image(vis_path_moving, caption=f"Epoch {current_epoch_num_for_log} Random Moving Sample (Action {moving_action_val_vis})")
                else:
                    print("  Warning: Sampler did not return output for moving sample.")
            else:
                print(f"  Warning: No moving (action {moving_action_val_vis}) samples found in validation set. Skipping random moving sample.")
            
            denoiser.train() # Set model back to training mode
            sample_duration = time.time() - sample_time_start
            print(f'Sampling for epoch {current_epoch_num_for_log} took {sample_duration:.2f}s')
            # Log all accumulated data for this epoch (losses + images)
            if wandb.run:
                wandb.log({**wandb_log_data, **vis_wandb_log_data, 'sampling_duration_sec': sample_duration})
        elif wandb.run: # If not sampling, still log epoch metrics
             wandb.log(wandb_log_data)
    
    
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
            ### WANDB: Log epoch loss plot ###
            wandb.log({"epoch_loss_plot": wandb.Image(plt, caption=f"Loss Plot Epoch {current_epoch_num_for_log}")})
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
    
    ### WANDB: Log final loss plot and finish run ###
    wandb.log({"final_loss_plot": wandb.Image(final_loss_plot_path, caption=f"Final Loss Plot up to Epoch {final_epoch_completed + 1}")})
    wandb.finish()
    print("Wandb run finished.")


# In[7]:


if __name__ == '__main__':
    _main_training()

