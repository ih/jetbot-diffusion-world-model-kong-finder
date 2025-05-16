#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
import datetime
import csv
import config
from torch.utils.data import random_split
import models
from importnb import Notebook
with Notebook():
    from jetbot_dataset import *


# In[2]:


print(f"📏  Training at {config.TARGET_HZ} Hz  "
      f"(keeping every {config.FRAME_STRIDE}ᵗʰ frame from 30 Hz logs)")


# In[3]:


# --- Diffusion Helpers ---
def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, betas, alphas_cumprod, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(torch.sqrt(alphas_cumprod), t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        torch.sqrt(1. - alphas_cumprod), t, x_0.shape
    )
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# --- U-Net Model ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Ensure output matches the embedding dim even if input dim is odd
        if self.dim % 2 == 1:
             embeddings = F.pad(embeddings, (0, 1))
        return embeddings

# --- Training Loop ---
def train(model, dataloader, optimizer, betas, alphas_cumprod, start_epoch, num_epochs,
          device, save_every, sample_every, checkpoint_dir, sample_dir, plot_dir,
          plot_every, use_fp16, accumulation_steps, num_prev_frames,
          early_stopping_patience, early_stopping_percentage, min_epochs):
    """
    Trains the diffusion model with early stopping and best model saving/deletion.
    """

    all_losses = []
    start_time = time.time()
    last_plot_epoch = start_epoch - 1
    best_loss = float('inf')
    best_epoch = start_epoch
    epochs_without_improvement = 0
    moving_avg_window = 10
    moving_avg_losses = []
    previous_best_model_path = None  # Keep track of the previous best model's path

    # --- Load previous best model path if resuming ---
    # Find the highest epoch 'best' model if resuming, to delete the correct one later
    if start_epoch > 0:
        try:
            existing_best = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_best_epoch_')]
            if existing_best:
                existing_best.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                previous_best_model_path = os.path.join(checkpoint_dir, existing_best[0])
                # Extract best_loss from the loaded best model checkpoint if desired
                # best_checkpoint = torch.load(previous_best_model_path, map_location='cpu')
                # best_loss = best_checkpoint.get('loss', float('inf')) # Restore best loss
                print(f"Found previous best model: {previous_best_model_path}") # Verify best loss restoration if needed
        except Exception as e:
            print(f"Warning: Could not determine previous best model path: {e}")
    # --------------------------------------------------
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, (images, actions, prev_frames) in enumerate(pbar):
            images = images.to(device)
            actions = actions.to(device)
            prev_frames = prev_frames.to(device)
            t = torch.randint(0, config.NUM_TIMESTEPS, (images.shape[0],), device=device).long()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                x_noisy, noise = forward_diffusion_sample(images, t, betas, alphas_cumprod, device)
                predicted_noise = model(x_noisy, t, actions, prev_frames)
                loss = F.mse_loss(noise, predicted_noise)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_losses.append(loss.item() * accumulation_steps)
            pbar.set_postfix({"Loss": loss.item() * accumulation_steps})

        if optimizer.param_groups[0]['params'][0].grad is not None:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)  if epoch_losses else float('nan') # Handle empty epoch_losses
        if np.isnan(avg_epoch_loss):
            print(f"Warning: NaN loss detected for epoch {epoch+1}. Skipping update/plot.")
            # Optionally: break or handle NaN case differently
            continue
        all_losses.append(avg_epoch_loss)

        moving_avg_losses.append(avg_epoch_loss)
        if len(moving_avg_losses) > moving_avg_window:
            moving_avg_losses.pop(0)
        current_moving_avg = sum(moving_avg_losses) / len(moving_avg_losses)

        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': { # Store relevant config settings used for *this* checkpoint
                      'model_architecture': model.__class__.__name__, # Save the class name
                      'image_size': config.IMAGE_SIZE,
                      'num_prev_frames': config.NUM_PREV_FRAMES
                      # Add other relevant hyperparameters like channel dimensions if needed
                  }
            }, os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
            print(f"Saved model checkpoint at epoch {epoch+1}")

        if (epoch + 1) % sample_every == 0:
            model.eval()
            with torch.no_grad():
                random_idx = torch.randint(0, len(dataset), (1,)).item()
                real_current_frame, action, real_prev_frames = dataset[random_idx]
                real_current_frame = real_current_frame.unsqueeze(0).to(device)
                real_prev_frames = real_prev_frames.unsqueeze(0).to(device)
                action = action.to(device)

                t_sample = torch.tensor([config.NUM_TIMESTEPS - 1], device=device, dtype=torch.long)
                x_noisy, _ = forward_diffusion_sample(real_current_frame, t_sample, betas, alphas_cumprod, device)
                x = x_noisy

                for i in reversed(range(1, config.NUM_TIMESTEPS)):
                    t = (torch.ones(1) * i).long().to(device)
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        predicted_noise = model(x, t, action, real_prev_frames)

                    alpha = alphas[t][:, None, None, None]
                    alpha_hat = alphas_cumprod[t][:, None, None, None]
                    beta = betas[t][:, None, None, None]

                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                predicted_next_frame = (x.clamp(-1, 1) + 1) / 2
                predicted_next_frame = (predicted_next_frame * 255).type(torch.uint8)
                prev_images = []

                for i in range(num_prev_frames):
                    frame = real_prev_frames[0, (i * 3):(i + 1) * 3, :, :]
                    frame = (frame.clamp(-1, 1) + 1) / 2
                    frame = (frame * 255).type(torch.uint8)
                    prev_images.append(transforms.ToPILImage()(frame))

                current_tensor = (real_current_frame[0].clamp(-1, 1) + 1) / 2 * 255
                current_image = transforms.ToPILImage()(current_tensor.type(torch.uint8)).convert("RGB")
                predicted_image = transforms.ToPILImage()(predicted_next_frame[0]).convert("RGB")

                total_width = (num_prev_frames + 2) * config.IMAGE_SIZE
                max_height = config.IMAGE_SIZE
                new_im = Image.new('RGB', (total_width, max_height))

                x_offset = 0
                for image in prev_images:
                    new_im.paste(image, (x_offset,0))
                    x_offset += config.IMAGE_SIZE
                new_im.paste(current_image, (x_offset, 0))
                x_offset += config.IMAGE_SIZE
                new_im.paste(predicted_image, (x_offset, 0))

                new_im.save(os.path.join(sample_dir, f"sample_epoch_{epoch+1}.png"))
                print(f"Saved sample image at epoch {epoch+1}")

            model.train()

            print(f"Epoch {epoch+1}, Step {step}:")
            print(f"  Mem Allocated: {torch.cuda.memory_allocated(config.DEVICE) / 1024**2:.2f} MB")
            print(f"  Max Mem Allocated: {torch.cuda.max_memory_allocated(config.DEVICE) / 1024**2:.2f} MB")
            print(f"  Mem Reserved: {torch.cuda.memory_reserved(config.DEVICE) / 1024**2:.2f} MB")
            print(f"  Max Mem Reserved: {torch.cuda.max_memory_reserved(config.DEVICE) / 1024**2:.2f} MB")
        
        
        if (epoch + 1) % plot_every == 0:
            elapsed_time = time.time() - start_time
            formatted_time = str(datetime.timedelta(seconds=elapsed_time))
    
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
            # --- Plot 1: Loss from START_EPOCH of this run ---
            # X-axis: Absolute epoch numbers (start_epoch + 1 up to current epoch + 1)
            # Y-axis: Losses collected *in this run* (all_losses indices 0 up to current)
            current_run_epochs_plotted = range(start_epoch + 1, epoch + 2)
            axes[0].plot(current_run_epochs_plotted, all_losses)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title(f"Loss Since Start (Epoch {start_epoch+1}, Time: {formatted_time})")
            axes[0].grid(True)
    
            # --- Plot 2: Loss since last plot ---
            # X-axis: Absolute epoch numbers for the segment
            x_values_ax1 = range(last_plot_epoch + 2, epoch + 2)
    
            # Y-axis: Slice all_losses using indices relative to this run's start
            # Calculate indices corresponding to the absolute epoch numbers
            start_slice_index = (last_plot_epoch + 1) - start_epoch # Index in all_losses for epoch last_plot_epoch+1
            end_slice_index = (epoch + 1) - start_epoch           # Index in all_losses for epoch epoch+1 (exclusive)
            y_values_ax1 = all_losses[start_slice_index : end_slice_index]
    
            if x_values_ax1 and y_values_ax1:
                if len(x_values_ax1) != len(y_values_ax1):
                     # This check should ideally not be needed with correct logic, but good safeguard
                     print(f"!!! ERROR: Mismatch detected plotting axes[1]: len(x)={len(x_values_ax1)}, len(y)={len(y_values_ax1)}")
                else:
                    axes[1].plot(x_values_ax1, y_values_ax1)
                    axes[1].set_xlabel("Epoch")
                    axes[1].set_ylabel("Loss")
                    axes[1].set_title(f"Loss Since Epoch {last_plot_epoch + 1}")
                    axes[1].grid(True)
            else:
                 axes[1].set_title(f"Loss Since Epoch {last_plot_epoch + 1} (No new data)")
                 axes[1].grid(True)
    
    
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"loss_plot_epoch_{epoch+1}.png"))
            plt.close()
            print(f"Epoch {epoch+1}: Avg Loss = {avg_epoch_loss:.6f}, Moving Avg = {current_moving_avg:.6f}, Time = {formatted_time}")
    
            last_plot_epoch = epoch # Update absolute last plot epoch index

        # --- Early Stopping (Dynamic Threshold) and Best Model Saving/Deletion---
        if early_stopping_patience is not None and epoch + 1 > min_epochs:
            should_stop = False
            improvement_calculated = False
            calculated_improvement = None # Define outside the inner ifs

            if current_moving_avg < best_loss:
                best_loss = current_moving_avg
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                # Save the *best* model
                new_best_model_path = os.path.join(checkpoint_dir, f"model_best_epoch_{best_epoch}.pth")
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss, # Save the best moving average loss
                    'config': { # Store relevant config settings used for *this* checkpoint
                          'model_architecture': model.__class__.__name__, # Save the class name
                          'image_size': config.IMAGE_SIZE,
                          'num_prev_frames': config.NUM_PREV_FRAMES
                          # Add other relevant hyperparameters like channel dimensions if needed
                      }
                }, new_best_model_path)
                print(f"Saved best model at epoch {best_epoch} (Moving Avg Loss: {best_loss:.6f})")

                # Delete the *previous* best model
                if previous_best_model_path and os.path.exists(previous_best_model_path):
                    try: # Add try-except for deletion
                        os.remove(previous_best_model_path)
                        print(f"Deleted previous best model: {previous_best_model_path}")
                    except OSError as e:
                        print(f"Warning: Could not delete previous best model '{previous_best_model_path}': {e}")
                previous_best_model_path = new_best_model_path

            else:
                epochs_without_improvement += 1

                # Check if patience is exceeded
                if epochs_without_improvement >= early_stopping_patience:
                    # Try to calculate improvement percentage ONLY if patience is met
                    if len(moving_avg_losses) >= moving_avg_window: # Use >= for window check
                        # Calculate previous average using the window *before* the non-improvement streak started
                        # Ensure indices are valid
                        if len(all_losses) > epochs_without_improvement:
                            # Index of the loss just before the non-improvement streak started
                            comparison_idx = len(all_losses) - epochs_without_improvement - 1
                            # Average of the window ending at that point
                            start_comparison_window = max(0, comparison_idx - moving_avg_window + 1)
                            prev_window_losses = all_losses[start_comparison_window : comparison_idx + 1]

                            if prev_window_losses:
                                prev_moving_avg = sum(prev_window_losses) / len(prev_window_losses)
                                if prev_moving_avg > 1e-9: # Check for non-zero denominator
                                    calculated_improvement = (prev_moving_avg - current_moving_avg) / prev_moving_avg * 100
                                    improvement_calculated = True
                                    print(f"  Epochs w/o improvement: {epochs_without_improvement}, Current MA: {current_moving_avg:.6f}, Prev MA: {prev_moving_avg:.6f}, Improvement: {calculated_improvement:.2f}%")
                                else:
                                    print(f"  Epochs w/o improvement: {epochs_without_improvement}. Previous MA near zero.")
                            else:
                                print(f"  Epochs w/o improvement: {epochs_without_improvement}. Not enough history for prev MA.")
                        else:
                            print(f"  Epochs w/o improvement: {epochs_without_improvement}. Not enough history for prev MA.")


                    else:
                        print(f"  Epochs w/o improvement: {epochs_without_improvement}. Waiting for full moving avg window to check percentage.")

                    # Decide whether to stop based on calculated improvement (if available)
                    # Stop if patience is met AND (improvement wasn't calculated OR improvement is too small)
                    if not improvement_calculated or (improvement_calculated and calculated_improvement < early_stopping_percentage):
                         should_stop = True
                         stop_reason = f"Improvement {calculated_improvement:.2f}% < {early_stopping_percentage}%" if improvement_calculated else "Patience reached, improvement could not be reliably calculated."
                         print(f"Early stopping triggered at epoch {epoch+1}. Reason: {stop_reason}")


            # Break the loop *outside* the nested conditions if stop flag is set
            if should_stop:
                break
                
    end_time = time.time()
    total_time = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=total_time))
    print(f"Total training time: {formatted_time}")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
        'config': { # Store relevant config settings used for *this* checkpoint
              'model_architecture': model.__class__.__name__, # Save the class name
              'image_size': config.IMAGE_SIZE,
              'num_prev_frames': config.NUM_PREV_FRAMES
              # Add other relevant hyperparameters like channel dimensions if needed
          }
    }, os.path.join(checkpoint_dir, "model_last.pth"))
    print(f"Saved last model at epoch {epoch+1} with loss {avg_epoch_loss}")

    return all_losses
    


# In[4]:


if __name__ == "__main__":
    # --- Data Transforms ---
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # --- Create Dataset and DataLoader ---
    dataset = JetbotDataset(config.CSV_PATH, config.DATA_DIR, config.IMAGE_SIZE, config.NUM_PREV_FRAMES, transform=transform)
    
    # Try to load existing split
    train_dataset, test_dataset = load_train_test_split(dataset, config.SPLIT_DATASET_FILENAME)
    
    if train_dataset is None or test_dataset is None:
        print("Dataset split file not found, creating a new split...")
        train_dataset, test_dataset = split_train_test_by_session_id(dataset)
    
        save_existing_split(train_dataset, test_dataset, config.SPLIT_DATASET_FILENAME)
    else:
        print("Loaded existing dataset split.")
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size 1 for easier evaluation
    
    # --- Calculate Betas and Alphas ---
    betas = linear_beta_schedule(config.NUM_TIMESTEPS, config.BETA_START, config.BETA_END).to(config.DEVICE)
    #betas = cosine_beta_schedule(NUM_TIMESTEPS).to(DEVICE) # Alternative
    
    alphas = (1. - betas).to(config.DEVICE)
    alphas_cumprod = torch.cumprod(alphas, axis=0).to(config.DEVICE)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(config.DEVICE)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(config.DEVICE)
        
    # --- Automatic Checkpoint Loading Logic ---
    START_EPOCH = 0
    checkpoint_to_load = None
    loaded_config = None # Store config from checkpoint if loaded

    # 1. Find potential checkpoint to load (prioritize best)
    try:
        best_checkpoints = glob.glob(os.path.join(config.CHECKPOINT_DIR, 'model_best_epoch_*.pth'))
        if best_checkpoints:
            best_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
            checkpoint_to_load = best_checkpoints[0]
            print(f"Found best model checkpoint to load: {checkpoint_to_load}")
        elif config.LOAD_CHECKPOINT and os.path.exists(config.LOAD_CHECKPOINT):
             checkpoint_to_load = config.LOAD_CHECKPOINT
             print(f"Using specific checkpoint from config: {checkpoint_to_load}")
        else:
             print("No best model found and no specific checkpoint set. Starting training from scratch.")
    except Exception as e:
        print(f"Could not search for checkpoints: {e}. Starting training from scratch.")


    # 2. Load checkpoint metadata and instantiate the correct model
    model = None
    if checkpoint_to_load:
        try:
            print(f"Loading checkpoint: {checkpoint_to_load}")
            # Load the entire checkpoint dictionary first
            checkpoint = torch.load(checkpoint_to_load, map_location=config.DEVICE)

            # --- Get Architecture Info from Checkpoint ---
            if 'config' in checkpoint and 'model_architecture' in checkpoint['config']:
                 loaded_config = checkpoint['config']
                 model_arch_name = loaded_config['model_architecture']
                 print(f"Checkpoint indicates model architecture: {model_arch_name}")

                 # Find the corresponding class in the models module
                 if hasattr(models, model_arch_name):
                      model_class = getattr(models, model_arch_name)
                      # Instantiate the model using info potentially from checkpoint config
                      # (or assume current config matches if only class name is stored)
                      model = model_class(
                          num_prev_frames=loaded_config.get('num_prev_frames', config.NUM_PREV_FRAMES) # Use loaded if available
                      ).to(config.DEVICE)
                      print(f"Instantiated model: {model_arch_name}")
                 else:
                      print(f"ERROR: Model class '{model_arch_name}' not found in models.py!")
                      model = None # Fallback
            else:
                 print("Warning: Checkpoint missing architecture info. Assuming current config matches.")
                 # Fallback: Instantiate model based on current config if checkpoint lacks info
                 model = models.get_model(config).to(config.DEVICE) # Use factory if available
                 # Or directly: model = models.MODEL_REGISTRY[config.MODEL_ARCHITECTURE](...).to(config.DEVICE)


            # --- Load State Dicts (if model was instantiated) ---
            if model:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Instantiate optimizer *after* model is on device
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                START_EPOCH = checkpoint['epoch']
                print(f"Successfully loaded states. Resuming from epoch {START_EPOCH}")
            else:
                 # Handle error: Could not instantiate model from checkpoint info
                 print("Error: Could not instantiate model based on checkpoint. Starting from scratch.")
                 START_EPOCH = 0

        except Exception as e:
             print(f"Error loading checkpoint {checkpoint_to_load}: {e}. Starting from scratch.")
             START_EPOCH = 0

    # 3. If no checkpoint loaded, instantiate model and optimizer from current config
    if model is None: # If not loaded above
         print("Instantiating new model based on current config.")
         # model = models.get_model(config).to(config.DEVICE) # Use factory
         # Or directly:
         if config.MODEL_ARCHITECTURE in models.MODEL_REGISTRY:
             model = models.MODEL_REGISTRY[config.MODEL_ARCHITECTURE](
                  num_prev_frames=config.NUM_PREV_FRAMES
             ).to(config.DEVICE)
         else:
              raise ValueError(f"MODEL_ARCHITECTURE '{config.MODEL_ARCHITECTURE}' in config not found in models.py")

         optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
         START_EPOCH = 0

    # --- Print parameter count for confirmation ---
    print(f"Using Model: {model.__class__.__name__}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    print(f"--- Training Configuration ---")
    # Print model parameter count
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Mem Allocated: {torch.cuda.memory_allocated(config.DEVICE) / 1024**2:.2f} MB")
    print(f"  Max Mem Allocated: {torch.cuda.max_memory_allocated(config.DEVICE) / 1024**2:.2f} MB")
    print(f"  Mem Reserved: {torch.cuda.memory_reserved(config.DEVICE) / 1024**2:.2f} MB")
    print(f"  Max Mem Reserved: {torch.cuda.max_memory_reserved(config.DEVICE) / 1024**2:.2f} MB")    
    print(f"--------------------------")    
    # --- Train the Model ---
    losses = train(model, train_dataloader, optimizer, betas, alphas_cumprod, START_EPOCH, config.NUM_EPOCHS, config.DEVICE,
                   config.SAVE_MODEL_EVERY, config.SAMPLE_EVERY, config.CHECKPOINT_DIR, config.SAMPLE_DIR, config.PLOT_DIR, config.PLOT_EVERY, config.USE_FP16,
                   config.ACCUMULATION_STEPS, config.NUM_PREV_FRAMES, early_stopping_patience=config.EARLY_STOPPING_PATIENCE, early_stopping_percentage=config.EARLY_STOPPING_PERCENTAGE, min_epochs=config.MIN_EPOCHS)
    print("Training complete!")
    
    # --- Final Loss Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(config.PLOT_DIR, "loss_plot_final.png"))  # Save to plot dir
    plt.close()


# In[7]:


config.SAMPLE_EVERY


# In[44]:





# In[ ]:




