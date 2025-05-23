#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install importnb')


# In[2]:


get_ipython().system('pip install scikit-image')


# In[1]:


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import os  # Import os
from importnb import Notebook
import config  # Import your configuration
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Subset, random_split # If not already imported
import pandas as pd # If not already imported
from tqdm.auto import tqdm
import models
import glob

with Notebook():
    from action_conditioned_diffusion_world_model_gemini import forward_diffusion_sample, linear_beta_schedule, cosine_beta_schedule  # Import necessary functions
    from jetbot_dataset import *


# In[4]:


def evaluate_model_quantitative(model, test_dataloader, betas, alphas_cumprod, num_timesteps, device, num_prev_frames, action_tolerance=1e-6):
    """
    Evaluates the model's single-step prediction accuracy using MSE and SSIM,
    correctly handling Subset objects.

    Args:
        model: The trained diffusion model (U-Net).
        test_dataloader: DataLoader for the test set (can wrap Subset).
        betas: Beta values for the diffusion schedule.
        alphas_cumprod: Cumulative product of alphas.
        num_timesteps: Total number of diffusion timesteps.
        device: Device (CPU or GPU).
        num_prev_frames: Number of previous frames used by the model.
        action_tolerance: Tolerance for comparing floating point action values.

    Returns:
        A dictionary containing average metrics (overall and per action).
    """
    model.eval()
    metrics = {
        'overall': {'mse': [], 'ssim': []},
        0.0: {'mse': [], 'ssim': []},
    }

    # --- Get the underlying original JetbotDataset and the indices used by the test_dataloader ---
    subset_dataset = test_dataloader.dataset # This is the direct dataset object from the loader
    original_jetbot_dataset = subset_dataset
    dataloader_indices = list(range(len(subset_dataset))) # Indices 0..N-1 relative to subset_dataset

    # Drill down to find the original dataset if subset_dataset is a Subset
    active_indices = list(range(len(original_jetbot_dataset))) # Start with indices of the current layer
    while isinstance(original_jetbot_dataset, Subset):
        # Map current indices to the parent dataset's indices
        active_indices = [original_jetbot_dataset.indices[i] for i in active_indices]
        original_jetbot_dataset = original_jetbot_dataset.dataset # Go to parent

    # Now, original_jetbot_dataset is the actual JetbotDataset instance
    # and active_indices contains the indices from the original JetbotDataset
    # that are actually used by the test_dataloader (after potential nested Subsets)

    full_dataset_df = original_jetbot_dataset.dataframe # Access dataframe safely

    # Find unique actions within the *active* subset for reporting
    # Map active_indices back to the valid_indices list of the original dataset
    original_valid_indices = original_jetbot_dataset.valid_indices
    actual_df_indices_in_subset = [original_valid_indices[i] for i in active_indices]

    test_actions = full_dataset_df.iloc[actual_df_indices_in_subset]['action'].unique()
    non_zero_actions = [a for a in test_actions if abs(a) > action_tolerance]
    for a in non_zero_actions:
        metrics[a] = {'mse': [], 'ssim': []}
    # --- End Setup ---

    print(f"Starting quantitative evaluation on {len(subset_dataset)} test samples...")

    with torch.no_grad():
        # We iterate using the dataloader, which handles batching (size 1) and indices
        for i, (current_frame_tensor, action_tensor, prev_frames_tensor) in \
                enumerate(tqdm(test_dataloader, desc="Evaluating Test Set")):

            # Data is already loaded and potentially transformed by DataLoader/Dataset
            current_frame_tensor = current_frame_tensor.to(device) # (1, C, H, W)
            action_tensor = action_tensor.to(device)        # (1, 1)
            prev_frames_tensor = prev_frames_tensor.to(device)    # (1, N*C, H, W)

            action_value = action_tensor.item() # Scalar action

            # --- Predict the next frame ---
            t_sample = torch.tensor([num_timesteps - 1], device=device, dtype=torch.long)
            # Use current_frame_tensor directly (it's the output of __getitem__)
            x_noisy, _ = forward_diffusion_sample(current_frame_tensor, t_sample, betas, alphas_cumprod, device)
            x = x_noisy

            for t_val in reversed(range(1, num_timesteps)):
                t = (torch.ones(1) * t_val).long().to(device)
                with torch.cuda.amp.autocast(enabled=config.USE_FP16):
                    predicted_noise = model(x, t, action_tensor, prev_frames_tensor)

                # Ensure alpha, beta, etc. are accessible (defined globally or passed)
                alpha = alphas[t][:, None, None, None]
                alpha_hat = alphas_cumprod[t][:, None, None, None]
                beta = betas[t][:, None, None, None]

                if t_val > 1: noise = torch.randn_like(x)
                else: noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            predicted_next_frame_tensor = x # Range [-1, 1]

            # --- Get the Ground Truth Next Frame ---
            ground_truth_next_frame_tensor = None
            # Get the index *within the current subset* for the current item `i`
            current_subset_index = i
            # Find the corresponding index in the original JetbotDataset's valid_indices list
            current_original_valid_index_position = active_indices[current_subset_index]
            current_actual_df_index = original_jetbot_dataset.valid_indices[current_original_valid_index_position]

            # Find the *next* sequential index in the *original* valid_indices list
            next_original_valid_index_position = -1
            for j in range(len(original_jetbot_dataset.valid_indices)):
                 if original_jetbot_dataset.valid_indices[j] == current_actual_df_index:
                      if j + 1 < len(original_jetbot_dataset.valid_indices):
                           next_original_valid_index_position = j + 1
                      break

            if next_original_valid_index_position != -1:
                 next_actual_df_index = original_jetbot_dataset.valid_indices[next_original_valid_index_position]
                 # Check if the next frame belongs to the SAME session
                 current_session_id = full_dataset_df.iloc[current_actual_df_index]['session_id']
                 next_session_id = full_dataset_df.iloc[next_actual_df_index]['session_id']

                 # Check if they were originally consecutive in the dataframe
                 if current_session_id == next_session_id and next_actual_df_index == current_actual_df_index + 1:
                    # Fetch the ground truth using the original dataset's __getitem__
                    # We need the index relative to the start of valid_indices list
                    ground_truth_next_frame_tensor, _, _ = original_jetbot_dataset[next_original_valid_index_position] # Use the index *within* valid_indices
                    ground_truth_next_frame_tensor = ground_truth_next_frame_tensor.unsqueeze(0).to(device)


            # --- Calculate Metrics (if ground truth is available) ---
            if ground_truth_next_frame_tensor is not None:
                # MSE
                mse = F.mse_loss(predicted_next_frame_tensor, ground_truth_next_frame_tensor).item()

                # SSIM
                pred_unnorm = (predicted_next_frame_tensor.clamp(-1, 1).squeeze(0).cpu().numpy() + 1) / 2
                gt_unnorm = (ground_truth_next_frame_tensor.clamp(-1, 1).squeeze(0).cpu().numpy() + 1) / 2
                pred_unnorm = np.transpose(pred_unnorm, (1, 2, 0))
                gt_unnorm = np.transpose(gt_unnorm, (1, 2, 0))
                try: # Add try-except for SSIM calculation issues
                    ssim_val = ssim(gt_unnorm, pred_unnorm, data_range=1.0, channel_axis=-1, win_size=7) # Use channel_axis for skimage >= 0.19
                except ValueError as e:
                    print(f"Warning: SSIM calculation failed for sample {i}. Error: {e}. Skipping SSIM for this sample.")
                    ssim_val = np.nan # Assign NaN if calculation fails

                # Store metrics if SSIM was successful
                if not np.isnan(ssim_val):
                    metrics['overall']['mse'].append(mse)
                    metrics['overall']['ssim'].append(ssim_val)

                    action_key = 0.0 if abs(action_value) < action_tolerance else action_value
                    # Find closest matching key if not exact match (for float issues)
                    closest_key = action_key
                    if action_key not in metrics:
                         found_key = False
                         for k in metrics.keys():
                              if isinstance(k, float) and abs(action_key - k) < action_tolerance:
                                   closest_key = k
                                   found_key = True
                                   break
                         if not found_key: # If truly unknown action, add it
                              metrics[action_key] = {'mse': [], 'ssim': []}

                    metrics[closest_key]['mse'].append(mse)
                    metrics[closest_key]['ssim'].append(ssim_val)

            # --- End Metric Calculation ---

    # --- Calculate Averages ---
    avg_metrics = {'overall': {}, **{a: {} for a in metrics if a != 'overall'}}
    for action_key, results in metrics.items():
        if results['mse']: # Check if list is not empty
            avg_metrics[action_key]['avg_mse'] = np.nanmean(results['mse']) # Use nanmean
            avg_metrics[action_key]['avg_ssim'] = np.nanmean(results['ssim']) # Use nanmean
            avg_metrics[action_key]['count'] = len(results['mse']) # Count successful calculations
        else:
            avg_metrics[action_key]['avg_mse'] = float('nan')
            avg_metrics[action_key]['avg_ssim'] = float('nan')
            avg_metrics[action_key]['count'] = 0

    # --- Print Results ---
    print("\n--- Quantitative Evaluation Results ---")
    for action_key, results in avg_metrics.items():
         if results['count'] > 0:
              print(f"Action: {action_key} (Count: {results['count']})")
              print(f"  Avg MSE:  {results['avg_mse']:.6f}")
              print(f"  Avg SSIM: {results['avg_ssim']:.4f}")
         else:
              print(f"Action: {action_key} (Count: 0) - No valid data found for evaluation.")
    print("------------------------------------")

    return avg_metrics


# In[5]:


# --- Helper Function for Displaying PIL Images in Subplots ---
def display_frame_pil(pil_frame, ax, title=None):
    """Displays a PIL Image on a matplotlib Axes object."""
    ax.imshow(pil_frame)
    ax.axis('off')
    if title:
        ax.set_title(title)

def test_multistep_prediction(
    model,
    initial_current_tensor,   # Starting frame as preprocessed Tensor
    initial_prev_frames_tensor,# Actual previous frames as preprocessed Tensor
    actions,                  # Tensor of actions for future steps
    betas,
    alphas_cumprod,
    num_timesteps,
    device,
    num_prev_frames,
    image_size,
    output_dir,
    display=True
):
    """
    Tests multi-step prediction using actual initial previous frames.
    Initial PIL image for display is generated internally.

    Args:
        model: Trained diffusion model (U-Net).
        initial_current_tensor: Starting frame (image_t) as preprocessed Tensor (shape [1, C, H, W]).
        initial_prev_frames_tensor: Stack of actual previous frames (image_{t-N+1} to image_{t-1})
                                     as preprocessed Tensor (shape [1, N*C, H, W]).
        actions: Tensor of actions (shape: (n, 1)) for steps t, t+1, ..., t+n-1.
        (Other arguments remain the same)
        display: Whether to display the plot using matplotlib.

    Returns:
        A list of PIL Images representing the predicted frames (step t+1 onwards).
    """

    model.eval()
    num_steps = actions.shape[0]
    predicted_frames = [] # Will store PIL Images

    # --- Use the provided initial tensors ---
    current_frame_tensor = initial_current_tensor.to(device) # Ensure it's on device
    prev_frames_tensor = initial_prev_frames_tensor.to(device) # Ensure it's on device

    # --- Create initial PIL image for display later ---
    initial_frame_pil = transforms.ToPILImage()(((initial_current_tensor[0].clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)).convert("RGB")

    start_time = time.time()

    with torch.no_grad():
        for step in range(num_steps):
            action = actions[step].unsqueeze(0).to(device) # Action for the current step

            # --- Predict frame step+1 based on frame step ---
            t_sample = torch.tensor([num_timesteps - 1], device=device, dtype=torch.long)
            x_noisy, _ = forward_diffusion_sample(current_frame_tensor, t_sample, betas, alphas_cumprod, device)
            x = x_noisy

            for i in reversed(range(1, num_timesteps)):
                t = (torch.ones(1) * i).long().to(device)
                with torch.cuda.amp.autocast(enabled=config.USE_FP16):
                    # Pass the *rolling* previous frames tensor here
                    predicted_noise = model(x, t, action, prev_frames_tensor) # Use the updated prev_frames_tensor

                # (Denoising formula remains the same)
                alpha = alphas[t][:, None, None, None]
                alpha_hat = alphas_cumprod[t][:, None, None, None]
                beta = betas[t][:, None, None, None]
                if i > 1: noise = torch.randn_like(x)
                else: noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            # --- End Denoising loop ---

            # Predicted frame for this step (t+1, t+2 etc.)
            predicted_frame_tensor = x # Range [-1, 1]

            # --- Post-process for storage/display ---
            predicted_frame_tensor_unnorm = (predicted_frame_tensor.clamp(-1, 1) + 1) / 2
            predicted_frame_tensor_uint8 = (predicted_frame_tensor_unnorm * 255).type(torch.uint8)
            predicted_frame_pil = transforms.ToPILImage()(predicted_frame_tensor_uint8[0].cpu()).convert("RGB")
            predicted_frames.append(predicted_frame_pil)

            # --- Update State for the *Next* Prediction Step ---
            current_frame_tensor = predicted_frame_tensor # Use the predicted tensor directly
            # Update the rolling history buffer
            prev_frames_tensor = torch.cat([prev_frames_tensor[:, 3:, :, :], current_frame_tensor], dim=1)


    end_time = time.time()
    total_time = end_time - start_time
    print(f"Generated {num_steps} frames in {total_time:.4f} seconds")
    print(f"Average time per frame: {total_time / num_steps:.4f} seconds")

    # --- Display using Matplotlib ---
    if display:
        num_cols = num_steps + 1 # Initial frame + N predicted steps
        fig, axes = plt.subplots(1, num_cols, figsize=(2 * num_cols, 2.5))

        display_frame_pil(initial_frame_pil, axes[0], title="Initial Frame (t)") # Use internally generated PIL

        for i, frame in enumerate(predicted_frames):
            display_frame_pil(frame, axes[i + 1], title=f"Pred Frame (t+{i+1})\nAction: {actions[i].item():.2f}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return predicted_frames

def filter_dataset_by_action(input_dataset, target_actions, tolerance=1e-6):
    """
    Creates a Subset of a dataset containing only samples with specific actions.

    Args:
        input_dataset (torch.utils.data.Dataset): The dataset or subset to filter
            (e.g., your train_dataset or test_dataset). It assumes the dataset's
            __getitem__ returns (image, action_tensor, prev_frames).
        target_actions (float or list/tuple of float): The action value(s) to keep.
        tolerance (float): Tolerance for floating-point comparison of actions.

    Returns:
        torch.utils.data.Subset: A new subset containing only the samples
                                 with the target action(s). Returns an empty
                                 Subset if no matching samples are found.
    """
    if not isinstance(target_actions, (list, tuple)):
        target_actions = [target_actions] # Ensure it's a list

    print(f"Filtering dataset with {len(input_dataset)} samples for actions: {target_actions}")
    filtered_indices = []
    for i in tqdm(range(len(input_dataset)), desc="Filtering Dataset"):
        try:
            # Access the data point to get the action
            # __getitem__ returns: image, action_tensor, prev_frames
            _, action_tensor, _ = input_dataset[i]
            action_value = action_tensor.item() # Get scalar value

            # Check if the action matches any of the target actions within tolerance
            for target in target_actions:
                if abs(action_value - target) < tolerance:
                    filtered_indices.append(i)
                    break # No need to check other targets for this index
        except Exception as e:
            print(f"Warning: Error processing index {i} during filtering: {e}")
            # Decide whether to skip or raise, depending on expected data integrity
            continue

    if not filtered_indices:
        print(f"Warning: No samples found matching actions {target_actions}.")

    print(f"Filtered down to {len(filtered_indices)} samples.")
    return Subset(input_dataset, filtered_indices)


# In[6]:


# --- Automatic Checkpoint Loading Logic ---
checkpoint_to_load = None
loaded_config_from_checkpoint = None # Store config loaded from checkpoint

# 1. Find potential checkpoint to load (prioritize best)
try:
    # Use config for checkpoint directory
    best_checkpoints = glob.glob(os.path.join(config.CHECKPOINT_DIR, 'model_best_epoch_*.pth'))
    if best_checkpoints:
        best_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        checkpoint_to_load = best_checkpoints[0] # Load the one with highest epoch
        print(f"Found best model checkpoint to load: {checkpoint_to_load}")
    # Fallback to config.LOAD_CHECKPOINT if set *and* no best model found
    elif config.LOAD_CHECKPOINT and os.path.exists(config.LOAD_CHECKPOINT):
         checkpoint_to_load = config.LOAD_CHECKPOINT
         print(f"Using specific checkpoint from config: {checkpoint_to_load}")
    # Fallback to last model if neither best nor specific found
    elif os.path.exists(os.path.join(config.CHECKPOINT_DIR, 'model_last.pth')):
        checkpoint_to_load = os.path.join(config.CHECKPOINT_DIR, 'model_last.pth')
        print(f"Using last saved model checkpoint: {checkpoint_to_load}")
    else:
         print("No suitable checkpoint found ('best', 'last', or specific in config). Cannot run test.")
         # exit() # Or raise error

except Exception as e:
    print(f"Could not search for checkpoints: {e}. Cannot run test.")
    # exit() # Or raise error

# 2. Load checkpoint metadata and instantiate the correct model
model = None
loaded_epoch = 'N/A' # Default if not found
if checkpoint_to_load:
    try:
        print(f"Loading checkpoint: {checkpoint_to_load}")
        checkpoint = torch.load(checkpoint_to_load, map_location=config.DEVICE)

        # --- Get Architecture Info from Checkpoint ---
        if 'config' in checkpoint and 'model_architecture' in checkpoint['config']:
             loaded_config_from_checkpoint = checkpoint['config']
             model_arch_name = loaded_config_from_checkpoint['model_architecture']
             print(f"Checkpoint indicates model architecture: {model_arch_name}")

             # Find the corresponding class in the models module
             if hasattr(models, model_arch_name):
                  model_class = getattr(models, model_arch_name)
                  # Instantiate the model using info from checkpoint config or current config as fallback
                  model = model_class(
                      num_prev_frames=loaded_config_from_checkpoint.get('num_prev_frames', config.NUM_PREV_FRAMES)
                  ).to(config.DEVICE)
                  print(f"Instantiated model: {model_arch_name}")
             else:
                  print(f"ERROR: Model class '{model_arch_name}' not found in models.py!")
                  model = None # Fallback
        else:
             # Fallback if checkpoint lacks architecture info
             print("Warning: Checkpoint missing architecture info. Instantiating based on current config.")
             model_arch_name = config.MODEL_ARCHITECTURE # Get from current config
             if model_arch_name in models.MODEL_REGISTRY: # Use registry if available
                  model = models.MODEL_REGISTRY[model_arch_name](
                       num_prev_frames=config.NUM_PREV_FRAMES
                  ).to(config.DEVICE)
                  print(f"Instantiated model: {model_arch_name}")
             else:
                  print(f"ERROR: MODEL_ARCHITECTURE '{model_arch_name}' in config not found!")
                  model = None

        # --- Load State Dict (if model was instantiated) ---
        if model:
            model.load_state_dict(checkpoint['model_state_dict'])
            # Optimizer state isn't needed for testing
            loaded_epoch = checkpoint.get('epoch', 'N/A') # Get epoch info
            print(f"Successfully loaded model state from epoch {loaded_epoch}.")
            model.eval() # Set to evaluation mode
        else:
             print("Error: Could not instantiate model based on checkpoint.")

    except Exception as e:
         print(f"Error loading checkpoint {checkpoint_to_load}: {e}.")
         model = None # Ensure model is None if loading failed

# Ensure model is loaded before proceeding
if model is None:
    print("Model could not be loaded. Exiting.")
    # exit()
else:
    print(f"Using Model: {model.__class__.__name__}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# In[3]:


model.eval()

# Load diffusion parameters (MUST BE THE SAME AS TRAINING)
if config.SCHEDULE_TYPE == 'linear':
    betas = linear_beta_schedule(config.NUM_TIMESTEPS, config.BETA_START, config.BETA_END).to(config.DEVICE)
elif config.SCHEDULE_TYPE == 'cosine':
    betas = cosine_beta_schedule(config.NUM_TIMESTEPS).to(config.DEVICE)
else:
    raise ValueError(f"Invalid SCHEDULE_TYPE: {config.SCHEDULE_TYPE}")

alphas = (1. - betas).to(config.DEVICE)
alphas_cumprod = torch.cumprod(alphas, axis=0).to(config.DEVICE)



# In[4]:


# --- Create Dataset and DataLoader ---
dataset = JetbotDataset(config.CSV_PATH, config.DATA_DIR, config.IMAGE_SIZE, config.NUM_PREV_FRAMES, transform=config.TRANSFORM)

train_dataset, test_dataset = load_train_test_split(dataset, config.SPLIT_DATASET_FILENAME)


# In[6]:


print(len(dataset))
print(len(train_dataset))
print(len(test_dataset))


# In[9]:


target_action = .1
train_dataset_action_0_1 = filter_dataset_by_action(train_dataset, target_action)


# In[10]:


target_action = .1
test_dataset_action_0_1 = filter_dataset_by_action(test_dataset, target_action)


# In[11]:


target_action = 0
test_dataset_action_0 = filter_dataset_by_action(test_dataset, target_action)


# In[12]:


entry_indx = 395
target_dataset = train_dataset_action_0_1
entry = target_dataset[entry_indx]
current_tensor, _, prev_frames_tensor = entry 


display_dataset_entry(entry)


# In[15]:


# Create a sequence of actions
actions = torch.full((4, 1), 0.1, dtype=torch.float32)

# Call the prediction function
predicted_frames = test_multistep_prediction(
            model,
            current_tensor.unsqueeze(0),  # Add batch dim for initial current tensor
            prev_frames_tensor.unsqueeze(0),# Add batch dim for initial prev tensor
            actions,                  # Future actions
            betas, alphas_cumprod,
            config.NUM_TIMESTEPS, config.DEVICE, config.NUM_PREV_FRAMES, config.IMAGE_SIZE,
            config.OUTPUT_DIR,
            display=True
        )

actual = target_dataset[entry_indx+config.NUM_PREV_FRAMES]
display_dataset_entry(actual)


# In[14]:


subset_idx = 375

# --- Get the index for the original dataset ---
test_dataset_idx = test_dataset_action_0_1.indices[subset_idx]
original_dataset_idx = test_dataset.indices[test_dataset_idx]

print(f"Index {subset_idx} in test_dataset_action_0_1 corresponds to:")
print(f"- Index {test_dataset_idx} in test_dataset")
print(f"- Index {original_dataset_idx} in the original dataset (relative to dataset.valid_indices)")

# You can now access the item directly from the original dataset if needed:
entry_from_original = dataset[original_dataset_idx]
display_dataset_entry(entry_from_original)


# In[ ]:


if test_dataset is None:
     print("Error: Could not load or create test dataset split. Exiting.")
     exit()
else:
     print(f"Using test dataset split with {len(test_dataset)} samples.")

# --- Create Test DataLoader ---
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# --- Run Quantitative Evaluation ---
evaluation_results = evaluate_model_quantitative(
    model, test_dataloader, betas, alphas_cumprod,
    config.NUM_TIMESTEPS, config.DEVICE, config.NUM_PREV_FRAMES
)

# --- Optional: Run Qualitative Multi-step Prediction (from before) ---
print("\n--- Running Qualitative Multi-step Prediction ---")
try:
    # Load a random frame for visualization
    random_idx_for_vis = random.randint(0, len(test_dataset) - 1)
    vis_image, vis_action, vis_prev_frames = test_dataset[random_idx_for_vis]
    initial_frame_pil = transforms.ToPILImage()(((vis_image.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)).convert("RGB")

    # Create a sequence of actions for visualization
    num_vis_steps = 4
    vis_actions = torch.tensor([[0.1]] * num_vis_steps, dtype=torch.float32) # Example: constant 0.1

    predicted_vis_frames = test_multistep_prediction(
        model, initial_frame_pil, vis_actions, betas, alphas_cumprod,
        config.NUM_TIMESTEPS, config.DEVICE, config.NUM_PREV_FRAMES, config.IMAGE_SIZE,
        config.OUTPUT_DIR,
        display=True # Show the plot
    )
except Exception as e:
    print(f"Error during qualitative visualization: {e}")

print("Evaluation Script Finished.")

