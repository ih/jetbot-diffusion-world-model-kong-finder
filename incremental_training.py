#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
from importnb import Notebook
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
import random
import pickle
import numpy as np
import wandb
import time
import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IncrementalTraining")


# In[ ]:


import config
with Notebook():
    from jetbot_dataset import JetbotDataset
    from combine_session_data import combine_sessions_append, gather_new_sessions_only
    from compare_diamond_models import load_sampler, evaluate_models_alternating
from diamond_world_model_trainer import train_diamond_model, split_dataset, log_gpu_memory


# In[ ]:


import models


# In[ ]:


MAX_HOLDOUT = 250
EVAL_SEED = 42
EPS_MSE  = 0.995   # ≥ 0.5 % relative MSE improvement
EPS_SSIM = 0.002   # ≥ 0.002 absolute SSIM gain (~0.2 %)


# In[ ]:


class MixedMapDataset(torch.utils.data.Dataset):
    """
    A high-performance map-style dataset that replaces the slow IterableDataset.

    It pre-calculates a mixed-and-shuffled index mapping from a "fresh"
    and a "replay" dataset based on a desired mixing ratio. This allows
    the PyTorch DataLoader to use its optimized, batched fetching strategy.
    """
    def __init__(self, fresh_ds, replay_ds, virtual_epoch_size, fresh_ratio=0.2):
        super().__init__()
        self.fresh_ds = fresh_ds
        self.replay_ds = replay_ds
        self.virtual_epoch_size = virtual_epoch_size

        num_fresh = int(virtual_epoch_size * fresh_ratio)

        # If there is no fresh data, all samples will come from the replay buffer
        if not self.fresh_ds or len(self.fresh_ds) == 0:
            num_fresh = 0

        num_replay = virtual_epoch_size - num_fresh

        # Create pointers to samples: (is_fresh_bool, index_in_source_dataset)
        # We sample with replacement to fill the desired virtual epoch size
        fresh_pointers = []
        if num_fresh > 0:
            fresh_indices = np.random.choice(len(self.fresh_ds), size=num_fresh, replace=True)
            fresh_pointers = [(True, idx) for idx in fresh_indices]

        replay_indices = np.random.choice(len(self.replay_ds), size=num_replay, replace=True)
        replay_pointers = [(False, idx) for idx in replay_indices]

        # Combine and shuffle the pointers to mix the data
        self.mapping = fresh_pointers + replay_pointers
        np.random.shuffle(self.mapping)

    def __len__(self):
        return self.virtual_epoch_size

    def __getitem__(self, idx):
        is_fresh, source_idx = self.mapping[idx]
        if is_fresh:
            return self.fresh_ds[source_idx]
        else:
            return self.replay_ds[source_idx]


# In[ ]:


def build_batch(samples):
    """Collate function building a ``models.Batch`` from dataset samples."""
    imgs, acts, prevs = zip(*samples)
    imgs  = torch.stack(imgs, 0)
    acts  = torch.stack(acts, 0)
    prevs = torch.stack(prevs, 0)

    b        = len(samples)
    num_prev = config.NUM_PREV_FRAMES
    c, h, w  = config.DM_IMG_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE
    prev_seq = prevs.view(b, num_prev, c, h, w)
    obs      = torch.cat((prev_seq, imgs.unsqueeze(1)), dim=1)
    act_seq  = acts.repeat(1, num_prev).long()
    mask     = torch.ones(b, num_prev + 1, dtype=torch.bool, device=imgs.device)
    return models.Batch(obs=obs, act=act_seq, mask_padding=mask, info=[{}] * b)


# In[ ]:


def main():
    # Step 1: gather only new sessions into a temporary dataset
    gather_new_sessions_only(
        config.SESSION_DATA_DIR,
        config.CSV_PATH,
        config.NEW_IMAGE_DIR,
        config.NEW_CSV_PATH,
    )

    fresh_ds = JetbotDataset(
        config.NEW_CSV_PATH,
        config.NEW_DATA_DIR,
        config.IMAGE_SIZE,
        config.NUM_PREV_FRAMES,
        transform=config.TRANSFORM,
    ) if os.path.exists(config.NEW_CSV_PATH) else []

    print(f"Fresh dataset size is {len(fresh_ds)}.")

    # --- Step 2: Set up datasets (MODIFIED) ---
    print(f"Fresh dataset size is {len(fresh_ds)}.")
    train_ds, val_ds = split_dataset()

    # The original replay buffer can be replaced by the training split directly
    replay_ds = train_ds

    # === START: MODIFICATION TO FIX SLOWDOWN ===

    # 1. Define a "virtual epoch size" based on your training logic
    # This determines how many samples are loaded before the dataloader shuffles again.
    # A larger number is fine; 20,000 is a reasonable default.
    VIRTUAL_EPOCH_SIZE = 20000

    # 2. Create the new high-performance dataset
    mixed_dataset = MixedMapDataset(
        fresh_ds=fresh_ds,
        replay_ds=replay_ds,
        virtual_epoch_size=VIRTUAL_EPOCH_SIZE,
        fresh_ratio=config.MIX_ALPHA  # Use MIX_ALPHA from your config
    )

    # 3. Create the DataLoader. It now uses shuffle=True because it's a map-style dataset.
    train_loader = DataLoader(
        mixed_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=build_batch,
        num_workers=4,
        pin_memory=False,
        shuffle=True,  # This is crucial for map-style datasets
        drop_last=True,
    )

    # === END: MODIFICATION TO FIX SLOWDOWN ===
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        collate_fn=build_batch,
        pin_memory=False,
    )

    # --- Step 3: Train the model (no change here) ---
    new_ckpt = train_diamond_model(
        train_loader,
        val_loader,
        len(fresh_ds),
        start_checkpoint=os.path.join(config.CHECKPOINT_DIR, 'denoiser_model_best_val_loss.pth')
    )
    print("Training Complete")

    # Step 3: compare old best with the newly trained checkpoint
    if os.path.exists(ckpt_path):
        sampler_a = load_sampler(ckpt_path, config.DEVICE)
        sampler_b = load_sampler(new_ckpt, config.DEVICE)
        dataset_holdout = JetbotDataset(
            config.HOLDOUT_CSV_PATH,
            config.HOLDOUT_DATA_DIR,
            config.IMAGE_SIZE,
            config.NUM_PREV_FRAMES,
            transform=config.TRANSFORM,
        )
        if MAX_HOLDOUT and MAX_HOLDOUT < len(dataset_holdout):
            rng = np.random.RandomState(EVAL_SEED)
            subset_idx = rng.choice(len(dataset_holdout), size=MAX_HOLDOUT, replace=False)
            dataset_holdout = Subset(dataset_holdout, subset_idx.tolist())
        dl_holdout = DataLoader(dataset_holdout, batch_size=1, shuffle=False)
        results = evaluate_models_alternating(
            sampler_a, sampler_b, dl_holdout, config.DEVICE, config.NUM_PREV_FRAMES
        )
        promoted = False
        incumbent_stats = {
            'mse': results['A']['avg_mse'],
            'ssim': results['A']['avg_ssim'],
        }
        candidate_stats = {
            'mse': results['B']['avg_mse'],
            'ssim': results['B']['avg_ssim'],
        }
        mse_better = candidate_stats['mse'] < incumbent_stats['mse'] * EPS_MSE
        ssim_better = candidate_stats['ssim'] > incumbent_stats['ssim'] + EPS_SSIM

        if mse_better and ssim_better:
            os.replace(new_ckpt, ckpt_path)
            promoted = True
            logger.info(f"✅ Promoted new model: MSE {incumbent_stats['mse']:.4f} → {candidate_stats['mse']:.4f}, SSIM {incumbent_stats['ssim']:.4f} → {candidate_stats['ssim']:.4f}")
        else:
            os.remove(new_ckpt)
            promoted = False
            logger.info(f"🛑 Rejected: ΔMSE={candidate_stats['mse'] - incumbent_stats['mse']:.4e}, ΔSSIM={candidate_stats['ssim'] - incumbent_stats['ssim']:.4e}")
    else:
        os.replace(new_ckpt, ckpt_path)
        promoted = True

    # After training, permanently add new sessions to the full dataset
    # The ReplayBuffer's dataset (`train_ds`) is a Subset. To correctly update 
    # the underlying full dataset and add new indices, we need to handle this carefully.
    # For now, we'll assume that `combine_sessions_append` updates the source from which `split_dataset` reads.
    # A more robust solution might involve updating the `full_dataset` object used by `split_dataset` 
    # and then re-splitting, or carefully managing indices if `train_ds` is a subset of a global dataset.

    # Assuming `split_dataset` will pick up new data on next run after `combine_sessions_append`
    # The current `replay_ds.dataset` (which is `train_ds`) will not reflect these new sessions until the script is rerun
    # and `split_dataset` is called again. This behavior is kept as is for now.
    # To add *newly collected* data (from `fresh_ds`) to the replay buffer for the *current* training run, that's handled by `MixedDataset`.
    # The logic below primarily concerns adding to the *persistent* dataset for future runs.

    # Ensure all new data (including fresh_ds from this run) is combined into the main persistent dataset.
    # This makes it available for the next execution of incremental_training.ipynb, 
    # where split_dataset will create new train/val splits from the complete data.
    if promoted:
        print("Combining all session data into the main dataset for future runs...")
        combine_sessions_append(config.SESSION_DATA_DIR, config.IMAGE_DIR, config.CSV_PATH)
        print("Session data combined.")

        # Delete the dataset split file to ensure a fresh split on the next run
        split_file_path = os.path.join(config.OUTPUT_DIR, getattr(config, 'SPLIT_DATASET_FILENAME', 'dataset_split.pth'))
        if os.path.exists(split_file_path):
            try:
                os.remove(split_file_path)
                print(f"Deleted dataset split file: {split_file_path}")
            except OSError as e:
                print(f"Error deleting dataset split file {split_file_path}: {e}")
    else:
        print("Skipping session combine since model was not promoted.")

    # No need to update the current run's replay_ds instance further, as it's ephemeral and will be rebuilt on the next run.



# In[ ]:


if __name__ == '__main__':
    wandb.init()
    start_time = time.time()
    main()
    duration = time.time() - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    print(f"Incremental training took : {formatted_duration}")
    wandb.finish()


# In[ ]:





# In[ ]:




