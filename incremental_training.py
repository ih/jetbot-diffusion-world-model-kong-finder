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


# In[2]:


import config
with Notebook():
    from jetbot_dataset import JetbotDataset
    from combine_session_data import combine_sessions_append, gather_new_sessions_only
    from compare_diamond_models import load_sampler, evaluate_models_alternating
from diamond_world_model_trainer import train_diamond_model, split_dataset


# In[3]:


import models


# In[4]:


MAX_HOLDOUT = 10
EVAL_SEED = 42


# In[5]:


class ReplayBuffer(Dataset):
    """A simple replay buffer storing dataset indices in memory."""

    def __init__(self, dataset, max_size=50000):
        self.dataset = dataset
        self.max_size = max_size
        # Initialize indices in memory. If dataset is large, take a random subset or the latest `max_size` items.
        # For simplicity, taking the first `max_size` indices, assuming newer data is appended.
        # If dataset can be shorter than max_size, list slicing handles it.
        if len(dataset) > max_size:
            # If you want to prioritize newest data (assuming it's at the end of `dataset` after updates):
            # self.indices = list(range(len(dataset) - max_size, len(dataset)))
            # Or, for random sampling from a large dataset initially:
            # self.indices = random.sample(range(len(dataset)), max_size)
            # Current: take the first max_size, new data added to front via add_episode
            self.indices = list(range(max_size)) 
        else:
            self.indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def sample(self, k):
        idxs = random.sample(self.indices, min(k, len(self.indices)))
        return [self.dataset[i] for i in idxs]


# In[6]:


class MixedDataset(IterableDataset):
    """Yields samples from fresh data with probability ``alpha`` and from the
    replay buffer otherwise."""

    def __init__(self, fresh_ds, replay_buffer, alpha=0.2):
        self.fresh_ds = fresh_ds
        self.replay_buffer = replay_buffer
        self.alpha = alpha

    def __iter__(self):
        while True:
            if random.random() < self.alpha and len(self.fresh_ds) > 0:
                idx = random.randint(0, len(self.fresh_ds) - 1)
                yield self.fresh_ds[idx]
            else:
                yield self.replay_buffer.sample(1)[0]


# In[7]:


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


# In[8]:


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

    # Use split_dataset from diamond_world_model_trainer
    train_ds, val_ds = split_dataset() # train_ds replaces full_ds, val_ds replaces val_dataset

    replay_ds = ReplayBuffer(train_ds, max_size=50000) # Removed index_path

    mixed_dataset = MixedDataset(fresh_ds, replay_ds, alpha=0.2)
    train_loader = DataLoader(
        mixed_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=build_batch,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds, # Use val_ds from split_dataset
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=build_batch,
        pin_memory=True,
    )

    # Step 2: train a new model starting from the last best checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'denoiser_model_best_val_loss.pth')
    print("Starting training")
    new_ckpt = train_diamond_model(
        train_loader,
        val_loader,
        start_checkpoint=ckpt_path,
        max_steps=config.NUM_TRAIN_STEPS,
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
        if results['B']['avg_mse'] < results['A']['avg_mse']:
            os.replace(new_ckpt, ckpt_path)
        else:
            os.remove(new_ckpt)
    else:
        os.replace(new_ckpt, ckpt_path)

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
    print("Combining all session data into the main dataset for future runs...")
    combine_sessions_append(config.SESSION_DATA_DIR, config.IMAGE_DIR, config.CSV_PATH)
    print("Session data combined.")

    # No need to update the current run's replay_ds instance further, as it's ephemeral and will be rebuilt on the next run.

    # Delete the dataset split file to ensure a fresh split on the next run
    split_file_path = os.path.join(config.OUTPUT_DIR, getattr(config, 'SPLIT_DATASET_FILENAME', 'dataset_split.pth'))
    if os.path.exists(split_file_path):
        try:
            os.remove(split_file_path)
            print(f"Deleted dataset split file: {split_file_path}")
        except OSError as e:
            print(f"Error deleting dataset split file {split_file_path}: {e}")


# In[9]:


if __name__ == '__main__':
    main()


# In[ ]:




