#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import os
from importnb import Notebook
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
import random
import pickle
import numpy as np


# In[3]:


import config
with Notebook():
    from jetbot_dataset import JetbotDataset
    from combine_session_data import combine_sessions_append, gather_new_sessions_only
    from compare_diamond_models import load_sampler, evaluate_models_alternating
from diamond_world_model_trainer import train_diamond_model


# In[3]:


import models


# In[4]:


MAX_HOLDOUT = 10
EVAL_SEED = 42


# In[5]:


class ReplayBuffer(Dataset):
    """A simple replay buffer storing dataset indices."""

    def __init__(self, dataset, max_size=50000, index_path=None):
        self.dataset = dataset
        self.max_size = max_size
        self.index_path = index_path
        if index_path and os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self.indices = pickle.load(f)
        else:
            self.indices = list(range(len(dataset)))[:max_size]
            if index_path:
                with open(index_path, "wb") as f:
                    pickle.dump(self.indices, f)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def sample(self, k):
        idxs = random.sample(self.indices, min(k, len(self.indices)))
        return [self.dataset[i] for i in idxs]

    def add_episode(self, new_idx):
        """Add new indices from a recently processed session."""
        self.indices = list(new_idx) + self.indices
        self.indices = self.indices[: self.max_size]
        if self.index_path:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.indices, f)


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

    full_ds = JetbotDataset(
        config.CSV_PATH,
        config.DATA_DIR,
        config.IMAGE_SIZE,
        config.NUM_PREV_FRAMES,
        transform=config.TRANSFORM,
    )
    replay_ds = ReplayBuffer(full_ds, max_size=50000, index_path=config.REPLAY_INDEX_PATH)

    mixed_dataset = MixedDataset(fresh_ds, replay_ds, alpha=0.2)
    train_loader = DataLoader(
        mixed_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=build_batch,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = JetbotDataset(
        config.HOLDOUT_CSV_PATH,
        config.HOLDOUT_DATA_DIR,
        config.IMAGE_SIZE,
        config.NUM_PREV_FRAMES,
        transform=config.TRANSFORM,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=build_batch,
        num_workers=4,
        pin_memory=True,
    )

    # Step 2: train a new model starting from the last best checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'denoiser_model_best_val_loss.pth')
    new_ckpt = train_diamond_model(
        train_loader,
        val_loader,
        start_checkpoint=ckpt_path,
        max_steps=config.NUM_TRAIN_STEPS,
    )

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
    old_len = len(full_ds)
    combine_sessions_append(config.SESSION_DATA_DIR, config.IMAGE_DIR, config.CSV_PATH)
    updated_ds = JetbotDataset(config.CSV_PATH, config.DATA_DIR, config.IMAGE_SIZE, config.NUM_PREV_FRAMES, transform=config.TRANSFORM)
    new_indices = range(old_len, len(updated_ds))
    replay_ds.dataset = updated_ds
    replay_ds.add_episode(new_indices)


# In[ ]:


if __name__ == '__main__':
    main()


# In[ ]:




