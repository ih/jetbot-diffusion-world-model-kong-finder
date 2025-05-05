#!/usr/bin/env python
# coding: utf-8

# # Reward Estimator Training – **ResNet‑18 + Multi‑Frame + Reward Labels**  (v4)
# Now with live **progress bars** and a quick sanity check that the GPU is actually used.
# *If your MSI Afterburner still shows 0 % GPU, scroll to the first code cell – it prints what Torch thinks the current device is and whether CUDA is available.*

# In[1]:


import os, random, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
import config
from models import RewardEstimatorResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, '| CUDA visible →', torch.cuda.is_available())


# In[2]:


# ---------------- Hyper‑parameters / paths ----------------
# CHECKPOINT_DIR = Path("outputs/reward_estimator_resnet")
# CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REWARD_MODEL_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "reward_estimator")
os.makedirs(REWARD_MODEL_OUTPUT_DIR, exist_ok=True)

REWARD_CSV_PATH = config.MANUAL_COLLECTED_REWARD_CSV   # labels CSV
MAIN_CSV_PATH   = config.CSV_PATH                      # master frames CSV
MAIN_DATA_DIR   = config.DATA_DIR                      # image folder

# print("CHECKPOINT_DIR           →", CHECKPOINT_DIR.resolve())
print("REWARD_MODEL_OUTPUT_DIR  →", os.path.abspath(REWARD_MODEL_OUTPUT_DIR))
print("REWARD_CSV_PATH          →", REWARD_CSV_PATH)
print("MAIN_CSV_PATH            →", MAIN_CSV_PATH)
print("MAIN_DATA_DIR            →", MAIN_DATA_DIR)

config.NUM_PREV_FRAMES = 4               # N previous frames (→ 5‑frame input)
config.BATCH_SIZE      = 64
config.LR              = 3e-4
config.IMAGE_SIZE      = getattr(config, 'IMAGE_SIZE', 128)
print('Config ready ✨')


# In[3]:


# ---------------- Dataset (unchanged from v3) ----------------
class StackedRewardDataset(Dataset):
    def __init__(self, main_csv_path, reward_csv_path, data_dir, image_size, num_prev_frames, transform=None):
        super().__init__()
        self.main_df   = pd.read_csv(main_csv_path)
        self.reward_df = pd.read_csv(reward_csv_path)
        self.data_dir  = data_dir
        self.image_size = image_size
        self.transform = transform
        self.num_prev_frames = num_prev_frames

        self.reward_map = dict(zip(self.reward_df['dataframe_index'], self.reward_df['assigned_reward']))
        self.valid_indices = [
            i for i in range(self.num_prev_frames, len(self.main_df))
            if i in self.reward_map and (
                'session_id' not in self.main_df.columns or
                self.main_df.iloc[i]['session_id'] == self.main_df.iloc[i - self.num_prev_frames]['session_id']
            )
        ]
        if not self.valid_indices:
            raise ValueError('No valid indices found')
        print(f'Dataset loaded → {len(self.valid_indices)} sequences with labels')

    def __len__(self):
        return len(self.valid_indices)

    def _load(self, rel):
        img = Image.open(os.path.join(self.data_dir, rel)).convert('RGB')
        return self.transform(img) if self.transform else transforms.ToTensor()(img)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        reward = self.reward_map[i]
        curr_row = self.main_df.iloc[i]
        curr = self._load(curr_row['image_path'])
        prev = [self._load(self.main_df.iloc[i - off]['image_path']) for off in range(self.num_prev_frames, 0, -1)]
        stacked = torch.cat(prev + [curr], dim=0)
        return stacked, torch.tensor(reward, dtype=torch.float32)


# In[4]:


# ---------------- Build loaders ----------------
tfm = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor()
])

full_ds = StackedRewardDataset(
    MAIN_CSV_PATH,        # ← new
    REWARD_CSV_PATH,      # ← new
    MAIN_DATA_DIR,        # ← new
    config.IMAGE_SIZE,
    config.NUM_PREV_FRAMES,
    tfm
)
train_ds, val_ds = random_split(full_ds, [int(0.8 * len(full_ds)), len(full_ds) - int(0.8 * len(full_ds))], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f'Train/Val split → {len(train_ds)} / {len(val_ds)} samples')


# In[5]:


# ---------------- Model / Optim / AMP ----------------
model = RewardEstimatorResNet(n_frames=config.NUM_PREV_FRAMES + 1).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=config.LR)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
loss_fn = nn.MSELoss()

print(f'Param count: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')


# In[8]:


# ---------------- Training loop with progress bars ----------------
EPOCHS   = 50
best_val = float('inf')

for epoch in range(1, EPOCHS + 1):
    # ----- training -----
    model.train(); running = 0
    for x, y in tqdm(train_loader, desc=f'E{epoch:02}/train', leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        running += loss.item() * x.size(0)
    train_loss = running / len(train_loader.dataset)

    # ----- validation -----
    model.eval(); running = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f'E{epoch:02}/val', leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                running += loss_fn(model(x).squeeze(), y).item() * x.size(0)
    val_loss = running / len(val_loader.dataset)

    # ----- checkpoint -----
if val_loss < best_val:
    best_val = val_loss



    # lightweight copy (model weights only) for inference notebooks
    torch.save(model.state_dict(),
               os.path.join(REWARD_MODEL_OUTPUT_DIR, "best_reward_estimator_weights.pth"))

    print(f"✨ Epoch {epoch:02} – new best val {val_loss:.4f}. "
          f"Saved {os.path.join(REWARD_MODEL_OUTPUT_DIR, 'best_reward_estimator_weights.pth')}")


# ### Why the GPU might still sit idle
# 1. **CPU transforms bottleneck** – heavy PIL transforms can starve the GPU; enable more `num_workers`.
# 2. **Small network / batch** – ResNet‑18 + 128×128 images at BS = 64 may use <10 % GPU; try bigger batches.
# 3. **CUDA toolkit mismatch** – if `torch.cuda.is_available()` prints **False**, reinstall PyTorch with the correct CUDA build.
# 4. **Data pinned to CPU** – ensure `.to(device)` is called (this notebook does).
# 
# Monitor real‑time usage with `nvidia‑smi dmon` or MSI Afterburner while a training epoch is running.

# In[ ]:




