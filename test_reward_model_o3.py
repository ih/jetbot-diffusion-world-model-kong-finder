#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random, asyncio, cv2
from pathlib import Path

import torch
from torch.utils.data import random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm

import config                                 # project config
from models import RewardEstimatorResNet       # trained model

from importnb import Notebook
with Notebook():
    from jetbot_dataset import JetbotDataset   # dataset class

device = torch.device(config.DEVICE)
print("Using device ➜", device)


# In[6]:


REWARD_MODEL_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "reward_estimator")
BEST_WEIGHTS = os.path.join(REWARD_MODEL_OUTPUT_DIR, "best_reward_estimator_weights.pth")
IMAGE_SIZE    = config.IMAGE_SIZE
N_PREV_FRAMES = config.NUM_PREV_FRAMES        # e.g. 4 → 5-frame input

TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
])
BEST_WEIGHTS


# In[7]:


def stack_frames(prev_frames: torch.Tensor, current_img: torch.Tensor):
    """Return (3×(N+1),H,W) tensor by concatenating prev+current frames."""
    return torch.cat([prev_frames, current_img], dim=0)


def show_reward_predictions(model, dataset, num_samples=5, title=""):
    """Display random samples with predicted rewards."""
    model.eval()
    idxs = random.sample(range(len(dataset)), num_samples)
    plt.figure(figsize=(12, 4 * num_samples))

    for i, idx in enumerate(idxs):
        curr, _, prev = dataset[idx]                       # unpack tuple
        stacked = stack_frames(prev, curr).unsqueeze(0).to(device)

        with torch.no_grad():
            reward = model(stacked).item()

        img_disp = T.ToPILImage()(curr.cpu().clamp(0, 1))
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(img_disp); plt.axis("off")
        plt.title(f"Predicted reward: {reward:.3f}")

    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout(); plt.show()


# In[8]:


model = RewardEstimatorResNet(n_frames=N_PREV_FRAMES + 1).to(device)
state = torch.load(BEST_WEIGHTS, map_location=device)
model.load_state_dict(state)
model.eval()
print("Model loaded with",
      sum(p.numel() for p in model.parameters()) / 1e6, "M params")


# In[9]:


dataset = JetbotDataset(
    csv_path=config.CSV_PATH,
    data_dir=config.DATA_DIR,
    image_size=IMAGE_SIZE,
    num_prev_frames=N_PREV_FRAMES,
    transform=TRANSFORM,
    seed=42,
)
print("Dataset length:", len(dataset))



# In[10]:


show_reward_predictions(
    model, dataset, num_samples=5, title="Random JetBot samples"
)


# In[13]:


JETBOT_IP  = "192.168.68.52"   # ← change to your robot’s IP
REFRESH_HZ = 15
from jetbot_remote_client import RemoteJetBot
import ipywidgets as widgets

bot = RemoteJetBot(JETBOT_IP)
print("Connected to JetBot at", JETBOT_IP)

reward_label = widgets.Label(value="Reward: ---")
display(reward_label)

prev_buf = []                       # holds the last N_PREV_FRAMES *previous* frames

async def live_loop():
    global prev_buf
    while True:
        bgr = bot.get_frame()
        if bgr is not None:
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb)
            curr = TRANSFORM(pil)                # (3, H, W)

            # --------------------------------------------------------
            # Maintain buffer of *previous* frames (not including curr)
            # --------------------------------------------------------
            if len(prev_buf) == N_PREV_FRAMES:
                prev_buf.pop(0)                  # drop oldest
            prev_buf.append(curr)

            if len(prev_buf) == N_PREV_FRAMES:   # we now have N prev + curr
                prev_tensor = torch.cat(prev_buf, dim=0)       # (3N, H, W)
                stacked     = torch.cat([prev_tensor, curr],   # (3N+3, H, W)
                                         dim=0).unsqueeze(0).to(device)

                with torch.no_grad():
                    r = model(stacked).item()
                reward_label.value = f"Reward: {r:.3f}"

        await asyncio.sleep(1 / REFRESH_HZ)

await live_loop()


# In[ ]:




