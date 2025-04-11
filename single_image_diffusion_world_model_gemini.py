#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # For progress bars
import time
import datetime  # Import the datetime module

# --- Configuration ---
IMAGE_PATH = 'single-image.jpg'  # Replace with the path to your image
IMAGE_SIZE = 224  # Resize images to this size
BATCH_SIZE = 1  # Start with 1 for single-image training
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000  # Adjust as needed
NUM_TIMESTEPS = 1000  # Number of diffusion timesteps
BETA_START = 1e-4  # Starting value for beta (noise schedule)
BETA_END = 0.02  # Ending value for beta (noise schedule)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL_EVERY = 10000  # Save model checkpoints every N epochs
SAMPLE_EVERY = 100      # Generate samples every N epochs
OUTPUT_DIR = 'output' # Directory to save results and model
LOAD_CHECKPOINT = None  # Path to checkpoint file (e.g., 'output/model_epoch_500.pth'), or None to start fresh
START_EPOCH = 0 # If LOAD_CHECKPOINT is not None, this will be overwritten

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading --- (No changes here)
class SingleImageDataset(Dataset):
    def __init__(self, image_path, image_size):
        self.image_path = image_path
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to (-1, 1)
        ])
        self.image = self.load_image()

    def load_image(self):
        image = Image.open(self.image_path).convert("RGB")
        return self.transform(image)

    def __len__(self):
        return 1  # Only one image

    def __getitem__(self, idx):
        # We return the image and a dummy action (which will be ignored for now)
        return self.image, torch.tensor([0.0]) # Dummy action.  Shape (1,)


# --- Diffusion Helpers --- (No changes here)
def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, betas, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list((torch.sqrt(alphas_cumprod)), t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        torch.sqrt(1. - alphas_cumprod), t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# --- U-Net Model --- (No changes here)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


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
        return embeddings


class SimpleUNet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512)
        up_channels = (512, 256, 128, 64)
        out_dim = 3  # Output is RGB image
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        # Final output layer
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


# --- Training Loop ---

def train(model, dataloader, optimizer, betas, start_epoch, num_epochs, device, save_every, sample_every, output_dir):
    start_time = time.time()  # Record the start time
    for epoch in range(start_epoch, num_epochs): # Start from start_epoch
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, (images, actions) in enumerate(pbar):  # Include actions
            images = images.to(device)
            actions = actions.to(device) # Move actions to device

            # Sample timesteps
            t = torch.randint(0, NUM_TIMESTEPS, (images.shape[0],), device=device).long()

            # Forward diffusion (add noise)
            x_noisy, noise = forward_diffusion_sample(images, t, betas, device)

            # Predict noise
            predicted_noise = model(x_noisy, t)  # No action passed.

            # Calculate loss
            loss = F.mse_loss(noise, predicted_noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})
        # --- Checkpointing and Sampling ---
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,  # Save the last loss value
            }, os.path.join(output_dir, f"model_epoch_img{IMAGE_SIZE}_{epoch+1}.pth"))
            print(f"Saved model checkpoint at epoch {epoch+1}")

        if (epoch + 1) % sample_every == 0:
            model.eval()
            with torch.no_grad():
                # Sample from the model
                x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device) #Single Image
                for i in reversed(range(1, NUM_TIMESTEPS)):
                    t = (torch.ones(1) * i).long().to(device)  # t is scalar, make it (1,)
                    predicted_noise = model(x, t)

                    alpha = alphas.to(device)[t][:, None, None, None]
                    alpha_hat = alphas_cumprod.to(device)[t][:, None, None, None]
                    beta = betas.to(device)[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                # Denormalize and convert to PIL image
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                sample_image = transforms.ToPILImage()(x[0])
                sample_image.save(os.path.join(output_dir, f"sample_epoch_{epoch+1}_img{IMAGE_SIZE}.png"))
                current_time = time.time()
                elapsed_time = current_time - start_time
                formatted_elapsed_time = str(datetime.timedelta(seconds=elapsed_time))
                print(f"Saved sample image at epoch {epoch+1} after {formatted_elapsed_time}")
            model.train()
    end_time = time.time()  # Record the end time
    total_time = end_time - start_time
    # Format the time in a human-readable way
    formatted_time = str(datetime.timedelta(seconds=total_time))
    print(f"Total training time: {formatted_time}")
# --- Main Execution ---

if __name__ == '__main__':

    # Create dataset and dataloader
    dataset = SingleImageDataset(IMAGE_PATH, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) # No Shuffling

    # Calculate betas and alphas
    betas = linear_beta_schedule(NUM_TIMESTEPS, BETA_START, BETA_END)
    #betas = cosine_beta_schedule(NUM_TIMESTEPS) # Or try cosine schedule

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # Create model, optimizer
    model = SimpleUNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Load checkpoint if specified
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(LOAD_CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {START_EPOCH}")


    # Train the model
    train(model, dataloader, optimizer, betas, START_EPOCH, NUM_EPOCHS, DEVICE, SAVE_MODEL_EVERY, SAMPLE_EVERY, OUTPUT_DIR)
    print("Training complete!")


# In[ ]:




