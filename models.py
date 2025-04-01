# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Common Blocks (Keep these defined once) ---
class SinusoidalPositionEmbeddings(nn.Module):
    # (Paste the code for SinusoidalPositionEmbeddings here)
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
        if self.dim % 2 == 1:
             embeddings = F.pad(embeddings, (0, 1))
        return embeddings

class Block(nn.Module):
    # (Paste the code for Block here, ensure it accepts t_emb)
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
    def forward(self, x, t_emb): # Accepts pre-computed embedding
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb_proj = self.relu(self.time_mlp(t_emb))
        time_emb_proj = time_emb_proj[(..., ) + (None, ) * 2]
        h = h + time_emb_proj
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

# --- Architecture Definitions ---

class SimpleUNetV1(nn.Module): # Your original smaller model
    def __init__(self, image_channels=3, time_emb_dim=32, num_prev_frames=4):
        super().__init__()
        down_channels = (64, 128, 256)  # Original channels
        up_channels = (256, 128, 64)    # Original channels
        in_img_channels = image_channels * (num_prev_frames + 1)
        action_dim = 1
        effective_time_emb_dim = time_emb_dim + action_dim

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU() )
        self.conv0 = nn.Conv2d(in_img_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([])
        for i in range(len(down_channels)-1):
            self.downs.append(Block(down_channels[i], down_channels[i+1], effective_time_emb_dim))
        self.ups = nn.ModuleList([])
        for i in range(len(up_channels)-1):
            self.ups.append(Block(up_channels[i], up_channels[i+1], effective_time_emb_dim, up=True))
        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep, action, prev_frames):
        x = torch.cat([x, prev_frames], dim=1)
        t_emb = self.time_mlp(timestep)
        if action is not None:
             action = action.float()
             if len(action.shape) == 1: action = action.unsqueeze(1)
             t_action_emb = torch.cat([t_emb, action], dim=1)
        else:
            padding = torch.zeros(t_emb.shape[0], 1, device=t_emb.device)
            t_action_emb = torch.cat([t_emb, padding], dim=1)

        x = self.conv0(x)
        residual_inputs = []
        for i, down_block in enumerate(self.downs):
            x = down_block(x, t_action_emb)
            residual_inputs.append(x)
        for i, up_block in enumerate(self.ups):
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up_block(x, t_action_emb)
        return self.output(x)


class SimpleUNetV2_Larger(nn.Module): # Your current larger model
    def __init__(self, image_channels=3, time_emb_dim=32, num_prev_frames=4):
        super().__init__()
        # Increased channels/depth
        down_channels = (128, 256, 512, 512)
        up_channels = (512, 512, 256, 128)
        in_img_channels = image_channels * (num_prev_frames + 1)
        action_dim = 1
        effective_time_emb_dim = time_emb_dim + action_dim

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU())
        self.conv0 = nn.Conv2d(in_img_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([])
        for i in range(len(down_channels)-1):
            self.downs.append(Block(down_channels[i], down_channels[i+1], effective_time_emb_dim))
        self.ups = nn.ModuleList([])
        for i in range(len(up_channels)-1):
            self.ups.append(Block(up_channels[i], up_channels[i+1], effective_time_emb_dim, up=True))
        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep, action, prev_frames):
        # (Forward pass logic is identical to V1, just uses different layers)
        x = torch.cat([x, prev_frames], dim=1)
        t_emb = self.time_mlp(timestep)
        if action is not None:
             action = action.float()
             if len(action.shape) == 1: action = action.unsqueeze(1)
             t_action_emb = torch.cat([t_emb, action], dim=1)
        else:
            padding = torch.zeros(t_emb.shape[0], 1, device=t_emb.device)
            t_action_emb = torch.cat([t_emb, padding], dim=1)

        x = self.conv0(x)
        residual_inputs = []
        for i, down_block in enumerate(self.downs):
            x = down_block(x, t_action_emb)
            residual_inputs.append(x)
        for i, up_block in enumerate(self.ups):
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up_block(x, t_action_emb)
        return self.output(x)

# --- Add more architectures here as needed (e.g., UNetWithAttention) ---

# --- Optional: Model Factory ---
# This dictionary maps names (used in config) to classes
MODEL_REGISTRY = {
    'SimpleUNetV1': SimpleUNetV1,
    'SimpleUNetV2_Larger': SimpleUNetV2_Larger,
    # Add other models here
}

def get_model(config):
    """Instantiates a model based on the configuration."""
    model_name = config.MODEL_ARCHITECTURE
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    # Pass relevant config parameters to the model constructor
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(
        image_channels=3, # Or get from config if variable
        time_emb_dim=config.TIME_EMB_DIM,
        num_prev_frames=config.NUM_PREV_FRAMES
    )
    return model