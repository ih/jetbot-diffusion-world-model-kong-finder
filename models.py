# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

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

class SimpleRewardEstimator(nn.Module):
    # MODIFIED: Takes input_channels=3 (single RGB frame)
    def __init__(self, input_channels=3, image_size=config.IMAGE_SIZE):
        super().__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Simple CNN architecture example
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), # Output: size/2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: size/4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: size/8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: size/16
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling
        )

        conv_output_size = 256 # Because of AdaptiveAvgPool2d

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # Output a single reward value
        )

    # MODIFIED: Takes single image tensor x_image
    def forward(self, x_image):
        # x_image shape: (batch_size, 3, image_size, image_size)
        x = self.conv_layers(x_image)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        reward = self.fc_layers(x)
        return reward


# --- Optional: Model Factory ---
# This dictionary maps names (used in config) to classes
MODEL_REGISTRY = {
    'SimpleUNetV1': SimpleUNetV1,
    'SimpleUNetV2_Larger': SimpleUNetV2_Larger,
    'SimpleRewardEstimator': SimpleRewardEstimator,
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