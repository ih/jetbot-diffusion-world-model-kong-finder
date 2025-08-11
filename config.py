import torch
import os
import torchvision.transforms as transforms
# Keep data and output in a separate directory to make uploading code to Gemini easier
AUXILIARY_DIR = r'C:\Projects\jetbot-diffusion-world-model-kong-finder-aux'
# --- Data ---
SESSION_DATA_DIR = os.path.join(AUXILIARY_DIR, 'jetbot_livingroom_session_data_single_position')
DATA_DIR = os.path.join(AUXILIARY_DIR, 'jetbot_data_two_actions_single_position')
# DATA_DIR = os.path.join(AUXILIARY_DIR, 'jetbot_data_two_actions')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'livingroom_data_incremental_test.csv')
NEW_DATA_DIR = os.path.join(AUXILIARY_DIR, 'jetbot_new_data_single_position')
NEW_IMAGE_DIR = os.path.join(NEW_DATA_DIR, 'images')
NEW_CSV_PATH = os.path.join(NEW_DATA_DIR, 'new.csv')
REPLAY_DIR = os.path.join(AUXILIARY_DIR, 'replay_buffer')
REPLAY_INDEX_PATH = os.path.join(REPLAY_DIR, 'index.pkl')
IMAGE_SIZE = 64
NUM_PREV_FRAMES = 4
MANUAL_COLLECTED_REWARD_CSV = os.path.join(DATA_DIR, "interactive_reward_labels_subset.csv")
MOVING_ACTION_VALUE = .15
# --- Paths for Model Comparison ---
HOLDOUT_DATA_DIR = os.path.join(AUXILIARY_DIR, 'jetbot_data_two_actions_holdout')

HOLDOUT_CSV_PATH = os.path.join(HOLDOUT_DATA_DIR, 'holdout.csv')


# --- Model ---
MODEL_ARCHITECTURE = 'Denoiser' # Name matching a class in models.py
NORM = 'batch'
# --- Training ---
DATALOADER_WORKERS   = 12         # or min(12, os.cpu_count())
PIN_MEMORY           = True
PERSISTENT_WORKERS   = True
PREFETCH_FACTOR      = 4          # optional; requires num_workers > 0

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LEARNING_RATE_WEIGHT_DECAY = 1e-2
LEARNING_RATE_EPS = 1e-8
LEARNING_RATE_WARMUP_STEPS = 100
NUM_EPOCHS = 1000  # Probably don't need this in the testing notebook
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
USE_FP16 = True
ACCUMULATION_STEPS = 4
START_EPOCH = 0
OUTPUT_DIR = os.path.join(AUXILIARY_DIR, 'output_model_4hz_DIAMOND_livingroom_model_plus_table_data')
# OUTPUT_DIR = os.path.join(AUXILIARY_DIR, 'output_model_small_session_split_data')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')  # Checkpoint directory
LOAD_CHECKPOINT = None # os.path.join(CHECKPOINT_DIR, 'model_best_epoch_62.pth')
# --- Output ---
SAVE_MODEL_EVERY = 100
SAMPLE_EVERY = 1
PLOT_EVERY = 10
SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'samples')        # Sample image directory
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')          # Loss plot directory
TEST_SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'test_samples')
SPLIT_DATASET_FILENAME = 'dataset_split.pth'

# For early stopping
MIN_EPOCHS = 20

# --- Incremental Training & Early Stopping (2000-Step Run) ---
MIX_ALPHA = .5
# Patience in epochs
EARLY_STOPPING_PATIENCE = 10
# --- Divergence Guard (These can often remain the same) ---
# Stop if training loss increases for 3 consecutive validation checks.
TRAIN_DIVERGE_PATIENCE_CHECKS = 3
# A 5% jump in training loss is still a good indicator of instability.
TRAIN_DIVERGE_THRESHOLD = 0.05

# --- Data-rate control ----------------------------------------------------
TARGET_HZ            = 4         # ← choose 5 or 10
SOURCE_HZ            = 30         # how fast the robot actually logged
FRAME_STRIDE         = SOURCE_HZ // TARGET_HZ   # 30→5 Hz => stride 6


# Denoiser & InnerModel specific
DM_SIGMA_DATA = 0.5
DM_SIGMA_OFFSET_NOISE = 0.1
DM_NOISE_PREVIOUS_OBS = True
DM_IMG_CHANNELS = 3
# DM_NUM_STEPS_CONDITIONING will use NUM_PREV_FRAMES, ensure NUM_PREV_FRAMES is defined above
DM_NUM_STEPS_CONDITIONING = NUM_PREV_FRAMES # Or set a specific integer value if preferred
DM_COND_CHANNELS = 2048
DM_UNET_DEPTHS = [2, 2, 2, 2]
DM_UNET_CHANNELS = [128, 256, 512, 1024]
DM_UNET_ATTN_DEPTHS = [False, False, True, True] # Boolean list
DM_NUM_ACTIONS = 2 # Adjust this based on your JetBot's actual number of discrete actions
DM_IS_UPSAMPLER = False
DM_UPSAMPLING_FACTOR = None # Or an integer like 2, 4 if DM_IS_UPSAMPLER is True

# Sampler specific (for inference/visualization)
SAMPLER_ORDER = 1
SAMPLER_NUM_STEPS = 50
SAMPLER_SIGMA_MIN = 0.002
SAMPLER_SIGMA_MAX = 80.0
SAMPLER_RHO = 7.0
SAMPLER_S_CHURN = 0.0
SAMPLER_S_TMIN = 0.0
SAMPLER_S_TMAX = float("inf")
SAMPLER_S_NOISE = 1.0

# Training specific (GRAD_CLIP_VALUE might be new if not used before)
GRAD_CLIP_VALUE = 10.0

# Karras-style sigma distribution parameters for training (NEW)
DM_SIGMA_P_MEAN = -1.2   # Log-mean of sigma distribution
DM_SIGMA_P_STD = 1.2     # Log-std of sigma distribution
DM_SIGMA_MIN_TRAIN = 0.002 # Min sigma during training
DM_SIGMA_MAX_TRAIN = 20.0  # Max sigma during training




## --- Create Directories ---
os.makedirs(SESSION_DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(NEW_DATA_DIR, exist_ok=True)
os.makedirs(NEW_IMAGE_DIR, exist_ok=True)
os.makedirs(REPLAY_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TEST_SAMPLE_DIR, exist_ok=True)
# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Diffusion Schedule ---
# You can also store the schedule type here, for maximum consistency:
SCHEDULE_TYPE = 'linear'  # Or 'cosine'

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])