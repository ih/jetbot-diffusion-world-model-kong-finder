import torch
import os
import torchvision.transforms as transforms
# Keep data and output in a separate directory to make uploading code to Gemini easier
AUXILIARY_DIR = r'C:\Projects\jetbot-diffusion-world-model-kong-finder-aux'
# --- Data ---
SESSION_DATA_DIR = os.path.join(AUXILIARY_DIR, 'jetbot_session_data_two_actions')
DATA_DIR = os.path.join(AUXILIARY_DIR, 'jetbot_data_two_actions')
# DATA_DIR = os.path.join(AUXILIARY_DIR, 'jetbot_data_two_actions')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'data.csv')
IMAGE_SIZE = 224
NUM_PREV_FRAMES = 4
MANUAL_COLLECTED_REWARD_CSV = os.path.join(DATA_DIR, "interactive_reward_labels_subset.csv") 


# --- Model ---
MODEL_ARCHITECTURE = 'SimpleUNetV1' # Name matching a class in models.py
NORM = 'group'
# --- Training ---
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000  # Probably don't need this in the testing notebook
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
USE_FP16 = True
ACCUMULATION_STEPS = 4
START_EPOCH = 0
OUTPUT_DIR = os.path.join(AUXILIARY_DIR, 'output_model_5hz_SimpleUnetV1')
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
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_PERCENTAGE = .1
MIN_EPOCHS = 5
# --- Data-rate control ----------------------------------------------------
TARGET_HZ            = 5          # ← choose 5 or 10
SOURCE_HZ            = 30         # how fast the robot actually logged
FRAME_STRIDE         = SOURCE_HZ // TARGET_HZ   # 30→5 Hz => stride 6




## --- Create Directories ---
os.makedirs(SESSION_DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
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