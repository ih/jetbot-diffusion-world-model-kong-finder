import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset # Modified this line
import torchvision.transforms as transforms
from PIL import Image
import config
from tqdm.auto import tqdm # Added this line

class JetbotDataset(Dataset):
    def __init__(self, csv_path, data_dir, image_size, num_prev_frames, transform=None, seed=42): # Renamed args
        """
        Loads combined Jetbot data and prepares sequences.
        Train/test splitting should be done externally.

        Args:
            csv_path: Path to the combined CSV file (e.g., 'jetbot_data/data.csv').
            data_dir: Path to the base directory containing the combined 'images' folder
                      (e.g., 'jetbot_data').
            image_size: Target image size.
            num_prev_frames: Number of previous frames for history.
            transform: PyTorch transforms to apply to images.
            seed: Random seed for reproducible train/test splits if done externally.
        """
        super().__init__()
        self.csv_path = csv_path
        self.data_dir = data_dir # Base directory for combined data
        self.image_size = image_size
        self.transform = transform
        self.num_prev_frames = num_prev_frames
        self.seed = seed

        self.dataframe = self.load_data()
        # Calculate indices in the dataframe that are valid STARTING points for a sequence
        self.valid_indices = self._calculate_valid_indices()

    def load_data(self):
        """Loads the combined dataframe."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Combined CSV file not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        print(f"Loaded combined CSV with columns: {df.columns.tolist()}")
        if 'session_id' not in df.columns:
             raise ValueError("'session_id' column not found in CSV. Please run combine_data.py again.")
        return df

    def _calculate_valid_indices(self):
        """
        Valid START indices i such that
           • i, i-stride, i-2*stride, … are in the same session
           • stride == config.FRAME_STRIDE  (e.g. 6 for 5 Hz)
        """
        stride = config.FRAME_STRIDE
        valid = []
        for i in range(self.num_prev_frames * stride, len(self.dataframe)):
            if i % stride:                # keep only every Nth frame
                continue
            sess_now = self.dataframe.iloc[i]['session_id']
            sess_hist = self.dataframe.iloc[i - self.num_prev_frames*stride]['session_id']
            if sess_now == sess_hist:
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        stride      = config.FRAME_STRIDE
        actual_idx  = self.valid_indices[idx]

        cur_row     = self.dataframe.iloc[actual_idx]
        cur_img     = Image.open(os.path.join(self.data_dir, cur_row['image_path'])).convert("RGB")
        action      = cur_row['action']

        prev_frames = []
        for n in range(self.num_prev_frames, 0, -1):
            prev_row = self.dataframe.iloc[actual_idx - n*stride]
            prev_img = Image.open(os.path.join(self.data_dir, prev_row['image_path'])).convert("RGB")
            prev_frames.append(self.transform(prev_img) if self.transform else prev_img)

        cur_img = self.transform(cur_img) if self.transform else cur_img
        prev_frames_tensor = torch.cat(prev_frames, dim=0)

        return cur_img, torch.tensor([action], dtype=torch.float32), prev_frames_tensor

def filter_dataset_for_action_after_inaction(input_dataset, tolerance=1e-6):
    """
    Creates a Subset of a dataset containing only samples where the current action
    is non-zero, but the action for at least one of the recent previous frames is zero.

    Args:
        input_dataset (torch.utils.data.Dataset): The dataset or subset to filter.
            It assumes the dataset's __getitem__ returns (image, action_tensor, prev_frames_tensor).
            It also assumes the input_dataset has an underlying 'dataset' attribute
            which is the original JetbotDataset, and 'indices' if it's a Subset.
            The original JetbotDataset should have 'dataframe' and 'valid_indices' attributes.
        tolerance (float): Tolerance for floating-point comparison of actions.

    Returns:
        torch.utils.data.Subset: A new subset containing only the samples
                                 matching the criteria. Returns an empty
                                 Subset if no matching samples are found.
    """
    filtered_indices = []

    # Determine the base dataset
    if isinstance(input_dataset, Subset):
        base_dataset = input_dataset.dataset
    else: # It's the full JetbotDataset
        base_dataset = input_dataset

    print(f"Filtering dataset with {len(input_dataset)} samples for action after inaction...")

    for i in tqdm(range(len(input_dataset)), desc="Filtering for Action after Inaction"):
        try:
            # Get current action from the input_dataset's __getitem__
            _, current_action_tensor, _ = input_dataset[i]
            current_action_value = current_action_tensor.item()

            if abs(current_action_value) > tolerance: # Current action is non-zero
                # Determine the actual index in the base_dataset.dataframe
                if isinstance(input_dataset, Subset):
                    # 'i' is an index into the Subset. We need the corresponding index in base_dataset.valid_indices
                    actual_base_valid_idx = input_dataset.indices[i]
                else:
                    # 'i' is already an index into base_dataset.valid_indices
                    actual_base_valid_idx = i

                # This is the index in the original full dataframe
                dataframe_idx = base_dataset.valid_indices[actual_base_valid_idx]

                action_in_prev_frames_is_zero = False
                # Iterate through the N previous frames in the dataframe
                for k in range(1, base_dataset.num_prev_frames + 1):
                    # Calculate index for the k-th previous frame in the dataframe
                    prev_dataframe_idx = dataframe_idx - (k * config.FRAME_STRIDE)

                    prev_action_value = base_dataset.dataframe.iloc[prev_dataframe_idx]['action']
                    if abs(prev_action_value) < tolerance: # Previous action is zero
                        action_in_prev_frames_is_zero = True
                        break

                if action_in_prev_frames_is_zero:
                    filtered_indices.append(i) # 'i' is relative to input_dataset

        except IndexError as e:
            print(f"Warning: IndexError while accessing previous frame for index {i} (dataframe_idx {dataframe_idx if 'dataframe_idx' in locals() else 'unknown'}): {e}. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Error processing index {i} (dataframe_idx {dataframe_idx if 'dataframe_idx' in locals() else 'unknown'}) during filtering: {e}")
            continue

    if not filtered_indices:
        print("Warning: No samples found matching the 'action after inaction' criteria.")

    print(f"Filtered down to {len(filtered_indices)} samples.")
    return Subset(input_dataset, filtered_indices)
