#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import config
import csv
from torch.utils.data import random_split
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import random


# In[2]:


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

def save_existing_split(train_dataset, test_dataset, filename="dataset_split.pth"):
    """Saves the indices of existing Subset objects to a file.

    Args:
        train_dataset: The Subset object representing the training set.
        test_dataset: The Subset object representing the test set.
        filename: The name of the file to save the indices to.
    """

    # Check if they are actually Subset objects.  Important!
    if not isinstance(train_dataset, Subset) or not isinstance(test_dataset, Subset):
        raise TypeError("Both train_dataset and test_dataset must be Subset objects.")

    # Extract the indices. This is the key step.
    train_indices = train_dataset.indices
    test_indices = test_dataset.indices

    # Combine them into a single list (or tuple) for saving.
    all_indices = [train_indices, test_indices]

    # Save the indices using torch.save
    torch.save(all_indices, os.path.join(config.OUTPUT_DIR, filename))

def load_train_test_split(dataset, filename="dataset_split.pth"):
    """Loads the indices of existing Subset objects from a file."""

    filepath = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return None, None # Return None if file does not exist

    splits = torch.load(filepath)
    return tuple(Subset(dataset, indices) for indices in splits)

def display_dataset_entry(dataset_entry):
    frame, action, previous_frames = dataset_entry
    
    # Calculate the total number of frames to display
    total_frames = config.NUM_PREV_FRAMES + 1  # Previous frames + current frame
    
    # Create a figure with horizontal subplots
    plt.figure(figsize=(5*total_frames, 5))
    
    # Print the action
    plt.suptitle(f'Action: {action}', fontsize=16)
    
    # Display previous frames
    for i in range(config.NUM_PREV_FRAMES):
        plt.subplot(1, total_frames, i+1)
        prev_frame = previous_frames[(i * 3):(i + 1) * 3, :, :]  # Extract each frame (C, H, W)
        display_frame(prev_frame, title=f'Previous Frame {i+1}')
    
    # Display current frame
    plt.subplot(1, total_frames, total_frames)
    display_frame(frame, title='Current Frame')
    
    plt.tight_layout()
    plt.show()

def display_frame(frame, title=None):
    # Unnormalize the frame
    frame = (frame.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
    frame = (frame * 255).type(torch.uint8)  # Convert to uint8
    
    # Convert to PIL Image and then to numpy for matplotlib
    pil_frame = transforms.ToPILImage()(frame)
    
    # Display the frame
    plt.imshow(pil_frame)
    plt.axis('off')
    
    if title:
        plt.title(title)


def get_action_percentages():
    try:
        # Construct the full path using the config variable
        full_csv_path = config.CSV_PATH
    
        print(f"Loading combined dataset from: {full_csv_path}")
        df = pd.read_csv(full_csv_path)
    
        # --- Calculate Value Counts ---
        action_counts = df['action'].value_counts()
    
        # --- Calculate Percentages ---
        action_percentages = df['action'].value_counts(normalize=True) * 100
    
        # --- Print Results ---
        print("\n--- Action Split in Combined Dataset ---")
        print("Action Value Counts:")
        print(action_counts)
        print("\nAction Percentages:")
        print(action_percentages.map('{:.2f}%'.format)) # Format as percentage
        print("----------------------------------------")
    
    except FileNotFoundError:
        print(f"Error: Combined CSV file not found at expected location: {full_csv_path}")
        print("Please ensure combine_data.py has run successfully and config.py points to the correct file.")
    except KeyError:
        print("Error: 'action' column not found in the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")    

def split_train_test_by_session_id(dataset, train_split=0.8, seed=42):
    """
    Splits a dataset (expected to have a .dataframe attribute with 'session_id')
    into training and testing Subsets based on session IDs.

    Args:
        dataset (torch.utils.data.Dataset): An instance of a dataset class
            that has a `.dataframe` attribute (like your JetbotDataset)
            containing a 'session_id' column and a `.valid_indices` list
            mapping subset indices to dataframe indices.
        train_split (float): The proportion of sessions to use for training (e.g., 0.8 for 80%).
        seed (int): Random seed for reproducible shuffling of sessions.

    Returns:
        tuple(Subset, Subset): A tuple containing the training Subset and testing Subset.
                               Returns (None, None) if splitting is not possible.
    """
    if not hasattr(dataset, 'dataframe') or 'session_id' not in dataset.dataframe.columns:
        raise AttributeError("Input dataset must have a '.dataframe' attribute with a 'session_id' column.")
    if not hasattr(dataset, 'valid_indices'):
         raise AttributeError("Input dataset must have a '.valid_indices' attribute.")
    if len(dataset) == 0:
        print("Warning: Input dataset has zero valid samples. Cannot create split.")
        return Subset(dataset, []), Subset(dataset, []) # Return empty subsets

    # 1. Get unique session IDs from the underlying dataframe
    # Ensure we only consider sessions present in the valid indices
    valid_df_indices = dataset.valid_indices
    session_ids = dataset.dataframe.iloc[valid_df_indices]['session_id'].unique().tolist()

    if not session_ids:
         print("Warning: No unique session IDs found within the valid indices.")
         return Subset(dataset, []), Subset(dataset, []) # Return empty subsets

    # 2. Shuffle session IDs for random split
    rng = random.Random(seed) # Use a specific RNG instance for reproducibility
    rng.shuffle(session_ids)

    # 3. Split session IDs
    split_idx = int(train_split * len(session_ids))
    train_session_ids = set(session_ids[:split_idx])
    test_session_ids = set(session_ids[split_idx:])

    if not train_session_ids or not test_session_ids:
         print("Warning: Could not create a valid train/test split of session IDs. "
               "Perhaps only one session exists or train_split is 0 or 1?")
         # Decide how to handle: maybe return full dataset as train, empty as test?
         # Returning empty test set for now if split fails.
         if not train_session_ids:
              return Subset(dataset, []), dataset # All test
         else:
              return dataset, Subset(dataset, []) # All train

    print(f"Splitting by session: {len(train_session_ids)} train sessions, {len(test_session_ids)} test sessions.")

    # 4. Get sample indices (relative to the input dataset's length) for each split
    train_indices = []
    test_indices = []
    for i in range(len(dataset)): # Iterate 0 to len(dataset)-1
        actual_df_idx = dataset.valid_indices[i] # Get the actual index in the dataframe
        # Handle potential IndexError if valid_indices is somehow wrong
        try:
            session_id = dataset.dataframe.iloc[actual_df_idx]['session_id']
            if session_id in train_session_ids:
                train_indices.append(i) # Append the index 'i' (relative to dataset)
            elif session_id in test_session_ids:
                test_indices.append(i) # Append the index 'i' (relative to dataset)
        except IndexError:
             print(f"Warning: Index {actual_df_idx} out of bounds for dataframe (length {len(dataset.dataframe)}). Skipping index {i} in split creation.")


    # 5. Create Subset objects
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    if len(train_subset) == 0 or len(test_subset) == 0:
        print("Warning: Created split resulted in an empty train or test Subset. "
              "Check session distribution or data.")

    return train_subset, test_subset


# In[3]:


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = JetbotDataset(config.CSV_PATH, config.DATA_DIR, config.IMAGE_SIZE, config.NUM_PREV_FRAMES, transform=config.TRANSFORM)
    

    train_dataset, test_dataset = split_train_test_by_session_id(dataset)

    # print(dataset[40])
    
    # display_dataset_entry(test_dataset[40])


# In[4]:


display_dataset_entry(test_dataset[40])


# In[ ]:




