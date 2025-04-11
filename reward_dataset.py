#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset


# In[2]:


class RewardDatasetSingleFrame(Dataset):
    # MODIFIED: num_prev_frames removed from args
    def __init__(self, main_csv_path, reward_csv_path, data_dir, image_size, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform

        # --- Load DataFrames ---
        try:
            self.main_df = pd.read_csv(main_csv_path)
            print(f"Loaded main data CSV: {main_csv_path} ({len(self.main_df)} rows)")
        except FileNotFoundError:
            print(f"Error: Main data CSV not found at {main_csv_path}")
            raise
        except Exception as e:
            print(f"Error loading main data CSV: {e}")
            raise

        try:
            self.reward_df = pd.read_csv(reward_csv_path)
            print(f"Loaded reward labels CSV: {reward_csv_path} ({len(self.reward_df)} rows)")
        except FileNotFoundError:
            print(f"Error: Reward labels CSV not found at {reward_csv_path}. Please run reward_labeling.ipynb first.")
            raise
        except Exception as e:
            print(f"Error loading reward labels CSV: {e}")
            raise

        # --- Merge DataFrames ---
        if 'dataframe_index' not in self.reward_df.columns:
             raise ValueError("Reward CSV must contain 'dataframe_index' column.")
        if 'assigned_reward' not in self.reward_df.columns:
             raise ValueError("Reward CSV must contain 'assigned_reward' column.")
        if 'image_path' not in self.main_df.columns:
             raise ValueError("Main CSV must contain 'image_path' column.")

        # Merge reward labels with the image path from the main dataframe
        # Keep only rows from main_df that have a corresponding reward label
        self.labeled_data_df = self.main_df[['image_path']].merge(
            self.reward_df[['dataframe_index', 'assigned_reward']],
            left_index=True,
            right_on='dataframe_index',
            how='inner' # Keep only rows with rewards
        )
        # Drop the potentially duplicate index column if needed, keep image_path and assigned_reward
        self.labeled_data_df = self.labeled_data_df[['image_path', 'assigned_reward']].reset_index(drop=True)

        print(f"Created labeled dataframe with {len(self.labeled_data_df)} entries.")
        if len(self.labeled_data_df) == 0:
            raise ValueError("Labeled dataframe is empty. No matching reward labels found.")

        # MODIFIED: No need for complex valid_indices based on history
        # The length is simply the number of labeled frames we have
        print(f"Dataset length: {len(self.labeled_data_df)}")


    def __len__(self):
        # MODIFIED: Length is the number of rows in the labeled dataframe
        return len(self.labeled_data_df)

    def __getitem__(self, idx):
        """
        Fetches the image and its associated reward label.
        """
        if idx >= len(self.labeled_data_df):
             raise IndexError("Index out of bounds for Labeled DataFrame.")

        # --- Fetch data for the current frame ---
        current_row = self.labeled_data_df.iloc[idx]
        current_image_path = os.path.join(self.data_dir, current_row['image_path'])
        reward_label = current_row['assigned_reward']

        # --- Load current image ---
        try:
            # Load as PIL Image
            current_image_pil = Image.open(current_image_path).convert("RGB")

        except Exception as e:
            print(f"Error loading image {current_image_path} for index {idx}: {e}")
            # Return dummy data or raise error
            dummy_img = torch.zeros(3, self.image_size, self.image_size)
            return dummy_img, torch.tensor([0.0], dtype=torch.float32)

        # --- Apply final transform (if any) ---
        if self.transform:
            final_current_image_tensor = self.transform(current_image_pil)
        else:
            # If no transform specified, apply ToTensor manually as a fallback
            # (though usually a transform pipeline including ToTensor is expected)
            to_tensor_transform = transforms.ToTensor()
            final_current_image_tensor = to_tensor_transform(current_image_pil)


        # --- Return: image_tensor, reward_label_tensor ---
        return final_current_image_tensor, torch.tensor([reward_label], dtype=torch.float32)


# In[ ]:




