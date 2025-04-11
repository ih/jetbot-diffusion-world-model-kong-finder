#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import csv
import shutil
import pandas as pd
from tqdm.auto import tqdm
import config


# In[13]:


def combine_sessions(session_base_dir, aggregate_image_dir, aggregate_csv_path):
    """
    Combines data from session directories into an aggregate dataset.
    - Uses session directory name as session_id.
    - Renames images using session_id as a prefix for unique naming suitable for incremental runs.
    """
    os.makedirs(aggregate_image_dir, exist_ok=True)

    all_data = []
    # global_image_counter = 0 # No longer needed for naming

    try:
        session_dirs = [d for d in os.listdir(session_base_dir) if os.path.isdir(os.path.join(session_base_dir, d)) and d.startswith('session_')]
        session_dirs.sort()
    except FileNotFoundError:
        print(f"Error: Base session directory not found: {session_base_dir}")
        return

    print(f"Found {len(session_dirs)} sessions to combine from '{session_base_dir}'.")

    for session_name in tqdm(session_dirs, desc="Combining Sessions"):
        session_path = os.path.join(session_base_dir, session_name)
        session_csv = os.path.join(session_path, 'data.csv')
        session_img_dir = os.path.join(session_path, 'images')

        # --- Basic session validity checks ---
        if not os.path.exists(session_csv) or not os.path.exists(session_img_dir):
            print(f"Warning: Skipping session {session_name}, missing data.csv or images directory.")
            continue

        try:
            df = pd.read_csv(session_csv)
            if df.empty:
                 print(f"Warning: Skipping session {session_name}, data.csv is empty.")
                 continue
        except Exception as e:
            print(f"Warning: Error reading {session_csv}, skipping session {session_name}. Error: {e}")
            continue

        print(f"Processing session: {session_name}, {len(df)} entries.")

        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {session_name}", leave=False):
            # original_relative_path is relative to session dir, e.g., 'images/image_00123.jpg'
            original_relative_path = row['image_path']
            original_absolute_path = os.path.join(session_path, original_relative_path)
            original_filename = os.path.basename(original_relative_path) # e.g., 'image_00123.jpg'

            if not os.path.exists(original_absolute_path):
                 print(f"  Warning: Image not found, skipping: {original_absolute_path}")
                 continue

            # --- Create new unique filename using session_id prefix ---
            new_filename = f"{session_name}_{original_filename}"
            # new_relative_path is relative to AGGREGATE_DATA_DIR, e.g., 'images/session_XYZ_image_00123.jpg'
            new_relative_path = os.path.join('images', new_filename)
            new_absolute_path = os.path.join(aggregate_image_dir, new_filename)

            # Copy and rename image
            try:
                # Check if target already exists - important if re-running/incremental
                if not os.path.exists(new_absolute_path):
                    shutil.copy2(original_absolute_path, new_absolute_path)
                # else: # Optional: Add logic here if you want to handle existing files differently
                #     print(f"  Info: Target image already exists, skipping copy: {new_absolute_path}")
                pass # If it exists, assume it's from a previous run, do nothing
            except Exception as e:
                print(f"  Error copying image {original_absolute_path} to {new_absolute_path}. Skipping. Error: {e}")
                continue

            # Append data to master list
            all_data.append({
                'session_id': session_name,
                'image_path': new_relative_path, # Store the new relative path with prefix
                'timestamp': row['timestamp'],
                'action': row['action']
            })
            # global_image_counter += 1 # No longer needed

    # --- Combine with existing data if AGGREGATE_CSV_PATH exists (Basic Incremental Logic) ---
    if os.path.exists(aggregate_csv_path):
        print(f"Found existing aggregate CSV: {aggregate_csv_path}. Appending new data.")
        try:
            existing_df = pd.read_csv(aggregate_csv_path)
            # Get session IDs already present
            existing_sessions = set(existing_df['session_id'].unique())
            # Filter new data to only include sessions not already present
            new_data_df = pd.DataFrame(all_data)
            new_data_to_add = new_data_df[~new_data_df['session_id'].isin(existing_sessions)]

            if not new_data_to_add.empty:
                print(f"Adding data for {len(new_data_to_add['session_id'].unique())} new sessions.")
                combined_df = pd.concat([existing_df, new_data_to_add], ignore_index=True)
            else:
                print("No new sessions found to add.")
                combined_df = existing_df # No changes needed
        except Exception as e:
            print(f"Error reading or processing existing aggregate CSV. Overwriting. Error: {e}")
            # Fallback to just writing the new data if reading fails
            combined_df = pd.DataFrame(all_data, columns=['session_id', 'image_path', 'timestamp', 'action'])

    elif all_data:
         print("No existing aggregate CSV found. Creating new file.")
         combined_df = pd.DataFrame(all_data, columns=['session_id', 'image_path', 'timestamp', 'action'])
    else:
         print("\nNo valid data found in session directories to combine.")
         return # Exit if no data


    # Write the combined/updated aggregate CSV
    try:
        combined_df.to_csv(aggregate_csv_path, index=False)
        print(f"\nAggregate data saved to {aggregate_csv_path}")
        print(f"Total entries in aggregate CSV: {len(combined_df)}")
    except Exception as e:
         print(f"\nError saving aggregated CSV file to {aggregate_csv_path}. Error: {e}")



# In[14]:


combine_sessions(config.SESSION_DATA_DIR, config.IMAGE_DIR, config.CSV_PATH)


# In[ ]:




