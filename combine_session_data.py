#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import shutil
import pandas as pd
from tqdm.auto import tqdm
import config


# In[2]:


def combine_sessions_append(session_base_dir, aggregate_image_dir, aggregate_csv_path):
    """
    Combines data from session directories into an aggregate dataset.
    - Uses session directory name as session_id.
    - Renames images using session_id as a prefix.
    - Appends data from new sessions to an existing CSV.
    """
    os.makedirs(aggregate_image_dir, exist_ok=True) #

    all_data = []

    try:
        session_dirs = [d for d in os.listdir(session_base_dir) if os.path.isdir(os.path.join(session_base_dir, d)) and d.startswith('session_')] #
        session_dirs.sort() #
    except FileNotFoundError: #
        print(f"Error: Base session directory not found: {session_base_dir}") #
        return #

    print(f"Found {len(session_dirs)} sessions to check from '{session_base_dir}'.") #

    # --- Determine which sessions are already processed (if CSV exists) ---
    existing_sessions = set()
    file_exists = os.path.exists(aggregate_csv_path) #
    if file_exists:
        try:
            print(f"Reading existing sessions from: {aggregate_csv_path}") #
            existing_df = pd.read_csv(aggregate_csv_path) #
            if 'session_id' in existing_df.columns:
                existing_sessions = set(existing_df['session_id'].unique()) #
            print(f"Found {len(existing_sessions)} existing sessions.")
        except pd.errors.EmptyDataError:
            print(f"Warning: Existing CSV '{aggregate_csv_path}' is empty.")
            file_exists = False # Treat as if it doesn't exist for writing header
        except Exception as e:
            print(f"Error reading existing aggregate CSV: {e}. Will attempt to proceed, but caution advised.")
            # We might proceed but risk duplicates if we can't read existing IDs

    # --- Process only new sessions ---
    sessions_to_process = [s for s in session_dirs if s not in existing_sessions]
    print(f"Found {len(sessions_to_process)} new sessions to process.")

    if not sessions_to_process:
        print("No new sessions to add. Exiting.")
        return

    for session_name in tqdm(sessions_to_process, desc="Processing New Sessions"): #
        session_path = os.path.join(session_base_dir, session_name) #
        session_csv = os.path.join(session_path, 'data.csv') #
        session_img_dir = os.path.join(session_path, 'images') #

        if not os.path.exists(session_csv) or not os.path.exists(session_img_dir): #
            print(f"Warning: Skipping session {session_name}, missing data.csv or images directory.") #
            continue #

        try:
            df = pd.read_csv(session_csv) #
            if df.empty: #
                 print(f"Warning: Skipping session {session_name}, data.csv is empty.") #
                 continue #
        except Exception as e: #
            print(f"Warning: Error reading {session_csv}, skipping session {session_name}. Error: {e}") #
            continue #

        print(f"Processing session: {session_name}, {len(df)} entries.") #

        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {session_name}", leave=False): #
            original_relative_path = row['image_path'] #
            original_absolute_path = os.path.join(session_path, original_relative_path) #
            original_filename = os.path.basename(original_relative_path) #

            if not os.path.exists(original_absolute_path): #
                 print(f"  Warning: Image not found, skipping: {original_absolute_path}") #
                 continue #

            new_filename = f"{session_name}_{original_filename}" #
            new_relative_path = os.path.join('images', new_filename) #
            new_absolute_path = os.path.join(aggregate_image_dir, new_filename) #

            try:
                if not os.path.exists(new_absolute_path): #
                    shutil.copy2(original_absolute_path, new_absolute_path) #
            except Exception as e: #
                print(f"  Error copying image {original_absolute_path} to {new_absolute_path}. Skipping. Error: {e}") #
                continue #

            all_data.append({ #
                'session_id': session_name, #
                'image_path': new_relative_path, #
                'timestamp': row['timestamp'], #
                'action': row['action'] #
            })

    # --- Write new data (if any) ---
    if not all_data:
         print("\nNo new valid data found in session directories to add.") #
         return #

    new_df_to_write = pd.DataFrame(all_data, columns=['session_id', 'image_path', 'timestamp', 'action']) #

    try:
        if file_exists:
            # Append to existing file without header
            print(f"Appending {len(new_df_to_write)} new entries to {aggregate_csv_path}")
            new_df_to_write.to_csv(aggregate_csv_path, mode='a', header=False, index=False)
        else:
            # Write new file with header
            print(f"Creating new aggregate file {aggregate_csv_path} with {len(new_df_to_write)} entries.")
            new_df_to_write.to_csv(aggregate_csv_path, mode='w', header=True, index=False)

        # Optional: Print total count after adding
        final_df = pd.read_csv(aggregate_csv_path)
        print(f"\nAggregate data saved. Total entries now: {len(final_df)}")

    except Exception as e:
         print(f"\nError writing aggregated CSV file to {aggregate_csv_path}. Error: {e}") #


# In[3]:


combine_sessions_append(config.SESSION_DATA_DIR, config.IMAGE_DIR, config.CSV_PATH)


# In[ ]:




