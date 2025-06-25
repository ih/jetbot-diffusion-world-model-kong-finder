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


def gather_new_sessions_only(session_base_dir, processed_csv_path, new_image_dir, new_csv_path):
    """Collects only sessions not already present in processed_csv_path and
    writes them to a separate aggregate located at ``new_image_dir`` and ``new_csv_path``.
    This is useful for incremental training before permanently adding the
    sessions to the full dataset."""
    os.makedirs(new_image_dir, exist_ok=True)
    if os.path.exists(new_csv_path):
        os.remove(new_csv_path)

    existing_sessions = set()
    if os.path.exists(processed_csv_path):
        try:
            df_existing = pd.read_csv(processed_csv_path)
            if 'session_id' in df_existing.columns:
                existing_sessions = set(df_existing['session_id'].unique())
        except Exception as exc:
            print(f"Error reading processed CSV {processed_csv_path}: {exc}")

    try:
        session_dirs = [d for d in os.listdir(session_base_dir)
                        if os.path.isdir(os.path.join(session_base_dir, d)) and d.startswith('session_')]
        session_dirs.sort()
    except FileNotFoundError:
        print(f"Base session directory not found: {session_base_dir}")
        return []

    sessions_to_process = [s for s in session_dirs if s not in existing_sessions]
    print(f"Found {len(sessions_to_process)} new sessions to collect.")

    all_rows = []
    for session_name in tqdm(sessions_to_process, desc="Collecting New Sessions"):
        session_path = os.path.join(session_base_dir, session_name)
        session_csv = os.path.join(session_path, 'data.csv')
        session_img_dir = os.path.join(session_path, 'images')
        if not os.path.exists(session_csv) or not os.path.exists(session_img_dir):
            print(f"Skipping {session_name}, missing data.csv or images")
            continue
        try:
            df = pd.read_csv(session_csv)
        except Exception as exc:
            print(f"Error reading {session_csv}: {exc}")
            continue
        for _, row in df.iterrows():
            orig_rel = row['image_path']
            orig_abs = os.path.join(session_path, orig_rel)
            new_filename = f"{session_name}_{os.path.basename(orig_rel)}"
            new_rel = os.path.join('images', new_filename)
            new_abs = os.path.join(new_image_dir, new_filename)
            if not os.path.exists(orig_abs):
                continue
            if not os.path.exists(new_abs):
                try:
                    shutil.copy2(orig_abs, new_abs)
                except Exception as exc:
                    print(f"Could not copy {orig_abs}: {exc}")
                    continue
            all_rows.append({'session_id': session_name,
                             'image_path': new_rel,
                             'timestamp': row.get('timestamp', ''),
                             'action': row['action']})

    if all_rows:
        pd.DataFrame(all_rows).to_csv(new_csv_path, index=False)
        print(f"Wrote {len(all_rows)} entries to {new_csv_path}")
    else:
        print("No new session data found.")
    return sessions_to_process


# In[3]:


combine_sessions_append(
    r'C:\Projects\jetbot-diffusion-world-model-kong-finder-aux\jetbot_session_data_two_actions_holdout_laundry', 
    r'C:\Projects\jetbot-diffusion-world-model-kong-finder-aux\jetbot_data_two_actions_holdout\images',
    r'C:\Projects\jetbot-diffusion-world-model-kong-finder-aux\jetbot_data_two_actions_holdout\holdout.csv'
)


# In[3]:


combine_sessions_append(config.SESSION_DATA_DIR, config.IMAGE_DIR, config.CSV_PATH)


# In[ ]:




