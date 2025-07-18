#!/usr/bin/env python
# coding: utf-8

# # Incremental vs Batch Training
# This notebook copies each session directory one by one, runs incremental training, and logs evaluation metrics with wandb.

# In[1]:


import os, shutil, time, datetime
import wandb
import torch
import gc
from incremental_training import main as incremental_main
from evaluate_holdout import evaluate_sampler_on_holdout

wandb.init(project='incremental_vs_batch')


# In[ ]:


source_dir = r'C:\Projects\jetbot-diffusion-world-model-kong-finder-aux\jetbot_laundry_session_data_two_actions'
dest_dir = r'C:\Projects\jetbot-diffusion-world-model-kong-finder-aux\jetbot_laundry_session_data_two_actions_incremental_test'
os.makedirs(dest_dir, exist_ok=True)
sessions = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

for session in sessions:
    src = os.path.join(source_dir, session)
    dst = os.path.join(dest_dir, session)
    if os.path.exists(dst):
        print(f'{dst} already exists, skipping copy')
        continue
    shutil.copytree(src, dst)
    print(f'Copied {src} -> {dst}')

    start = time.time()
    incremental_main()
    duration = str(datetime.timedelta(seconds=time.time() - start))
    print(f'Training took: {duration}')
    wandb.log({'session': session, 'training_duration': duration})

    gc.collect()
    torch.cuda.empty_cache()
    
    metrics, paths = evaluate_sampler_on_holdout()
    flat = {f'eval/{session}_{k}_{m}': v for k, vals in metrics.items() for m, v in vals.items()}
    wandb.log(flat)
    images = {f'eval/{session}_{name}': wandb.Image(path) for name, path in paths.items()}
    if images:
        wandb.log(images)

    gc.collect()
    torch.cuda.empty_cache()


wandb.finish()


# In[ ]:




