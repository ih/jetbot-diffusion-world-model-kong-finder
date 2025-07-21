#!/usr/bin/env python
# coding: utf-8

# ## Timing Comparison for Training Loops

# In[1]:


import wandb
import config
import diamond_world_model_trainer as trainer
import incremental_training as incremental_trainer
import os
import shutil
import time
import gc
import torch

run = wandb.init(project='timing-comparison', reinit=True)


# ### Run `_main_training` on non-incremental dataset

# In[2]:


gc.collect()
torch.cuda.empty_cache()

config.OUTPUT_DIR = os.path.join(config.AUXILIARY_DIR, 'output_model_2hz_DIAMOND_laundry_nonincremental_test')
config.DATA_DIR = os.path.join(config.AUXILIARY_DIR, 'jetbot_data_two_actions_nonincremental_test')
config.IMAGE_DIR = os.path.join(config.DATA_DIR, 'images')
config.CSV_PATH = os.path.join(config.DATA_DIR, 'laundry_data_incremental_test.csv')
config.NUM_EPOCHS = 1

if os.path.exists(config.OUTPUT_DIR):
    shutil.rmtree(config.OUTPUT_DIR)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

start = time.time()
trainer_run = trainer._main_training(finish_run=False)
noninc_duration = time.time() - start
wandb.finish()


# ### Run `train_diamond_model` on incremental dataset

# In[ ]:


gc.collect()
torch.cuda.empty_cache()

config.OUTPUT_DIR = os.path.join(config.AUXILIARY_DIR, 'output_model_2hz_DIAMOND_laundry_incremental_test')
config.DATA_DIR = os.path.join(config.AUXILIARY_DIR, 'jetbot_data_two_actions_incremental_test')
config.IMAGE_DIR = os.path.join(config.DATA_DIR, 'images')
config.CSV_PATH = os.path.join(config.DATA_DIR, 'laundry_data_incremental_test.csv')
config.EARLY_STOPPING_PATIENCE = 1

if os.path.exists(config.OUTPUT_DIR):
    shutil.rmtree(config.OUTPUT_DIR)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

wandb.init(project='timing-comparison', reinit=True)
start = time.time()
incremental_trainer.main()
inc_duration = time.time() - start
wandb.finish()


# ### Compare timings

# In[ ]:


import pandas as pd

comparison_df = pd.DataFrame({
    'run': ['non_incremental', 'incremental'],
    'duration_sec': [noninc_duration, inc_duration]
})
comparison_df

