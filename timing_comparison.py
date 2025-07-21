#!/usr/bin/env python
# coding: utf-8

# ## Timing Comparison for Training Loops

# In[1]:


import wandb
import config
import diamond_world_model_trainer as trainer
import incremental_training as incremental_trainer
import os

run = wandb.init(project="timing-comparison", reinit=True)


# ### Run `_main_training` on non-incremental dataset

# In[3]:


import pdb

config.OUTPUT_DIR = os.path.join(config.AUXILIARY_DIR, 'output_model_2hz_DIAMOND_laundry_nonincremental_test')
config.DATA_DIR = os.path.join(config.AUXILIARY_DIR, 'jetbot_data_two_actions_nonincremental_test')
config.NUM_EPOCHS = 1
trainer_run = trainer._main_training(finish_run=False)
pdb.set_trace()
noninc_table = trainer_run.history[-1].get("train_epoch_perf")
wandb.finish()


# ### Run `train_diamond_model` on incremental dataset

# In[ ]:


config.OUTPUT_DIR = os.path.join(config.AUXILIARY_DIR, 'output_model_2hz_DIAMOND_laundry_incremental_test')
config.DATA_DIR = os.path.join(config.AUXILIARY_DIR, 'jetbot_data_two_actions_incremental_test')
wandb.init(project="timing-comparison", reinit=True)

incremental_trainer.main()
inc_table = wandb.run.history[-1].get("incremental_perf")
wandb.finish()


# ### Compare timings

# In[ ]:


import pandas as pd

if noninc_table is not None and inc_table is not None:
    df_non = pd.DataFrame(noninc_table.data, columns=noninc_table.columns)
    df_inc = pd.DataFrame(inc_table.data, columns=inc_table.columns)
    display(df_non.describe())
    display(df_inc.describe())

