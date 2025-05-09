#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U "transformers>=4.39" accelerate')


# In[2]:


#!/usr/bin/env python
# fine_tune_clip_reward.py
# ----------------------------------------------------------
# Fine-tune CLIP ViT-L/14 on your labelled Kong-centering data
# ----------------------------------------------------------
import os, random, json, torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToPILImage
from transformers import (
    CLIPProcessor, CLIPModel,
    TrainingArguments, Trainer
)
from transformers import TrainerCallback
import config
from math import ceil
import torch.nn.functional as F
from importnb import Notebook
with Notebook():
    from reward_dataset import RewardDatasetSingleFrame  # <- merges main & label CSVs

# ------------------------------ Hyper-params --------------------------------
MODEL_NAME        = "openai/clip-vit-large-patch14"   # 
# MODEL_NAME        = "openai/clip-vit-base-patch32"   # 
POS_PROMPT        = "a red Kong dog toy centered in the frame"
NEG_PROMPT        = "an empty kitchen floor with no toy"

POS_THRESH        = 0.6      # reward ≥ thresh → positive
BATCH_SIZE        =  64
LR                = 1e-5
EPOCHS            =  250
FREEZE_BACKBONES  = True    # set True if GPU RAM limited
OUTPUT_DIR        = os.path.join(config.OUTPUT_DIR, "clip_kong_finetune")
# ---------------------------------------------------------------------------

device = torch.device(config.DEVICE)
print("Device:", device)

def clip_val_top1(trainer):
    val_loader = trainer.get_eval_dataloader()
    hits = total = 0
    model = trainer.model.eval()
    for batch in val_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        preds = out.logits_per_image.argmax(dim=1)
        hits  += (preds == torch.arange(len(preds), device=preds.device)).sum().item()
        total += len(preds)
    return hits / total if total else 0.0

class Top1Callback(TrainerCallback):
    def __init__(self, trainer_ref):
        self.trainer_ref = trainer_ref          # stash a pointer

    def on_epoch_end(self, args, state, control, **kwargs):
        top1 = clip_val_top1(self.trainer_ref)
        print(f"Epoch {int(state.epoch)}  •  val top-1 = {top1:.3f}")

class CLIPContrastiveTrainer(Trainer):
    """Trainer that computes the standard CLIP InfoNCE loss."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text  = outputs.logits_per_text

        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=logits_per_image.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text,  labels)
        loss   = (loss_i + loss_t) / 2

        return (loss, outputs) if return_outputs else loss

# --------------------------- Base labelled dataset --------------------------
base_ds = RewardDatasetSingleFrame(
    main_csv_path = config.CSV_PATH,
    reward_csv_path = config.MANUAL_COLLECTED_REWARD_CSV,
    data_dir = config.DATA_DIR,
    image_size = config.IMAGE_SIZE,
    transform = None                      # we want raw PIL later
)

# --------------------------- Wrapper for CLIP -------------------------------
class CLIPFinetuneDataset(Dataset):
    """Turns the labelled frame → reward pairs into (PIL, text) pairs for CLIP."""
    def __init__(self, reward_ds, pos_thresh, pos_prompt, neg_prompt):
        self.ds          = reward_ds
        self.pos_thresh  = pos_thresh
        self.pos_prompt  = pos_prompt
        self.neg_prompt  = neg_prompt
        # Cache image paths + reward to avoid extra openings in __getitem__
        self.records = reward_ds.labeled_data_df.copy()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        image_path = os.path.join(self.ds.data_dir, row["image_path"])
        reward_val = row["assigned_reward"]
        prompt     = self.pos_prompt if reward_val >= self.pos_thresh else self.neg_prompt
        # PIL read here; no transforms (CLIPProcessor handles resize / norm)
        from PIL import Image
        pil_img = Image.open(image_path).convert("RGB")
        return {"image": pil_img, "text": prompt}

clip_ds = CLIPFinetuneDataset(base_ds, POS_THRESH, POS_PROMPT, NEG_PROMPT)

# --------------------------- Train / val split ------------------------------
VAL_SPLIT = 0.1
val_size  = int(len(clip_ds) * VAL_SPLIT)
train_size= len(clip_ds) - val_size
torch.manual_seed(42)
train_ds, val_ds = random_split(clip_ds, [train_size, val_size])

print(f"train={len(train_ds)}, val={len(val_ds)}")

# --------------------------- Model & processor ------------------------------
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model      = CLIPModel.from_pretrained(MODEL_NAME).to(device)

if FREEZE_BACKBONES:
    for n, p in model.named_parameters():
        if not n.startswith(("text_projection", "visual_projection", "logit_scale")):
            p.requires_grad = False
    print("Froze ViT & text backbones; training projection layers only.")

# --------------------------- Data collator ----------------------------------
def collate_fn(batch):
    images = [item["image"] for item in batch]
    texts  = [item["text"]  for item in batch]
    return processor(text=texts, images=images, return_tensors="pt", padding=True)

# --------------------------- Training setup ---------------------------------
steps_per_epoch = ceil(len(train_ds) / BATCH_SIZE)         # for save/eval

args = TrainingArguments(
    output_dir                   = OUTPUT_DIR,
    per_device_train_batch_size  = BATCH_SIZE,
    per_device_eval_batch_size   = BATCH_SIZE,
    learning_rate                = LR,
    num_train_epochs             = EPOCHS,
    fp16                         = True,
    logging_steps                = 50,            # still supported
    save_steps                   = steps_per_epoch,    # checkpoint each epoch
    eval_steps                   = steps_per_epoch,    # run eval each epoch
    save_total_limit             = 3,             # keep last 3 ckpts
    remove_unused_columns        = False,
)

trainer = CLIPContrastiveTrainer(        # <-- use subclass
    model           = model,
    args            = args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    data_collator   = collate_fn,
)

trainer.add_callback(Top1Callback(trainer))
# --------------------------- Train! -----------------------------------------
trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "ckpt-final"))
print("✅ Fine-tuning complete; model saved to", OUTPUT_DIR)

# --------------------------- Reward helper ----------------------------------
# After training you can create a one-liner reward function:
#
# ft_model = CLIPModel.from_pretrained(OUTPUT_DIR + "/ckpt-final").eval().to(device)
# with torch.no_grad():
#     pos_emb = ft_model.get_text_features(**processor(text=POS_PROMPT,
#                                                      return_tensors="pt").to(device)
#                   ).float().norm(dim=-1)
# def finetuned_reward(pil_img):
#     img_emb = ft_model.get_image_features(**processor(images=pil_img,
#                                                       return_tensors="pt").to(device)
#                   ).float().norm(dim=-1)
#     return (img_emb @ pos_emb.T).item()


# In[ ]:




