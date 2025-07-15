import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from pathlib import Path
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from utils import set_seed, get_logger, timer
# from model.audioset_tagging_cnn import PANNsCNN14Att
from dataset import *
# from model.audioset_tagging_cnn import AttBlock
from model.cnn_models import Cnn10, AttBlock
from sklearn.model_selection import train_test_split

ROOT = Path.cwd()
DATA_DIR = ROOT / "datasets" / 'DoorSoundDataset'
WORK_DIR = ROOT / "work_dir"
SR=16000
RANDOM_SEED = 42
set_seed(RANDOM_SEED)
model_config = {
    "sample_rate": SR,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 2
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 50

# 创建工作目录
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
WORK_DIR = WORK_DIR / current_time
os.makedirs(WORK_DIR, exist_ok=True)
logger = get_logger(out_file=WORK_DIR / "train.log")
logger.info("Config: \n" + str(model_config))
logger.info("Random seed: " + str(RANDOM_SEED))
logger.info("Device: " + str(device))

# 读取数据集
tmp_list = []
for sound_d in DATA_DIR.iterdir():
    if sound_d.is_file():
        continue
    for wav_f in sound_d.iterdir():
        tmp_list.append([sound_d.name, wav_f.name, wav_f.as_posix()])
train_all = pd.DataFrame(
    tmp_list, columns=["sound_code", "filename", "file_path"])

file_label_list = train_all[["file_path", "sound_code"]].values.tolist()

# 按30%划分验证集，剩下70%做训练集
train_file_list, val_file_list = train_test_split(
    file_label_list,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

logger.info(f"Train samples: {len(train_file_list)}")
logger.info(f"Val samples: {len(val_file_list)}")

waveform_transforms = ComposeTransforms([
    AddGaussianNoise(std=0.01),
    TimeStretch(rate_range=(0.9, 1.1)),
    PitchShift(sr=SR, n_steps_range=(-1, 1)),
    VolumeControl(gain_range=(0.8, 1.2))
])

loaders = {
    "train": data.DataLoader(PANNsDataset(train_file_list, waveform_transforms), 
                             batch_size=8, 
                             shuffle=True, 
                             drop_last=True),
    "valid": data.DataLoader(PANNsDataset(val_file_list, None), 
                             batch_size=8, 
                             shuffle=False,
                             drop_last=False)
}

# 模型
model_config["classes_num"] = 527
model = Cnn10(**model_config)
weights = torch.load("/Users/xucong/Desktop/subway_door/checkpoints/Cnn10_mAP=0.380.pth", map_location=device)
model.load_state_dict(weights["model"])
model.fc_audioset = nn.Linear(512, 2, bias=True)
# model.fc_audioset.init_weights()
# model.att_block = AttBlock(2048, 2, activation='sigmoid')
# model.att_block.init_weights()
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model loading completed, total parameters: {total_params}")

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Loss
criterion = nn.BCEWithLogitsLoss()

# ------------- 训练函数 ----------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, log_interval=10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in dataloader:
        x = batch['waveform'].to(device)
        y = batch['targets'].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        labels = torch.argmax(y, dim=1)
        correct += (pred == labels).sum().item()
        total += x.size(0)

    if scheduler is not None:
        scheduler.step() 

    return total_loss / total, correct / total

# ------------- 验证函数 ----------------
def eval_model_with_metrics(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['waveform'].to(device)
            y = batch['targets'].to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            labels = torch.argmax(y, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels)

    avg_loss = total_loss / total
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1


best_f1 = 0
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, loaders['train'], optimizer, criterion, device)
    val_loss, val_acc, val_prec, val_recall, val_f1 = eval_model_with_metrics(model, loaders['valid'], criterion, device)
    logger.info(
        f"Epoch {epoch} | "
        f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
        f"Val loss {val_loss:.4f}, acc {val_acc:.4f}, "
        f"prec {val_prec:.4f}, recall {val_recall:.4f}, f1 {val_f1:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), WORK_DIR / f"PANNsCNN10_SubwayDoor_F1_{best_f1:.3f}.pth")
        logger.info("Best model saved.")

