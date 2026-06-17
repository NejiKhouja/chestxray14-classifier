import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import ChestXrayDataset, get_transforms, build_image_index, LABELS, NUM_CLASSES
from model import ChestXrayModel

TRAIN_CSV  = '/kaggle/working/train.csv'
VAL_CSV    = '/kaggle/working/val.csv'
IMAGE_ROOT = '/kaggle/input/nih-chest-xrays/data'
CKPT_DIR   = '/kaggle/working/checkpoints'

BATCH_SIZE   = 32
NUM_EPOCHS   = 15
NUM_WORKERS  = 4
IMG_SIZE     = 224
LR_BACKBONE  = 2e-5
LR_HEAD      = 2e-4
WEIGHT_DECAY = 1e-4
POS_CAP      = 10.0     

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_pos_weight(csv_path, cap=POS_CAP):
    df = pd.read_csv(csv_path)
    w = []
    for label in LABELS:
        pos = float(df[label].sum())
        neg = float(len(df) - pos)
        w.append(min(neg / max(pos, 1.0), cap))
    return torch.tensor(w, dtype=torch.float32)


def get_loaders():
    # build the filename->path index once and share it across both datasets
    index = build_image_index(IMAGE_ROOT)
    train_ds = ChestXrayDataset(TRAIN_CSV, IMAGE_ROOT,
                                transform=get_transforms(train=True, img_size=IMG_SIZE),
                                image_index=index)
    val_ds   = ChestXrayDataset(VAL_CSV, IMAGE_ROOT,
                                transform=get_transforms(train=False, img_size=IMG_SIZE),
                                image_index=index)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0, prefetch_factor=2 if NUM_WORKERS else None)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0, prefetch_factor=2 if NUM_WORKERS else None)
    return train_loader, val_loader


def per_class_auc(targets, probs):
    """AUC per class (NaN where a class has no positive in val) + their mean."""
    aucs = []
    for i in range(targets.shape[1]):
        if 0 < targets[:, i].sum() < len(targets):
            aucs.append(roc_auc_score(targets[:, i], probs[:, i]))
        else:
            aucs.append(float('nan'))
    mean = float(np.nanmean(aucs))
    return aucs, mean


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, probs, targets = 0.0, [], []
    for images, ages, genders, y in tqdm(loader, desc="  val", leave=False):
        images, ages, genders, y = images.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE), y.to(DEVICE)
        with autocast():
            logits = model(images, ages, genders)
            loss_sum += criterion(logits, y).item()
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(y.cpu().numpy())
    probs, targets = np.concatenate(probs), np.concatenate(targets)
    aucs, mean = per_class_auc(targets, probs)
    return loss_sum / len(loader), aucs, mean


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Device: {DEVICE} | classes: {NUM_CLASSES}")

    model = ChestXrayModel(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    pos_w = compute_pos_weight(TRAIN_CSV).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = GradScaler()

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': LR_BACKBONE},
        {'params': list(model.clinical_branch.parameters()) +
                   list(model.classifier.parameters()), 'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    train_loader, val_loader = get_loaders()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR_BACKBONE, LR_HEAD],
        epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.2)

    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} "
          f"| Batches/epoch: {len(train_loader)}")
    print("-" * 60)

    best_auc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0
        for images, ages, genders, y in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS}"):
            images, ages, genders, y = images.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(images, ages, genders), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        val_loss, aucs, mean_auc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1:02d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | mean_AUC {mean_auc:.4f}")

        if val_loss == val_loss and (epoch == NUM_EPOCHS - 1 or mean_auc > best_auc):
            # print the per-class breakdown on the best/last epoch
            worst = sorted(zip(LABELS, aucs), key=lambda t: (t[1] != t[1], t[1]))[:5]
            print("   weakest classes:", ", ".join(f"{l} {a:.2f}" for l, a in worst))

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'val_auc': mean_auc, 'per_class_auc': dict(zip(LABELS, aucs)),
                        'labels': LABELS},
                       os.path.join(CKPT_DIR, 'model_best.pth'))
            print(f"   ✓ saved best (mean AUC {best_auc:.4f})")

    print(f"\nDone. Best mean val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    train()
