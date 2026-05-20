import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import ChestXrayDataset, get_transforms
from model import ChestXrayModel

# ── config ────────────────────────────────────────────────────────────────────
# Kaggle paths — adjust if running locally
TRAIN_CSV      = '/kaggle/input/chest-xray14-preprocessed/train.csv'
IMAGE_ROOT     = '/kaggle/input/nih-chest-xrays/data'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'

BATCH_SIZE   = 32      # safe for Kaggle P100/T4 with AMP; try 64 if VRAM allows
NUM_EPOCHS   = 20
NUM_WORKERS  = 4       # Kaggle allows up to 4 in notebooks
VAL_SPLIT    = 0.1     # 10% held out for validation

# backbone (pretrained DenseNet) needs a much smaller LR than the new head
LR_BACKBONE  = 2e-5
LR_HEAD      = 2e-4
WEIGHT_DECAY = 1e-4    # AdamW regularization

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

POS_WEIGHT = torch.tensor([
     9.4496, 49.6807, 29.3338, 50.0000,  8.9910,
    50.0000, 50.0000, 50.0000,  5.2785, 20.4563,
     0.7134, 17.3755, 37.5870, 50.0000, 31.8070
], dtype=torch.float32).to(DEVICE)


# ── data ──────────────────────────────────────────────────────────────────────
def get_loaders():
    # two instances of the same CSV — one with augmentation, one without
    # this lets us apply different transforms to train vs val subsets
    train_full = ChestXrayDataset(TRAIN_CSV, IMAGE_ROOT, transform=get_transforms(train=True))
    val_full   = ChestXrayDataset(TRAIN_CSV, IMAGE_ROOT, transform=get_transforms(train=False))

    n     = len(train_full)
    val_n = int(n * VAL_SPLIT)
    rng   = np.random.default_rng(42)
    idx   = rng.permutation(n).tolist()
    train_idx, val_idx = idx[val_n:], idx[:val_n]

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(val_full,   val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    return train_loader, val_loader


def mean_auc(targets, probs):
    """Mean AUC-ROC across classes that have at least one positive sample."""
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            aucs.append(roc_auc_score(targets[:, i], probs[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


# ── training loop ─────────────────────────────────────────────────────────────
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model     = ChestXrayModel(num_classes=15, pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    scaler    = GradScaler()

    # differential LRs: backbone is pretrained so it needs tiny updates;
    # the head is randomly initialized so it needs larger updates
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(),                               'lr': LR_BACKBONE},
        {'params': list(model.clinical_branch.parameters()) +
                   list(model.classifier.parameters()),                       'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    train_loader, val_loader = get_loaders()

    # OneCycleLR: warms up then anneals over the full run — converges faster than ReduceLROnPlateau
    # pct_start=0.2 means 20% of batches are warmup, 80% are cooldown
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR_HEAD],
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
    )

    best_auc = 0.0
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print(f"Batches/epoch: {len(train_loader)}")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for images, ages, genders, targets in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} train"):
            images, ages, genders, targets = (
                images.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE), targets.to(DEVICE)
            )
            optimizer.zero_grad()

            with autocast():
                loss = criterion(model(images, ages, genders), targets)

            scaler.scale(loss).backward()
            # clip gradients — high pos_weights can cause loss spikes → exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # OneCycleLR updates every batch, not every epoch

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        all_probs   = []
        all_targets = []

        with torch.no_grad():
            for images, ages, genders, targets in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} val  "):
                images, ages, genders, targets = (
                    images.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE), targets.to(DEVICE)
                )
                with autocast():
                    logits = model(images, ages, genders)
                    val_loss += criterion(logits, targets).item()

                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss   /= len(val_loader)
        val_auc     = mean_auc(np.concatenate(all_targets), np.concatenate(all_probs))

        print(f"Epoch {epoch+1:02d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_AUC: {val_auc:.4f}")

        # save best checkpoint (by val AUC, not by loss)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch':             epoch + 1,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':          val_loss,
                'val_auc':           val_auc,
            }, os.path.join(CHECKPOINT_DIR, 'model_best.pth'))
            print(f"  ✓ best checkpoint (AUC {best_auc:.4f})")

        # also keep a periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch':             epoch + 1,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':          val_loss,
                'val_auc':           val_auc,
            }, os.path.join(CHECKPOINT_DIR, f'model_epoch{epoch+1}.pth'))

    print(f"\nDone. Best val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    train()
