"""
Synthesis-driven self-supervised pretraining on unlabeled CCTA volumes.
The model learns to detect synthetically inserted coronary lesions,
biasing representations toward clinically relevant vascular pathology.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CORAPretrainingDataset
from models.model import CORAPretrainModel


# =============================================================================
# Loss Functions
# =============================================================================

class TverskyLoss(nn.Module):
    """
    Tversky loss for segmentation with controllable FP/FN trade-off.

    Setting beta > 0.5 penalizes false negatives more heavily,
    prioritizing recall for sparse lesion detection.
    """

    def __init__(self, alpha=0.1, beta=0.9, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = ((1 - targets) * probs).sum(dim=1)
        FN = (targets * (1 - probs)).sum(dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky).mean()


class FocalLoss(nn.Module):
    """
    Focal loss to down-weight easy background voxels.

    A high gamma (e.g., 4.0) forces the model to focus on hard
    pathological boundaries.
    """

    def __init__(self, alpha=0.25, gamma=4.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


class LesionSegmentationLoss(nn.Module):
    """
    Combined Tversky + Focal loss for lesion segmentation pretraining.

    L_total = L_Tversky(alpha=0.1, beta=0.9) + L_Focal(gamma=4.0)
    """

    def __init__(self, tversky_beta=0.9, focal_gamma=4.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=(1 - tversky_beta), beta=tversky_beta)
        self.focal = FocalLoss(gamma=focal_gamma)

    def forward(self, logits, targets):
        return self.tversky(logits, targets) + self.focal(logits, targets)


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Ensure each DataLoader worker has a unique but reproducible seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_snapshot(images, targets, logits, epoch, save_dir, batch_idx):
    """Save a visualization comparing input, ground truth, and prediction."""
    os.makedirs(save_dir, exist_ok=True)

    img = images.detach().cpu().numpy()[0, 2]  # Channel 2 (angiographic window)
    tgt = targets.detach().cpu().numpy()[0, 0]
    pred = (torch.sigmoid(logits).detach().cpu().numpy()[0, 0] > 0.5).astype(np.float32)

    # Slice at lesion center if present, otherwise at volume center
    d, h, w = img.shape
    if tgt.sum() > 0:
        coords = np.argwhere(tgt > 0)
        cz, cy, cx = coords.mean(axis=0).astype(int)
    else:
        cz, cy, cx = d // 2, h // 2, w // 2
    cz, cy, cx = np.clip([cz, cy, cx], 0, [d - 1, h - 1, w - 1])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Epoch {epoch + 1} | Batch {batch_idx}", fontsize=16)

    for row_idx, (title, vol, cmap) in enumerate([
        ("Input", img, "gray"),
        ("Ground Truth", tgt, "Greens"),
        ("Prediction", pred, "Reds"),
    ]):
        axes[row_idx, 0].imshow(vol[cz], cmap=cmap, vmin=0, vmax=1)
        axes[row_idx, 0].set_ylabel(title, fontsize=14, fontweight="bold")
        axes[row_idx, 1].imshow(np.rot90(vol[:, cy, :]), cmap=cmap, vmin=0, vmax=1)
        axes[row_idx, 2].imshow(np.rot90(vol[:, :, cx]), cmap=cmap, vmin=0, vmax=1)
        if row_idx == 0:
            for j, t in enumerate(["Axial", "Sagittal", "Coronal"]):
                axes[0, j].set_title(t)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_epoch_{epoch + 1:03d}.png"))
    plt.close()


def find_latest_checkpoint(checkpoint_dir):
    """Check for an existing checkpoint to resume training."""
    path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    return (path, 0) if os.path.exists(path) else (None, 0)


# =============================================================================
# Pretraining Loop
# =============================================================================

def main():
    # --- Configuration ---
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 8
    WARMUP_EPOCHS = 3
    PATCH_SHAPE = (96, 96, 96)
    SAVE_EVERY = 1

    EXCEL_FILE = "data/CTA_all_list.xlsx"
    NPZ_ROOT = "/path/to/preprocessed/npz"
    CHECKPOINT_DIR = "checkpoints/cora_pretrain"
    VIS_DIR = os.path.join(CHECKPOINT_DIR, "vis_snapshots")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    dataset = CORAPretrainingDataset(
        excel_file=EXCEL_FILE,
        npz_root=NPZ_ROOT,
        patch_shape=PATCH_SHAPE,
    )
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # --- Model ---
    model = CORAPretrainModel().to(device)

    # --- Loss ---
    criterion = LesionSegmentationLoss(tversky_beta=0.9, focal_gamma=4.0).to(device)

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    scaler = GradScaler()

    # --- Resume ---
    start_epoch = 0
    ckpt_path, _ = find_latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
    else:
        print("No checkpoint found. Starting from scratch.")

    # --- Training ---
    print(f"Starting pretraining from epoch {start_epoch + 1}...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch_idx, (images, labels, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=True):
                _, logits_seg = model(images)
                loss = criterion(logits_seg, masks)

                # Save snapshot for the first batch
                if batch_idx == 0:
                    save_snapshot(images, masks, logits_seg, epoch, VIS_DIR, batch_idx)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = running_loss / len(loader)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        # --- Save checkpoint ---
        if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "avg_loss": avg_loss,
                },
                os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    # Save final weights
    final_path = os.path.join(CHECKPOINT_DIR, "cora_pretrained.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Pretraining complete. Final weights saved to {final_path}")


if __name__ == "__main__":
    main()
