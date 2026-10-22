import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ImageToVoxel
from dataset import ShapeNetVoxelDataset

def dice_loss(pred, target, smooth=1e-5):
    """
    Computes Soft Dice Loss for 3D binary voxel occupancy grids.
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train(data_root="data/ShapeNet", epochs=50, batch_size=16, lr=1e-4, num_workers=0,
          save_path="best_model.pth", synsets=None, max_samples=None, resume=None, val_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    synset_list = [s.strip() for s in synsets.split(",")] if synsets else None

    train_set = ShapeNetVoxelDataset(root=data_root, split="train", synsets=synset_list, max_samples=max_samples)
    val_set   = ShapeNetVoxelDataset(root=data_root, split="val",   synsets=synset_list, max_samples=max_samples)

    if len(train_set) == 0:
        print("\n[Warning] No training data found!")
        print(f"Please check dataset path: '{data_root}'")
        return

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin
    )

    model = ImageToVoxel().to(device)

    # Resume if requested
    if resume and os.path.exists(resume):
        print(f"Resuming weights from '{resume}'...")
        model.load_state_dict(torch.load(resume, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    best_iou = 0.0
    print(f"Starting training for {epochs} epochs (Train samples: {len(train_set)}, Batch: {batch_size}, LR: {lr})...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}")
        for imgs, voxs in pbar:
            imgs, voxs = imgs.to(device), voxs.to(device)

            pred = model(imgs)
            pred_clamped = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)

            loss = bce(pred_clamped, voxs) + dice_loss(pred, voxs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validation IoU
        model.eval()
        iou_sum, count = 0.0, 0
        with torch.no_grad():
            for imgs, voxs in val_loader:
                if val_samples and val_samples > 0 and count >= val_samples:
                    break
                imgs, voxs = imgs.to(device), voxs.to(device)
                pred = (model(imgs) > 0.5).float()
                intersection = (pred * voxs).sum(dim=(1, 2, 3, 4))
                union = ((pred + voxs) > 0).float().sum(dim=(1, 2, 3, 4))
                iou_sum += (intersection / (union + 1e-5)).sum().item()
                count += imgs.size(0)

        mean_iou = iou_sum / max(count, 1)
        print(f"Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f} | Val IoU: {mean_iou:.4f} (tested on {count} samples)")

        # Save latest model checkpoint
        torch.save(model.state_dict(), "latest_model.pth")

        # Save best model checkpoint
        if mean_iou > best_iou or epoch == 1:
            if mean_iou > best_iou:
                best_iou = mean_iou
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model checkpoint (IoU: {mean_iou:.4f}) to '{save_path}'")

    print(f"\nTraining completed. Best Val IoU: {best_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D Image to 32x32x32 Voxel Model")
    parser.add_argument("--data_root", type=str, default="data/ShapeNet", help="Root folder of ShapeNet data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (0 recommended on Windows to avoid error 1455)")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Checkpoint save filename")
    parser.add_argument("--synsets", type=str, default=None, help="Comma-separated synsets (e.g. '03001627' for chairs)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of sample pairs to use")
    parser.add_argument("--val_samples", type=int, default=500, help="Max validation samples to evaluate per epoch (0 for all)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        save_path=args.save_path,
        synsets=args.synsets,
        max_samples=args.max_samples,
        val_samples=args.val_samples,
        resume=args.resume
    )
