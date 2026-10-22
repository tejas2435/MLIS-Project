import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "image2voxel"))
from train import train

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 2D Image to 32x32x32 Voxel Model")
    parser.add_argument("--data_root", type=str, default="image2voxel/data/ShapeNet", help="Root folder of ShapeNet data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (0 recommended on Windows to avoid error 1455)")
    parser.add_argument("--save_path", type=str, default="image2voxel/best_model.pth", help="Checkpoint save filename")
    parser.add_argument("--synsets", type=str, default=None, help="Comma-separated synsets")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of sample pairs")
    parser.add_argument("--val_samples", type=int, default=500, help="Max validation samples to evaluate per epoch")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")

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
