import os
import argparse
import shutil
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from skimage import measure
import open3d as o3d
import matplotlib
matplotlib.use("Agg")  # Non-interactive, lightweight headless backend
import matplotlib.pyplot as plt
from model import ImageToVoxel

def predict(image_path, model_path="best_model.pth", threshold=0.5, output_obj="output.obj", visualize=False, save_voxel_plot=True):
    """
    Run 3D voxel reconstruction from a single 2D image and export an OBJ 3D mesh.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check image path
    if not os.path.exists(image_path):
        alt_path = os.path.join("image2voxel", image_path)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            raise FileNotFoundError(f"Input image not found: '{image_path}'")

    # Check model path (supports best_model.pth, best_model_fp16.pth, and subfolders)
    if not os.path.exists(model_path):
        candidates = [
            os.path.join("image2voxel", model_path),
            "best_model_fp16.pth",
            os.path.join("image2voxel", "best_model_fp16.pth"),
            "best_model.pth",
            os.path.join("image2voxel", "best_model.pth")
        ]
        for cand in candidates:
            if os.path.exists(cand):
                model_path = cand
                break

    model = ImageToVoxel().to(device)
    if os.path.exists(model_path):
        print(f"Loading trained weights from '{model_path}'...")
        state_dict = torch.load(model_path, map_location=device)
        # Convert fp16 weights to float32 automatically if needed
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        print(f"[Warning] Model weights '{model_path}' not found! Running with randomly initialized weights.")

    model.eval()

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"Loading input image: {image_path}")
    raw_pil = Image.open(image_path)

    # Handle transparent alpha channels: composite on pure white background
    if raw_pil.mode in ("RGBA", "LA") or (raw_pil.mode == "P" and "transparency" in raw_pil.info):
        rgba = raw_pil.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img_raw = Image.alpha_composite(white_bg, rgba).convert("RGB")
    else:
        img_raw = raw_pil.convert("RGB")

    img_t = transform(img_raw).unsqueeze(0).to(device)

    print("Running inference forward pass...")
    with torch.no_grad():
        pred_tensor = model(img_t)
        pred = pred_tensor.squeeze().cpu().numpy()  # (32, 32, 32)

    val_min, val_max, val_mean = float(pred.min()), float(pred.max()), float(pred.mean())
    val_spread = val_max - val_min
    print(f"Predicted voxel stats: Min={val_min:.4f}, Max={val_max:.4f}, Mean={val_mean:.4f}, Spread={val_spread:.4f}")

    if val_spread < 0.08:
        print("\n" + "="*70)
        print("[NOTICE - LOW CONFIDENCE WARNING]")
        print(f"The voxel outputs have low variance [{val_min:.3f}, {val_max:.3f}].")
        print("More training epochs or a cleaner silhouette image will improve detail.")
        print("="*70 + "\n")

    # Adaptive threshold check for Marching Cubes
    effective_thresh = threshold
    if val_max <= effective_thresh:
        effective_thresh = (val_min + val_max) / 2.0
        print(f"[Notice] Max voxel occupancy ({val_max:.4f}) <= threshold ({threshold}).")
        print(f"         Using adaptive threshold {effective_thresh:.4f} for mesh extraction.")

    # Marching Cubes -> extract surface mesh
    occupied_count = int(np.sum(pred > effective_thresh))
    print(f"Occupied voxels at threshold {effective_thresh:.3f}: {occupied_count} / {32*32*32}")

    if 5 <= occupied_count <= 28000:
        try:
            verts, faces, normals, _ = measure.marching_cubes(pred, level=effective_thresh)
            print(f"Extracted mesh with {len(verts)} vertices and {len(faces)} triangular faces.")

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()

            # Save OBJ in primary destination
            o3d.io.write_triangle_mesh(output_obj, mesh)
            print(f"Successfully saved 3D mesh -> '{output_obj}'")

            # Also mirror to image2voxel/ if running from root, or vice-versa
            alt_obj = os.path.join("image2voxel", os.path.basename(output_obj))
            if os.path.abspath(alt_obj) != os.path.abspath(output_obj):
                try:
                    shutil.copyfile(output_obj, alt_obj)
                    print(f"Synchronized mesh -> '{alt_obj}'")
                except Exception:
                    pass

            if visualize:
                try:
                    print("Opening Open3D interactive viewer (close window to finish)...")
                    o3d.visualization.draw_geometries([mesh], window_name="3D Reconstructed Voxel Mesh")
                except Exception as e:
                    print(f"[Notice] Open3D GUI viewer skipped: {e}")
        except Exception as e:
            print(f"[Warning] Marching cubes failed: {e}")
    else:
        print(f"[Warning] Marching cubes skipped: occupied voxel count ({occupied_count}) outside valid range.")

    # Visual preview plot with Matplotlib
    if save_voxel_plot:
        plot_path = os.path.splitext(output_obj)[0] + "_preview.png"
        try:
            fig = plt.figure(figsize=(9, 4))

            # 1. Input image
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(img_raw)
            ax1.set_title(f"Input Image (128x128)", fontsize=10)
            ax1.axis("off")

            # 2. Voxel 3D grid
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            occupied = pred > effective_thresh
            num_occ = int(np.sum(occupied))

            if 0 < num_occ <= 5000:
                ax2.voxels(occupied, facecolors="dodgerblue", edgecolor=None, alpha=0.8)
            elif num_occ > 5000:
                coords = np.argwhere(occupied)
                if len(coords) > 1500:
                    idx = np.random.choice(len(coords), 1500, replace=False)
                    coords = coords[idx]
                ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="dodgerblue", s=5, alpha=0.6)
            else:
                ax2.text(16, 16, 16, "No voxels above threshold", ha="center")

            ax2.set_xlim(0, 32)
            ax2.set_ylim(0, 32)
            ax2.set_zlim(0, 32)
            ax2.set_title(f"32³ Voxels (active={num_occ})", fontsize=10)

            plt.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved visual reconstruction preview -> '{plot_path}'")

            # Mirror preview to image2voxel/
            alt_preview = os.path.join("image2voxel", os.path.basename(plot_path))
            if os.path.abspath(alt_preview) != os.path.abspath(plot_path):
                try:
                    shutil.copyfile(plot_path, alt_preview)
                    print(f"Synchronized preview -> '{alt_preview}'")
                except Exception:
                    pass
        except Exception as e:
            print(f"[Notice] Preview image generation skipped ({e}), but 3D mesh (.obj) was saved.")

    print("\nInference complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image to 3D Voxel Mesh Inference")
    parser.add_argument("--image", type=str, default="test_image.png", help="Path to input 2D image")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy probability threshold (default: 0.5)")
    parser.add_argument("--output", type=str, default="output.obj", help="Path to output .obj mesh")
    parser.add_argument("--gui", action="store_true", help="Open interactive Open3D window")

    args = parser.parse_args()
    predict(
        image_path=args.image,
        model_path=args.model,
        threshold=args.threshold,
        output_obj=args.output,
        visualize=args.gui
    )
