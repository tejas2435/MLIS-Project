import os
import numpy as np
from PIL import Image, ImageDraw
from dataset import write_binvox

def create_sample_dataset(base_dir="data/ShapeNet"):
    """
    Creates a small synthetic dataset of 3D geometric shapes (sphere, cube, cylinder)
    with 2D rendered views and corresponding 32x32x32 binvox ground truth.
    Allows testing train.py and infer.py immediately without waiting for multi-gigabyte downloads.
    """
    shapes = ["02691156", "02958343", "03001627"]  # Synthetic synset IDs
    models = ["model_01", "model_02", "model_03", "model_04"]

    grid_size = 32
    rendering_base = os.path.join(base_dir, "ShapeNetRendering")
    vox_base = os.path.join(base_dir, "ShapeNetVox32")

    for syn_idx, syn in enumerate(shapes):
        for mod_idx, mod in enumerate(models):
            # Create voxel grid (32, 32, 32)
            vox = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
            z, y, x = np.indices((grid_size, grid_size, grid_size))

            # Generate shape based on indices
            shape_type = (syn_idx + mod_idx) % 3
            center = grid_size // 2

            if shape_type == 0:
                # 3D Sphere
                radius = 10
                mask = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= radius**2
                vox[mask] = 1
                color = (40, 120, 220)
            elif shape_type == 1:
                # 3D Cube
                half_side = 8
                mask = (np.abs(x - center) <= half_side) & (np.abs(y - center) <= half_side) & (np.abs(z - center) <= half_side)
                vox[mask] = 1
                color = (220, 70, 40)
            else:
                # 3D Cylinder
                radius = 8
                height = 10
                mask = ((x - center)**2 + (y - center)**2 <= radius**2) & (np.abs(z - center) <= height)
                vox[mask] = 1
                color = (40, 200, 100)

            # 1. Save binvox
            vox_path = os.path.join(vox_base, syn, mod, "model.binvox")
            write_binvox(vox_path, vox)

            # 2. Render 2D image (128x128)
            img_dir = os.path.join(rendering_base, syn, mod, "rendering")
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, "00.png")

            img = Image.new("RGB", (128, 128), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            # Draw synthetic 2D projection
            if shape_type == 0:
                draw.ellipse([34, 34, 94, 94], fill=color, outline=(20, 20, 20), width=2)
            elif shape_type == 1:
                draw.rectangle([34, 34, 94, 94], fill=color, outline=(20, 20, 20), width=2)
            else:
                draw.rounded_rectangle([38, 28, 90, 100], radius=15, fill=color, outline=(20, 20, 20), width=2)

            img.save(img_path)

    # Save a test_image.png in project root
    test_img = Image.new("RGB", (128, 128), color=(245, 245, 245))
    draw = ImageDraw.Draw(test_img)
    draw.ellipse([30, 30, 98, 98], fill=(50, 130, 240), outline=(10, 10, 10), width=2)
    test_img.save("test_image.png")

    print(f"Sample synthetic dataset generated at: {base_dir}")
    print(f"Total model pairs: {len(shapes) * len(models)}")
    print("Created test image: test_image.png")

if __name__ == "__main__":
    create_sample_dataset()
