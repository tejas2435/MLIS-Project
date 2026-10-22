import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def read_binvox(path):
    """
    Robust binvox reader for 32³ voxel models.
    Supports standard binvox formats with varying header lines.
    """
    with open(path, 'rb') as f:
        line = f.readline().strip()
        if not line.startswith(b'#binvox'):
            raise ValueError(f"Not a binvox file: {path}")

        dims = [32, 32, 32]
        translate = [0.0, 0.0, 0.0]
        scale = 1.0

        while True:
            line = f.readline().strip()
            if not line:
                continue
            if line.startswith(b'data'):
                break
            elif line.startswith(b'dim'):
                dims = [int(x) for x in line.split()[1:]]
            elif line.startswith(b'translate'):
                translate = [float(x) for x in line.split()[1:]]
            elif line.startswith(b'scale'):
                scale = float(line.split()[1])

        raw = np.frombuffer(f.read(), dtype=np.uint8)

    values, counts = raw[::2], raw[1::2]
    data = np.repeat(values, counts).astype(np.float32)
    expected_size = np.prod(dims)
    if data.size != expected_size:
        if data.size < expected_size:
            data = np.pad(data, (0, expected_size - data.size))
        else:
            data = data[:expected_size]

    data = data.reshape(dims)
    return data  # (32, 32, 32)

def write_binvox(path, voxel_array):
    """
    Writes a (32, 32, 32) numpy array (0 or 1) to a valid binvox file.
    Uses Run-Length Encoding (RLE).
    """
    dims = voxel_array.shape
    flat = voxel_array.flatten().astype(np.uint8)

    bytes_list = bytearray()
    if len(flat) > 0:
        current_val = flat[0]
        count = 0
        for val in flat:
            if val == current_val and count < 255:
                count += 1
            else:
                bytes_list.append(current_val)
                bytes_list.append(count)
                current_val = val
                count = 1
        bytes_list.append(current_val)
        bytes_list.append(count)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b"#binvox 1\n")
        f.write(f"dim {dims[0]} {dims[1]} {dims[2]}\n".encode("ascii"))
        f.write(b"translate 0 0 0\n")
        f.write(b"scale 1.0\n")
        f.write(b"data\n")
        f.write(bytes_list)

class ShapeNetVoxelDataset(Dataset):
    def __init__(self, root="data/ShapeNet", split="train", img_size=128, synsets=None, max_samples=None, use_cache=True):
        """
        ShapeNet 3D Voxel Dataset.
        root: directory containing ShapeNetRendering and ShapeNetVox32
        split: 'train' (80%) or 'val' (20%)
        synsets: optional list of synset strings (e.g. ['03001627', '02691156']) to filter classes
        max_samples: optional cap on total pairs for fast training / prototyping
        """
        # Resolve path whether run from image2voxel/ or workspace root
        if not os.path.exists(root) and os.path.exists(os.path.join("image2voxel", root)):
            root = os.path.join("image2voxel", root)
        elif not os.path.exists(root) and os.path.exists(os.path.join("..", root)):
            root = os.path.join("..", root)

        self.root = root
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        rendering_root = os.path.join(root, "ShapeNetRendering")
        voxel_root = os.path.join(root, "ShapeNetVox32")
        cache_file = os.path.join(root, "dataset_pairs_cache.json")

        all_pairs = []

        # 1. Load from cache if available
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    all_pairs = json.load(f)
            except Exception:
                all_pairs = []

        # 2. Build cache using ultra-fast set intersections if not cached
        if not all_pairs and os.path.isdir(rendering_root) and os.path.isdir(voxel_root):
            print(f"Indexing ShapeNet dataset at {root} (this runs once and caches)...")
            vox_synsets = set(os.listdir(voxel_root))
            rend_synsets = set(os.listdir(rendering_root))
            common_synsets = sorted(list(vox_synsets & rend_synsets))

            for s in common_synsets:
                s_vox = os.path.join(voxel_root, s)
                s_rend = os.path.join(rendering_root, s)
                if not (os.path.isdir(s_vox) and os.path.isdir(s_rend)):
                    continue

                common_models = sorted(list(set(os.listdir(s_vox)) & set(os.listdir(s_rend))))
                for m in common_models:
                    img_path = os.path.join(s_rend, m, "rendering", "00.png")
                    vox_path = os.path.join(s_vox, m, "model.binvox")
                    # Quick path record
                    all_pairs.append([img_path, vox_path, s])

            # Save cache
            try:
                with open(cache_file, "w") as f:
                    json.dump(all_pairs, f)
                print(f"Cached {len(all_pairs)} dataset pairs to {cache_file}")
            except Exception as e:
                print(f"Could not write cache: {e}")

        # 3. Filter by synsets if requested
        if synsets:
            synset_set = set(synsets)
            all_pairs = [p for p in all_pairs if p[2] in synset_set]

        # 4. Filter max_samples if requested
        if max_samples and max_samples < len(all_pairs):
            # Deterministic selection
            all_pairs = all_pairs[:max_samples]

        self.samples = [(p[0], p[1]) for p in all_pairs]

        # 5. Split train/val
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(0.8 * n)]
        else:
            self.samples = self.samples[int(0.8 * n):]

        print(f"[{split.upper()}] Loaded {len(self.samples)} sample pairs from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, vox_path = self.samples[idx]

        # Safety check for file existence
        if not os.path.exists(img_path):
            # Attempt to resolve relative to dataset root
            img_path = os.path.join(self.root, os.path.relpath(img_path, "data/ShapeNet"))
        if not os.path.exists(vox_path):
            vox_path = os.path.join(self.root, os.path.relpath(vox_path, "data/ShapeNet"))

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        vox = read_binvox(vox_path)               # (32, 32, 32)
        vox = torch.from_numpy(vox).unsqueeze(0)  # (1, 32, 32, 32)

        return img, vox

if __name__ == "__main__":
    ds = ShapeNetVoxelDataset(split="train")
    print(f"Total train items: {len(ds)}")
    if len(ds) > 0:
        img, vox = ds[0]
        print(f"Sample 0 -> Image shape: {img.shape}, Voxel shape: {vox.shape}, Voxel sum: {vox.sum()}")
