# Image2Voxel: Single-Image 3D Voxel Reconstruction (32³ Resolution)

A PyTorch deep learning project that takes a single 2D RGB image ($128 \times 128$) and predicts a $32 \times 32 \times 32$ 3D voxel occupancy grid, with Marching Cubes isosurface extraction to export textured 3D meshes (`.obj`).

---

## 1. Project Structure

```
image2voxel/
├── dataset.py               # ShapeNet dataset loader & RLE .binvox parser
├── model.py                 # Custom CNN Encoder-Decoder (2D Conv -> 3D ConvTranspose)
├── train.py                 # Training pipeline with BCE + Dice loss & validation IoU
├── infer.py                 # Single-image inference & Marching Cubes 3D OBJ exporter
├── download_data.py         # Automated downloader & extractor for ShapeNet
├── generate_sample_data.py  # Quick synthetic dataset generator for immediate testing
├── requirements.txt         # Dependencies
├── test_image.png           # Sample input image
└── output.obj               # Exported 3D mesh
```

---

## 2. Architecture Overview

- **Input:** $128 \times 128 \times 3$ RGB image
- **2D Encoder:** 5-stage convolutional downsampling with BatchNorm and ReLU:
  - `Conv2d(3 -> 64, k=4, s=2, p=1)` $\rightarrow 64 \times 64 \times 64$
  - `Conv2d(64 -> 128, k=4, s=2, p=1)` $\rightarrow 32 \times 32 \times 128$
  - `Conv2d(128 -> 256, k=4, s=2, p=1)` $\rightarrow 16 \times 16 \times 256$
  - `Conv2d(256 -> 512, k=4, s=2, p=1)` $\rightarrow 8 \times 8 \times 512$
  - `Conv2d(512 -> 512, k=4, s=2, p=1)` $\rightarrow 4 \times 4 \times 512$
- **Bottleneck Projection:**
  - `Linear(8192 -> 2048)` $\rightarrow$ Reshaped to 3D feature volume $256 \times 2 \times 2 \times 2$
- **3D Decoder:** 4-stage 3D transposed convolution upsampling:
  - `ConvTranspose3d(256 -> 128, k=4, s=2, p=1)` $\rightarrow 4 \times 4 \times 4 \times 128$
  - `ConvTranspose3d(128 -> 64, k=4, s=2, p=1)` $\rightarrow 8 \times 8 \times 8 \times 64$
  - `ConvTranspose3d(64 -> 32, k=4, s=2, p=1)` $\rightarrow 16 \times 16 \times 16 \times 32$
  - `ConvTranspose3d(32 -> 1, k=4, s=2, p=1)` $\rightarrow 32 \times 32 \times 32 \times 1$
- **Output:** Sigmoid activation producing voxel occupancy probability in $[0, 1]$.

---

## 3. Installation

All required libraries (`torch` with CUDA, `torchvision`, `open3d`, `scikit-image`, `numpy`, `pillow`, `tqdm`, `matplotlib`) are installed and verified:

```bash
pip install -r requirements.txt
```

---

## 4. Dataset Setup

### Option A: Official 3D-R2N2 ShapeNet Dataset
Download and extract the official Stanford CVGL dataset:
```bash
python download_data.py
```
Or manually via terminal / curl:
```bash
mkdir -p data/ShapeNet
cd data/ShapeNet

# Multi-view renderings (~2.1 GB)
curl -O http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
tar -xzf ShapeNetRendering.tgz

# 32³ Voxel ground truth (.binvox) (~230 MB)
curl -O http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
tar -xzf ShapeNetVox32.tgz
```

Expected directory tree:
```
data/ShapeNet/
├── ShapeNetRendering/
│   └── <synset_id>/<model_id>/rendering/00.png ... 23.png
└── ShapeNetVox32/
    └── <synset_id>/<model_id>/model.binvox
```

### Option B: Quick Synthetic Test Dataset (Instant Sandbox)
To verify the entire training and inference pipeline immediately without waiting for the large download:
```bash
python generate_sample_data.py
```

---

## 5. Training

Run the training pipeline with BCE loss + Soft Dice loss and validation IoU:

```bash
python train.py --epochs 50 --batch_size 16 --lr 1e-4
```

Key arguments:
- `--data_root`: Path to dataset folder (default: `data/ShapeNet`)
- `--epochs`: Number of epochs (default: `50`)
- `--batch_size`: Batch size (default: `16`)
- `--lr`: Learning rate (default: `0.0001`)
- `--num_workers`: DataLoader worker threads (default: `2` on Windows)
- `--save_path`: Destination checkpoint path (default: `best_model.pth`)

The training script automatically tracks the validation IoU across epochs and saves the best model weights to `best_model.pth`.

---

## 6. Inference (Image → 3D Mesh)

Run reconstruction on any RGB test image:

```bash
python infer.py --image test_image.png --model best_model.pth --threshold 0.5 --output output.obj
```

This will:
1. Predict the $32^3$ voxel grid.
2. Run Marching Cubes to extract the 3D surface mesh.
3. Save the mesh to `output.obj` (readable in Blender, MeshLab, 3D Viewer, etc.).
4. Save a visual reconstruction preview to `output_preview.png`.
5. Open an interactive Open3D 3D viewer window (pass `--no_gui` to run headlessly).
