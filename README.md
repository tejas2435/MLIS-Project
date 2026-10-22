# Image2Voxel: Single-Image 3D Voxel Reconstruction (32³ Resolution)

An end-to-end PyTorch deep learning pipeline that reconstructs a **32×32×32 3D voxel grid** from a single **2D RGB image** ($128 \times 128$) and extracts textured 3D surface meshes (`.obj`) via Marching Cubes.

---

## Quickstart (For Demonstrations & Faculty Evaluation)

### 1. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_FOLDER>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Runs on both NVIDIA GPU with CUDA and standard CPU).*

### 3. Run Instant 3D Reconstruction Demo
Run inference on any provided demo sample:
```bash
python infer.py --image demo_samples/airplane.png --gui
```
Or for a chair:
```bash
python infer.py --image demo_samples/chair.png --gui
```

### What You Will See:
1. **Interactive 3D Window (Open3D)**: Opens a live 3D window where you can rotate, pan, and zoom into the reconstructed 3D model.
2. **`output.obj`**: Exported 3D wavefront triangle mesh (openable directly in **Windows 3D Viewer**, Blender, or MeshLab).
3. **`output_preview.png`**: Side-by-side comparison showing the 2D input image alongside the reconstructed 32³ voxel occupancy grid.

---

## Interactive 3D Viewer Controls
- **Rotate**: Left-click + drag
- **Pan / Move**: Right-click + drag (or `Shift` + Left-click + drag)
- **Zoom**: Mouse scroll wheel
- **Close Viewer**: Press `Q` or `Esc` (or click `X`)

---

## Project Structure

```
├── model.py                 # Custom CNN Encoder-Decoder architecture
├── infer.py                 # Inference script with Marching Cubes & OBJ exporter
├── train.py                 # Training pipeline (BCE + Soft Dice Loss)
├── dataset.py               # Dataset loader & RLE .binvox parser
├── best_model_fp16.pth      # Pretrained weights (50 MB, included in repository)
├── requirements.txt         # Dependencies
├── demo_samples/            # Ready-to-use sample images for testing
│   ├── airplane.png
│   ├── chair.png
│   └── triangle.png
├── output.obj               # Latest exported 3D mesh
└── output_preview.png       # Latest visual reconstruction preview
```

---

## Architecture Overview

- **Input:** $128 \times 128 \times 3$ RGB image
- **2D CNN Encoder:** 5 convolutional layers downsampling $128 \times 128 \times 3 \rightarrow 4 \times 4 \times 512$
- **Bottleneck Projection:** Fully connected layer projecting $8192 \rightarrow 2048$, reshaped to $(256, 2, 2, 2)$
- **3D CNN Decoder:** 4 transposed convolution layers upsampling $2 \times 2 \times 2 \times 256 \rightarrow 32 \times 32 \times 32 \times 1$
- **Isosurface Extraction:** Marching Cubes algorithm converts voxel occupancy probabilities into a smooth polygonal 3D surface mesh (`.obj`).

---

## Training (Optional)

To train on your own dataset or the full 43,795 ShapeNet dataset:

```bash
python train.py --epochs 50 --batch_size 16 --lr 0.0001 --num_workers 0
```
- On Windows, `--num_workers 0` is recommended to avoid shared memory paging errors.
- Pretrained weights are automatically saved to `best_model.pth` whenever validation IoU improves.
