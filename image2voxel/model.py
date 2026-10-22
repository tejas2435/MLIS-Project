import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageToVoxel(nn.Module):
    """
    Fully custom CNN: 2D image → 32³ voxel occupancy grid.
    
    Architecture:
      Encoder: 5 Conv2d layers with BatchNorm and ReLU downsampling 128x128 -> 4x4x512
      Bottleneck: Linear projection 512*4*4 (8192) -> 2048, reshaped to (256, 2, 2, 2)
      Decoder: 4 ConvTranspose3d layers upsampling 2x2x2 -> 32x32x32x1
      Output: Sigmoid activation producing occupancy probabilities in [0, 1]
    """
    def __init__(self):
        super().__init__()

        # ---------- 2D Encoder ----------
        self.encoder = nn.Sequential(
            # 128x128x3 → 64x64x64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x64x64 → 32x32x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32x32x128 → 16x16x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 16x16x256 → 8x8x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 8x8x512 → 4x4x512
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Bottleneck → reshape to 3D
        self.fc = nn.Linear(512 * 4 * 4, 2048)

        # ---------- 3D Decoder ----------
        self.decoder = nn.Sequential(
            # 2x2x2x256 → 4x4x4x128
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # 4x4x4x128 → 8x8x8x64
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # 8x8x8x64 → 16x16x16x32
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # 16x16x16x32 → 32x32x32x1
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # x: (B, 3, 128, 128)
        feat = self.encoder(x)                     # (B, 512, 4, 4)
        feat = feat.view(feat.size(0), -1)         # (B, 512*4*4)
        latent = self.fc(feat)                     # (B, 2048)
        latent = latent.view(-1, 256, 2, 2, 2)     # (B, 256, 2, 2, 2)

        out = self.decoder(latent)                 # (B, 1, 32, 32, 32)
        return torch.sigmoid(out)                  # probability occupancy

if __name__ == "__main__":
    model = ImageToVoxel()
    dummy_img = torch.randn(2, 3, 128, 128)
    out = model(dummy_img)
    print(f"Input shape:  {dummy_img.shape}")
    print(f"Output shape: {out.shape}")
    print("Model forward pass successful!")
