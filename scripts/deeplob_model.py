"""
deeplob_model.py
================
DeepLOB implementation in PyTorch.

Reference: Zhang, Zohren, Roberts (2019)
"DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
https://arxiv.org/abs/1808.03668

Architecture summary:
  Input:  (batch, 1, T=100, D=40) -- treat LOB snapshot sequence as a 2D image
  Block 1: 3 x Conv2D (spatial LOB structure extraction per level)
  Block 2: 1 x Conv2D + MaxPool (inception-style temporal compression)
  Block 3: LSTM (sequential dependencies across time)
  Output: softmax over 3 classes (DOWN=0, FLAT=1, UP=2)

Dimension notes for YOUR data (T=100, D=40):
  D=40 = 10 levels * 4 features (ask_p, ask_s, bid_p, bid_s)
  Conv kernel (1, 2) slides along the feature axis, pairing ask/bid per level.
  Conv kernel (4, 1) slides along the time axis in the inception block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLOB(nn.Module):
    """
    DeepLOB for 3-class LOB mid-price movement prediction.

    Input tensor shape: (batch_size, 1, T, D)
      batch_size: number of samples
      1:          single channel (treat as grayscale image)
      T:          time window (100 snapshots)
      D:          feature dimension (40)
    """

    def __init__(self, num_classes: int = 3, T: int = 100, D: int = 40):
        super(DeepLOB, self).__init__()

        self.T = T
        self.D = D

        # ---- Block 1: Spatial feature extraction (LOB structure) ----
        # Kernel (1, 2): slides along feature axis, pairs ask+bid per level
        # This is the key spatial prior from the paper:
        # adjacent columns in the feature matrix are (ask, bid) pairs at same level
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),  # (B, 32, T, 20)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)), # (B, 32, T, 10)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(1, 10)),               # (B, 32, T, 1)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        # After block 1: (B, 32, T, 1)

        # ---- Block 2: Inception-style temporal compression ----
        # Three parallel paths with different temporal receptive fields
        # Then concatenate and pool
        self.inception_1x1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inception_3x1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inception_5x1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inception_maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        # After inception concat: (B, 256, T, 1)
        # MaxPool over time: (B, 256, T//2, 1) -- reduces sequence for LSTM

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        # After pool: (B, 256, T//2, 1) = (B, 256, 50, 1)

        # ---- Block 3: LSTM for temporal sequence modeling ----
        # Reshape to (B, T//2, 256) before LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # ---- Output head ----
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, T, D)
        returns: (batch, num_classes) -- raw logits (apply softmax for probs)
        """
        # Block 1: spatial
        x = self.conv1(x)    # (B, 32, T, 1)

        # Block 2: inception (all branches operate on same input)
        b1 = self.inception_1x1(x)
        b2 = self.inception_3x1(x)
        b3 = self.inception_5x1(x)
        b4 = self.inception_maxpool(x)
        x = torch.cat([b1, b2, b3, b4], dim=1)  # (B, 256, T, 1)
        x = self.pool(x)                          # (B, 256, T//2, 1)

        # Reshape for LSTM: (B, T//2, 256)
        B, C, Tred, _ = x.shape
        x = x.squeeze(-1).permute(0, 2, 1)       # (B, T//2, C)

        # Block 3: LSTM -- take last hidden state
        x, (h_n, _) = self.lstm(x)
        x = x[:, -1, :]                           # (B, 64) -- last timestep

        # Output
        x = self.fc(x)                            # (B, 3)
        return x


def get_model(device: torch.device = None) -> DeepLOB:
    """Instantiate and return DeepLOB model on specified device."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLOB(num_classes=3, T=100, D=40)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape verification -- run this to confirm dimensions before training
    device = torch.device('cpu')
    model = get_model(device)

    print("DeepLOB Architecture:")
    print(f"  Trainable parameters: {count_parameters(model):,}")

    # Test forward pass with correct input shape
    dummy = torch.zeros(8, 1, 100, 40)  # batch=8, channel=1, T=100, D=40
    out   = model(dummy)
    print(f"  Input shape:  {tuple(dummy.shape)}")
    print(f"  Output shape: {tuple(out.shape)}  (should be [8, 3])")
    assert out.shape == (8, 3), f"Output shape mismatch: {out.shape}"
    print("  Forward pass: OK")