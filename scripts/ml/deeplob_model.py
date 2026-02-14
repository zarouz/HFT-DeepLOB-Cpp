"""
deeplob_model.py
================
Two models:

  DeepLOB (Config A):
    Zhang et al. (2019) CNN+LSTM architecture.
    4-branch inception, 256 channels, stride fix applied.
    Input: (B, 1, 100, 40) → Output: (B, 3)

  HawkesTransformer (Configs B-F):
    Dual-stream architecture per Research Plan Section 4.1.
    LOB stream  : Transformer encoder d=128, 4 heads, 2 layers
                  with timestamp-aware positional encoding
    Hawkes stream: 1D-Conv (kernel=3, ch=32) over 4 Hawkes features
    Fusion      : Cross-attention (LOB as query, Hawkes as key+value)
    Configs B/B2 use D_HAWKES=0 (LOB only / LOB+hand features).
    Configs C-F use D_HAWKES=4 (full Hawkes features).
    Input: (B, 100, 40+[0|3|4]) → Output: (B, 3)
           [optional: timestamp (B, 100) for positional encoding]
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.dirname(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    WINDOW_SIZE, D_FEATURES, D_HAWKES, D_TOTAL,
    DEEPLOB_INCEPTION_CHANNELS, DEEPLOB_LSTM_HIDDEN, DEEPLOB_LSTM_LAYERS,
    TRANSFORMER_D_MODEL, TRANSFORMER_N_HEADS, TRANSFORMER_N_LAYERS,
    HAWKES_CONV_CHANNELS, HAWKES_CONV_KERNEL,
    NUM_CLASSES,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Config A: DeepLOB (Zhang et al. 2019)
# ═══════════════════════════════════════════════════════════════════════════════

class _InceptionModule(nn.Module):
    """Single inception block with 4 parallel branches."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class DeepLOB(nn.Module):
    """
    DeepLOB (Config A baseline).
    Input : (B, 1, 100, 40)  -- 1 channel, T=100 timesteps, F=40 features
    Output: (B, 3)
    """
    def __init__(self, inception_channels: int = DEEPLOB_INCEPTION_CHANNELS,
                 lstm_hidden: int = DEEPLOB_LSTM_HIDDEN,
                 lstm_layers: int = DEEPLOB_LSTM_LAYERS,
                 n_classes: int = NUM_CLASSES,
                 dropout: float = 0.1):
        super().__init__()
        ch = inception_channels // 4   # each branch produces ch channels → 4×ch total

        # Spatial CNN block
        # Input:  (B, 1,  T=100, F=40)
        # After Conv2d(1,32, k=(1,2), stride=(1,2)):  (B, 32, T=100, F=20)
        # After Conv2d(32,32, k=(4,1)):               (B, 32, T=97,  F=20)
        # After Conv2d(32,32, k=(4,1)):               (B, 32, T=94,  F=20)
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm2d(32),
        )

        # Inception blocks preserve T and F dims (same-padding on T, no F change)
        # After inception ×3: (B, 256, T=94, F=20)
        self.inception1 = _InceptionModule(32, ch)
        self.inception2 = _InceptionModule(inception_channels, ch)
        self.inception3 = _InceptionModule(inception_channels, ch)

        # Collapse the F dimension to 1 with a 1×F conv (Zhang et al. Fig 2 step)
        # (B, 256, T=94, F=20) → (B, 256, T=94, F=1)
        self.f_collapse = nn.Conv2d(inception_channels, inception_channels,
                                    kernel_size=(1, 20))

        # Pool temporal dimension: (B, 256, T=94, 1) → (B, 256, T=47, 1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # LSTM: input is inception_channels (F=1 collapsed away)
        lstm_in = inception_channels
        self.lstm = nn.LSTM(lstm_in, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, T, F) or (B, T, F) -- both accepted.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)   # (B, T, F) → (B, 1, T, F)

        # CNN spatial
        x = self.spatial(x)                      # (B, 32, T', 20)

        # Inception
        x = self.inception1(x)                   # (B, 256, T', 20)
        x = self.inception2(x)
        x = self.inception3(x)

        # Collapse feature dimension: (B, 256, T', 20) → (B, 256, T', 1)
        x = self.f_collapse(x)
        x = torch.relu(x)

        # Pool temporal: (B, 256, T', 1) → (B, 256, T'//2, 1)
        x = self.pool(x)

        # Reshape for LSTM: (B, 256, T'//2, 1) → (B, T'//2, 256)
        B, C, T, _ = x.shape
        x = x.squeeze(-1).permute(0, 2, 1)       # (B, T, C)

        # LSTM
        x, _ = self.lstm(x)                      # (B, T, lstm_hidden)
        x = self.dropout(x[:, -1, :])            # last timestep
        return self.fc(x)                        # (B, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  Configs B-F: Dual-Stream Hawkes Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class _TimestampPositionalEncoding(nn.Module):
    """
    Continuous positional encoding using actual Unix timestamps.
    Encodes event density implicitly (Research Plan Section 4.1).
    Dense projection of normalised inter-event time onto d_model/2 sin/cos features.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Learned frequency scales
        self.freq_proj = nn.Linear(1, d_model // 2, bias=False)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        """
        x          : (B, T, d_model)
        timestamps : (B, T) unix seconds (optional -- falls back to index PE)
        """
        if timestamps is None:
            # Fall back to standard sinusoidal index PE
            T = x.size(1)
            pos = torch.arange(T, dtype=x.dtype, device=x.device).unsqueeze(0)  # (1, T)
            timestamps = pos

        # Normalise inter-event times to [0, 1] range
        # Use the time deltas (inter-arrival) rather than absolute time
        dt = torch.diff(timestamps, dim=1, prepend=timestamps[:, :1])  # (B, T)
        dt_norm = (dt / (dt.max(dim=1, keepdim=True).values + 1e-8)).unsqueeze(-1)  # (B, T, 1)

        # Project to d_model/2 frequencies
        freqs = self.freq_proj(dt_norm)                  # (B, T, d_model//2)
        pe = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # (B, T, d_model)

        return x + pe


class _LOBEncoder(nn.Module):
    """
    Transformer encoder for the 40-dim LOB stream.
    d_model=128, 4 heads, 2 layers.
    """
    def __init__(self, d_in: int = D_FEATURES,
                 d_model: int = TRANSFORMER_D_MODEL,
                 n_heads: int = TRANSFORMER_N_HEADS,
                 n_layers: int = TRANSFORMER_N_LAYERS,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_enc    = _TimestampPositionalEncoding(d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        """x: (B, T, d_in) → (B, T, d_model)"""
        x = self.input_proj(x)
        x = self.pos_enc(x, timestamps)
        x = self.encoder(x)
        return self.norm(x)


class _HawkesEncoder(nn.Module):
    """
    Lightweight 1D-Conv encoder for the 4-dim Hawkes feature stream.
    Intentionally small (32 channels) so it doesn't overpower the LOB signal.
    """
    def __init__(self, d_in: int = D_HAWKES,
                 channels: int = HAWKES_CONV_CHANNELS,
                 kernel: int = HAWKES_CONV_KERNEL,
                 dropout: float = 0.1):
        super().__init__()
        # 1D convolution along time axis: (B, T, d_in) → treat as (B, d_in, T)
        self.conv1 = nn.Conv1d(d_in, channels, kernel_size=kernel,
                               padding=kernel // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel,
                               padding=kernel // 2)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.d_out = channels

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D_HAWKES) → (B, T, channels)"""
        h = h.permute(0, 2, 1)          # (B, D_HAWKES, T)
        h = F.gelu(self.norm1(self.conv1(h)))
        h = self.drop(h)
        h = F.gelu(self.norm2(self.conv2(h)))
        return h.permute(0, 2, 1)       # (B, T, channels)


class _CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: LOB as query, Hawkes as key+value.
    The model learns to upweight Hawkes features during bursts.
    Attention weights saved to self.attn_weights for interpretability (Figure 5).
    """
    def __init__(self, d_lob: int, d_hawkes: int, n_heads: int = 4):
        super().__init__()
        # Project Hawkes to same dim as LOB
        self.hawkes_proj = nn.Linear(d_hawkes, d_lob)
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=d_lob, num_heads=n_heads,
            batch_first=True, dropout=0.1,
        )
        self.norm    = nn.LayerNorm(d_lob)
        self.attn_weights = None   # saved for Figure 5 (interpretability)

    def forward(self, lob: torch.Tensor, hawkes: torch.Tensor) -> torch.Tensor:
        """
        lob   : (B, T, d_lob)
        hawkes: (B, T, d_hawkes)
        Returns: (B, T, d_lob)
        """
        h_proj = self.hawkes_proj(hawkes)   # (B, T, d_lob)
        fused, attn_w = self.cross_attn(
            query=lob, key=h_proj, value=h_proj,
            need_weights=True, average_attn_weights=True,
        )
        self.attn_weights = attn_w.detach()  # (B, T, T) -- save for vis
        return self.norm(lob + fused)


class HawkesTransformer(nn.Module):
    """
    Dual-Stream Hawkes Transformer (Configs B-F).

    Input:
      lob_x   : (B, T, D_FEATURES)   always required
      hawkes_x: (B, T, D_HAWKES)     optional -- None for Configs B/B2
      timestamps: (B, T)             optional unix seconds for positional enc

    For Config B  (Transformer baseline, LOB only):     hawkes_x=None
    For Config B2 (Transformer + hand LOB features):    pass hand features
                                                         as lob_x extra cols
    For Configs C-F (full Hawkes):                      pass hawkes_x

    Output: (B, 3) logits
    """
    def __init__(self,
                 d_lob: int = D_FEATURES,
                 d_hawkes: int = D_HAWKES,
                 d_model: int = TRANSFORMER_D_MODEL,
                 n_heads: int = TRANSFORMER_N_HEADS,
                 n_layers: int = TRANSFORMER_N_LAYERS,
                 n_classes: int = NUM_CLASSES,
                 dropout: float = 0.1,
                 use_hawkes: bool = True):
        super().__init__()
        self.use_hawkes = use_hawkes and d_hawkes > 0

        self.lob_encoder = _LOBEncoder(d_lob, d_model, n_heads, n_layers, dropout)

        if self.use_hawkes:
            self.hawkes_encoder = _HawkesEncoder(d_hawkes, HAWKES_CONV_CHANNELS,
                                                  HAWKES_CONV_KERNEL, dropout)
            self.fusion = _CrossAttentionFusion(d_model, HAWKES_CONV_CHANNELS, n_heads)
        else:
            self.hawkes_encoder = None
            self.fusion = None

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, n_classes)

    def forward(self, lob_x: torch.Tensor,
                hawkes_x: torch.Tensor = None,
                timestamps: torch.Tensor = None) -> torch.Tensor:
        """
        lob_x    : (B, T, d_lob)
        hawkes_x : (B, T, d_hawkes) or None
        timestamps: (B, T) or None
        Returns  : (B, n_classes)
        """
        lob_enc = self.lob_encoder(lob_x, timestamps)   # (B, T, d_model)

        if self.use_hawkes and hawkes_x is not None:
            h_enc   = self.hawkes_encoder(hawkes_x)      # (B, T, hawkes_ch)
            lob_enc = self.fusion(lob_enc, h_enc)        # (B, T, d_model)

        # Global average pool over time, then classify
        pooled = lob_enc.mean(dim=1)                     # (B, d_model)
        return self.fc(self.dropout(pooled))


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(config: str = 'A', **kwargs) -> nn.Module:
    """
    Build model by config letter.
    config: 'A'            → DeepLOB baseline
            'B'            → Transformer, LOB only
            'B2'           → Transformer, LOB + hand features (pass d_lob=43)
            'C','D','E','F'→ Full Hawkes Transformer
    """
    config = config.upper()
    if config == 'A':
        return DeepLOB(**kwargs)
    elif config in ('B', 'B2'):
        return HawkesTransformer(use_hawkes=False, **kwargs)
    elif config in ('C', 'D', 'E', 'F'):
        return HawkesTransformer(use_hawkes=True, **kwargs)
    else:
        raise ValueError(f"Unknown config '{config}'. Must be A, B, B2, C, D, E, or F.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from config import DEVICE
    print(f"Device: {DEVICE}")

    # Verify Config A
    model_a = build_model('A').to(DEVICE)
    x_a = torch.randn(4, 1, WINDOW_SIZE, D_FEATURES, device=DEVICE)
    out_a = model_a(x_a)
    assert out_a.shape == (4, NUM_CLASSES), f"Config A output shape wrong: {out_a.shape}"
    print(f"Config A: {count_parameters(model_a):,} params  out={out_a.shape}  ✅")

    # Verify Config E (full Hawkes)
    model_e = build_model('E').to(DEVICE)
    x_lob = torch.randn(4, WINDOW_SIZE, D_FEATURES, device=DEVICE)
    x_haw = torch.randn(4, WINDOW_SIZE, D_HAWKES, device=DEVICE)
    ts    = torch.linspace(1.7e9, 1.7e9 + 100, WINDOW_SIZE).unsqueeze(0).expand(4, -1).to(DEVICE)
    out_e = model_e(x_lob, x_haw, ts)
    assert out_e.shape == (4, NUM_CLASSES), f"Config E output shape wrong: {out_e.shape}"
    print(f"Config E: {count_parameters(model_e):,} params  out={out_e.shape}  ✅")

    # Verify attention weights are saved (for Figure 5)
    assert model_e.fusion.attn_weights is not None
    print(f"  Attention weights shape: {model_e.fusion.attn_weights.shape}")
    print("All model checks passed ✅")
