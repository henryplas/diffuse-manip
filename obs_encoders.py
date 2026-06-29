"""
obs_encoders.py — observation encoders for Diffusion Policy.

M1 (low-dim): LowDimEncoder flattens (To, obs_dim) -> fixed-size conditioning vector.
M2 (image):   ImageEncoder is a stub; see the TODO comment for what to implement.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LowDimEncoder(nn.Module):
    """Encode a stack of low-dim observations into a conditioning vector.

    Input:  (B, To, obs_dim)  — last To normalized observations
    Output: (B, out_dim)      — global conditioning vector for the U-Net
    """

    def __init__(
        self,
        obs_dim: int,
        obs_horizon: int,
        out_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # (B, To, obs_dim) -> (B, To*obs_dim) -> (B, out_dim)
        return self.net(obs.flatten(start_dim=1))


class ImageEncoder(nn.Module):
    """M2 stub: spatial-softmax ResNet-18 per camera view.

    TODO (M2):
      - One ImageEncoder instance per camera (agentview RGB + wrist RGB).
      - Replace the global-average-pool head with a SpatialSoftmax layer
        (learnable keypoint detection; robomimic's VisualCore is the reference).
      - Concatenate features from all camera views, project to out_dim.
      - Images: 84x84 or 128x128, uint8 from robosuite's offscreen renderer;
        normalise to [-1, 1] before encoding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "ImageEncoder is a M2 milestone stub. "
            "Implement with a spatial-softmax ResNet-18 (see class docstring)."
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    enc = LowDimEncoder(obs_dim=23, obs_horizon=2, out_dim=256)
    x = torch.randn(4, 2, 23)
    out = enc(x)
    assert out.shape == (4, 256), out.shape
    print(f"LowDimEncoder OK: {tuple(x.shape)} -> {tuple(out.shape)}")
    print("ALL CHECKS PASSED")
