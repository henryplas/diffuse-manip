"""
diffusion_policy.py — Diffusion Policy: 1-D temporal U-Net + DDPM/DDIM.

Architecture (Chi et al. 2023, CNN variant):

  ConditionalUNet1D
    A 1-D temporal U-Net that operates over the TIME axis of the action
    sequence (length Tp), NOT over image spatial axes. Structurally this is
    your DDPM U-Net with Conv2d replaced by Conv1d — same residual blocks,
    same skip connections, same FiLM conditioning.

    Conditioning path:
      timestep t  ->  SinusoidalPosEmb  ->  2-layer MLP  -> (B, dsed)
      obs         ->  LowDimEncoder (external)            -> (B, obs_cond_dim)
      cond = cat([time_emb, obs_emb])                     -> (B, dsed + obs_cond_dim)
      Each ResidualBlock1D receives `cond` and applies FiLM (scale + shift).

  GaussianDiffusion
    Linear DDPM noise schedule, epsilon-prediction MSE training loss,
    DDIM deterministic inference (10-16 steps at eval time).

Training tips (critical):
  - Use EMA of model weights (see train.py). Without it DP training is unstable.
  - Normalize actions to [-1, 1] before training (done in datasets.py).
  - Prediction horizon Tp=16, obs horizon To=2, action horizon Ta=8 are the
    paper defaults; sweep if results are weak.

Inference (receding-horizon control):
  obs_cond = encoder(last_To_obs)          # (B, obs_cond_dim)
  actions = diffusion.ddim_sample(obs_cond)  # (B, Tp, act_dim), in [-1, 1]
  actions = normalizer.unnormalize_action(actions)
  env.step(actions[:, :Ta])               # execute first Ta, then re-plan
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Timestep embedding
# --------------------------------------------------------------------------- #
class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding (Vaswani et al. 2017)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer diffusion timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


# --------------------------------------------------------------------------- #
# Residual block with FiLM conditioning
# --------------------------------------------------------------------------- #
class ResidualBlock1D(nn.Module):
    """1-D residual block conditioned via FiLM (scale + shift from cond vector).

    Channels flow: in_ch -> out_ch (via Conv1d).
    Residual path uses a 1x1 Conv1d when in_ch != out_ch.
    FiLM is applied after the first GroupNorm, before the first activation.
    n_groups must divide out_ch.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(n_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(n_groups, out_ch)
        self.act = nn.SiLU()
        # FiLM: project conditioning to (scale, shift) for out_ch features
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x:    (B, in_ch, T)
        # cond: (B, cond_dim)
        residual = self.residual_conv(x)

        x = self.norm1(self.conv1(x))
        # FiLM scale + shift
        film = self.cond_proj(self.act(cond))          # (B, 2*out_ch)
        scale, shift = film.chunk(2, dim=-1)           # each (B, out_ch)
        x = x * (1 + scale[:, :, None]) + shift[:, :, None]
        x = self.act(x)

        x = self.act(self.norm2(self.conv2(x)))
        return x + residual


# --------------------------------------------------------------------------- #
# 1-D temporal U-Net
# --------------------------------------------------------------------------- #
class ConditionalUNet1D(nn.Module):
    """1-D temporal U-Net for Diffusion Policy action sequence denoising.

    Input:
      noisy_actions : (B, Tp, act_dim)  — noisy action sequence at timestep t
      timesteps     : (B,)              — integer diffusion timestep
      global_cond   : (B, obs_cond_dim) — obs conditioning from LowDimEncoder

    Output:
      predicted_noise : (B, Tp, act_dim)

    Architecture:
      Encoder: n-1 levels, each with 2 ResBlocks + stride-2 Conv1d downsample.
      Bottleneck: 2 ResBlocks (no spatial change).
      Decoder: n-1 levels, each with ConvTranspose1d upsample + 2 ResBlocks
               fed the concatenated skip connection from the encoder.
      Final: 1x1 Conv1d -> act_dim.

    n_groups must divide every value in down_dims.
    """

    def __init__(
        self,
        act_dim: int,
        obs_cond_dim: int,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 256,
    ):
        super().__init__()
        # Combined conditioning dim: time embedding + obs embedding
        cond_dim = diffusion_step_embed_dim + obs_cond_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Channel pairs: [(act_dim, 256), (256, 512), (512, 1024)]
        in_out = list(zip([act_dim, *down_dims[:-1]], down_dims))
        encoder_pairs = in_out[:-1]           # all but the bottleneck
        bottleneck_in, bottleneck_out = in_out[-1]

        # Encoder
        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock1D(ic, oc, cond_dim, kernel_size, n_groups),
                ResidualBlock1D(oc, oc, cond_dim, kernel_size, n_groups),
                nn.Conv1d(oc, oc, 3, stride=2, padding=1),  # halve T
            ])
            for ic, oc in encoder_pairs
        ])

        # Bottleneck
        self.mid_modules = nn.ModuleList([
            ResidualBlock1D(bottleneck_in, bottleneck_out, cond_dim, kernel_size, n_groups),
            ResidualBlock1D(bottleneck_out, bottleneck_out, cond_dim, kernel_size, n_groups),
        ])

        # Decoder (reversed encoder, skip channels = oc of each encoder level)
        ch = bottleneck_out
        decoder = []
        for ic, oc in reversed(encoder_pairs):
            decoder.append(nn.ModuleList([
                nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1),  # double T
                ResidualBlock1D(ch + oc, oc, cond_dim, kernel_size, n_groups),
                ResidualBlock1D(oc, oc, cond_dim, kernel_size, n_groups),
            ]))
            ch = oc
        self.up_modules = nn.ModuleList(decoder)

        self.final_conv = nn.Conv1d(ch, act_dim, 1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        # noisy_actions: (B, Tp, act_dim) -> permute -> (B, act_dim, Tp)
        x = noisy_actions.permute(0, 2, 1)

        time_emb = self.time_mlp(timesteps)                    # (B, dsed)
        cond = torch.cat([time_emb, global_cond], dim=-1)      # (B, cond_dim)

        # Encoder
        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        for res in self.mid_modules:
            x = res(x, cond)

        # Decoder
        for upsample, res1, res2 in self.up_modules:
            x = upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)

        return self.final_conv(x).permute(0, 2, 1)  # (B, Tp, act_dim)


# --------------------------------------------------------------------------- #
# DDPM noise schedule + training loss + DDIM inference
# --------------------------------------------------------------------------- #
class GaussianDiffusion(nn.Module):
    """Wraps ConditionalUNet1D with a linear DDPM noise schedule.

    Training:  .loss(x0, global_cond)  -> scalar MSE loss on noise prediction.
    Inference: .ddim_sample(global_cond, n_steps) -> (B, Tp, act_dim) in [-1,1].
    """

    def __init__(
        self,
        model: ConditionalUNet1D,
        pred_horizon: int,
        act_dim: int,
        n_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.n_timesteps = n_timesteps

        # Linear noise schedule
        betas = torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    # ---- forward diffusion ------------------------------------------------ #
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample q(x_t | x_0) = sqrt(a_bar_t)*x0 + sqrt(1-a_bar_t)*eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    # ---- training loss ---------------------------------------------------- #
    def loss(self, x0: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """MSE loss on epsilon prediction.

        x0          : (B, Tp, act_dim)  normalized action sequences
        global_cond : (B, obs_cond_dim) from LowDimEncoder
        """
        B = x0.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt, noise = self.q_sample(x0, t, noise)
        noise_pred = self.model(xt, t, global_cond)
        return F.mse_loss(noise_pred, noise)

    # ---- DDIM inference --------------------------------------------------- #
    @torch.no_grad()
    def ddim_sample(
        self,
        global_cond: torch.Tensor,
        n_steps: int = 16,
    ) -> torch.Tensor:
        """Deterministic DDIM denoising (eta=0).

        global_cond : (B, obs_cond_dim)
        Returns     : (B, Tp, act_dim) in [-1, 1]
        """
        B, device = global_cond.shape[0], global_cond.device

        x = torch.randn(B, self.pred_horizon, self.act_dim, device=device)

        # Uniformly spaced timesteps from T-1 down to 0
        ts = torch.linspace(
            self.n_timesteps - 1, 0, n_steps, dtype=torch.long, device=device
        )

        for i, t_val in enumerate(ts):
            t = t_val.expand(B)
            abar_t = self.alphas_cumprod[t_val].reshape(1, 1, 1)

            # alpha_bar for the previous (less noisy) step; 1.0 at the last step
            abar_prev = (
                self.alphas_cumprod[ts[i + 1]].reshape(1, 1, 1)
                if i + 1 < n_steps
                else torch.ones(1, 1, 1, device=device)
            )

            eps_pred = self.model(x, t, global_cond)
            x0_pred = (x - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM update: deterministic direction toward x0, scaled by abar_prev
            x = abar_prev.sqrt() * x0_pred + (1 - abar_prev).sqrt() * eps_pred

        return x  # (B, Tp, act_dim)


# --------------------------------------------------------------------------- #
# Convenience constructor
# --------------------------------------------------------------------------- #
def build_diffusion_policy(
    obs_dim: int,
    obs_horizon: int,
    obs_cond_dim: int,
    act_dim: int,
    pred_horizon: int,
    down_dims: Sequence[int] = (256, 512, 1024),
    n_timesteps: int = 100,
) -> tuple[ConditionalUNet1D, GaussianDiffusion]:
    """Build a (unet, diffusion) pair ready for training.

    Call obs_encoder = LowDimEncoder(obs_dim, obs_horizon, obs_cond_dim)
    separately; pass its output as global_cond to diffusion.loss / ddim_sample.
    """
    unet = ConditionalUNet1D(
        act_dim=act_dim,
        obs_cond_dim=obs_cond_dim,
        down_dims=down_dims,
    )
    diffusion = GaussianDiffusion(
        model=unet,
        pred_horizon=pred_horizon,
        act_dim=act_dim,
        n_timesteps=n_timesteps,
    )
    return unet, diffusion


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    Tp, To, act_dim, obs_dim, obs_cond_dim = 16, 2, 7, 23, 256
    B = 4

    unet, diffusion = build_diffusion_policy(
        obs_dim=obs_dim,
        obs_horizon=To,
        obs_cond_dim=obs_cond_dim,
        act_dim=act_dim,
        pred_horizon=Tp,
    )

    obs_cond = torch.randn(B, obs_cond_dim)
    x0 = torch.randn(B, Tp, act_dim)

    loss = diffusion.loss(x0, obs_cond)
    print(f"loss={loss.item():.4f}  (expected ~1.0 at random init)")

    samples = diffusion.ddim_sample(obs_cond, n_steps=16)
    assert samples.shape == (B, Tp, act_dim), samples.shape
    print(f"ddim_sample shape={tuple(samples.shape)}  "
          f"range=[{samples.min():.3f}, {samples.max():.3f}]")
    print("ALL CHECKS PASSED")
