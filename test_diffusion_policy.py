"""
Unit tests for diffusion_policy.py and obs_encoders.py.
No robosuite/robomimic required — pure PyTorch.
Run: python test_diffusion_policy.py
"""

import torch
import torch.nn as nn
from obs_encoders import LowDimEncoder
from diffusion_policy import (
    SinusoidalPosEmb,
    ResidualBlock1D,
    ConditionalUNet1D,
    GaussianDiffusion,
    build_diffusion_policy,
)

B, Tp, act_dim = 4, 16, 7
obs_dim, obs_horizon, obs_cond_dim = 23, 2, 256
cond_dim = 256 + obs_cond_dim   # dsed + obs_cond_dim (matches default dsed=256)

# ---- 1. SinusoidalPosEmb: shape and no NaN --------------------------------
emb = SinusoidalPosEmb(dim=256)
t = torch.randint(0, 100, (B,))
out = emb(t)
assert out.shape == (B, 256), out.shape
assert torch.isfinite(out).all(), "SinusoidalPosEmb produced NaN/Inf"
print("[1] SinusoidalPosEmb shape + finite values OK")

# ---- 2. ResidualBlock1D: output shape + residual handles channel change ----
T = 16
for in_ch, out_ch in [(7, 256), (256, 256), (512, 256)]:
    blk = ResidualBlock1D(in_ch, out_ch, cond_dim)
    x = torch.randn(B, in_ch, T)
    cond = torch.randn(B, cond_dim)
    y = blk(x, cond)
    assert y.shape == (B, out_ch, T), f"in={in_ch} out={out_ch}: {y.shape}"
print("[2] ResidualBlock1D shapes OK (including channel-change residual)")

# ---- 3. ConditionalUNet1D: output same shape as input ----------------------
unet = ConditionalUNet1D(act_dim=act_dim, obs_cond_dim=obs_cond_dim, down_dims=(256, 512, 1024))
noisy = torch.randn(B, Tp, act_dim)
ts = torch.randint(0, 100, (B,))
cond_vec = torch.randn(B, obs_cond_dim)
pred = unet(noisy, ts, cond_vec)
assert pred.shape == (B, Tp, act_dim), pred.shape
print(f"[3] ConditionalUNet1D forward shape OK: {tuple(pred.shape)}")

# ---- 4. GaussianDiffusion.loss: returns scalar, roughly ~1 at random init --
_, diffusion = build_diffusion_policy(
    obs_dim=obs_dim, obs_horizon=obs_horizon,
    obs_cond_dim=obs_cond_dim, act_dim=act_dim, pred_horizon=Tp,
)
x0 = torch.randn(B, Tp, act_dim)
obs_cond = torch.randn(B, obs_cond_dim)
loss = diffusion.loss(x0, obs_cond)
assert loss.shape == (), f"loss not scalar: {loss.shape}"
assert torch.isfinite(loss), f"loss is NaN/Inf: {loss}"
print(f"[4] GaussianDiffusion.loss scalar OK: loss={loss.item():.4f}")

# ---- 5. q_sample: check noisy sample is in the right range at t=0 and t=99 -
for t_val, expect_clean in [(0, True), (99, False)]:
    t_tensor = torch.full((B,), t_val, dtype=torch.long)
    xt, noise = diffusion.q_sample(x0, t_tensor)
    assert xt.shape == x0.shape
    if expect_clean:
        # at t=0, xt should be very close to x0 (almost no noise added)
        assert torch.allclose(xt, x0, atol=0.05), "t=0: xt should be ~x0"
    else:
        # at t=99, xt should differ significantly from x0
        assert (xt - x0).abs().mean() > 0.1, "t=99: xt should differ from x0"
print("[5] q_sample boundary conditions OK (t=0 clean, t=99 noisy)")

# ---- 6. ddim_sample: output shape and values in a reasonable range ---------
with torch.no_grad():
    samples = diffusion.ddim_sample(obs_cond, n_steps=16)
assert samples.shape == (B, Tp, act_dim), samples.shape
assert torch.isfinite(samples).all(), "ddim_sample produced NaN/Inf"
print(f"[6] ddim_sample shape={tuple(samples.shape)} "
      f"range=[{samples.min():.3f}, {samples.max():.3f}] OK")

# ---- 7. LowDimEncoder: shape + varies with input ---------------------------
enc = LowDimEncoder(obs_dim=obs_dim, obs_horizon=obs_horizon, out_dim=obs_cond_dim)
obs_a = torch.randn(B, obs_horizon, obs_dim)
obs_b = torch.randn(B, obs_horizon, obs_dim)
out_a = enc(obs_a)
out_b = enc(obs_b)
assert out_a.shape == (B, obs_cond_dim), out_a.shape
assert not torch.allclose(out_a, out_b), "Encoder output should vary with input"
print(f"[7] LowDimEncoder shape={tuple(out_a.shape)} and input-sensitive OK")

# ---- 8. Gradient flows through U-Net + encoder (backward pass) -------------
enc2 = LowDimEncoder(obs_dim=obs_dim, obs_horizon=obs_horizon, out_dim=obs_cond_dim)
_, diff2 = build_diffusion_policy(
    obs_dim=obs_dim, obs_horizon=obs_horizon,
    obs_cond_dim=obs_cond_dim, act_dim=act_dim, pred_horizon=Tp,
)
obs_in = torch.randn(B, obs_horizon, obs_dim)
x0_in = torch.randn(B, Tp, act_dim)
cond_in = enc2(obs_in)
loss2 = diff2.loss(x0_in, cond_in)
loss2.backward()
assert all(
    p.grad is not None for p in list(enc2.parameters()) + list(diff2.parameters())
), "Some parameters have no gradient"
print("[8] Backward pass: gradients flow through encoder + U-Net OK")

print("\nALL CHECKS PASSED")
