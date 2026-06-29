"""
demo_diffusion.py — animated GIF showing Diffusion Policy DDIM denoising.

Runs N_SAMPLES independent denoising trajectories from the same observation
conditioning, capturing each DDIM step. Animates the convergence from pure
noise to a set of coherent (if randomly initialized) action sequences.

This illustrates the core mechanism that lets DP handle multimodal action
distributions — unlike BC which averages, each sample follows its own
denoising path to a distinct mode.

Run:  python demo_diffusion.py
Out:  diffuse_manip_demo.gif
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from obs_encoders import LowDimEncoder
from diffusion_policy import build_diffusion_policy

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(7)
np.random.seed(7)

# ── Config ───────────────────────────────────────────────────────────────────
N_SAMPLES      = 8
Tp, act_dim    = 16, 7
obs_dim, To    = 23, 2
obs_cond_dim   = 256
N_DDIM_STEPS   = 16
HOLD_FRAMES    = 4      # extra frames on the final clean result
FPS            = 5
OUT_PATH       = "diffuse_manip_demo.gif"
DPI            = 130

# ── Build model (random init — denoising mechanism, not trained behaviour) ───
enc = LowDimEncoder(obs_dim=obs_dim, obs_horizon=To, out_dim=obs_cond_dim)
_, diffusion = build_diffusion_policy(
    obs_dim=obs_dim, obs_horizon=To,
    obs_cond_dim=obs_cond_dim, act_dim=act_dim, pred_horizon=Tp,
)
enc.eval(); diffusion.eval()

# All N samples share the same observation — shows multimodality (same obs,
# different noise seeds → different trajectory modes).
obs_fixed = torch.randn(1, To, obs_dim)
with torch.no_grad():
    global_cond = enc(obs_fixed).expand(N_SAMPLES, -1)   # (N, obs_cond_dim)

# ── DDIM loop capturing intermediate states ───────────────────────────────────
def run_ddim_with_capture(diffusion, cond, n_steps):
    B, device = cond.shape[0], cond.device
    x = torch.randn(B, diffusion.pred_horizon, diffusion.act_dim, device=device)
    ts = torch.linspace(diffusion.n_timesteps - 1, 0, n_steps, dtype=torch.long)

    snapshots = [x.detach().cpu().numpy().copy()]   # frame 0 = pure noise
    for i, t_val in enumerate(ts):
        t = t_val.expand(B)
        ab  = diffusion.alphas_cumprod[t_val].reshape(1, 1, 1)
        ab_prev = (
            diffusion.alphas_cumprod[ts[i + 1]].reshape(1, 1, 1)
            if i + 1 < n_steps else torch.ones(1, 1, 1)
        )
        with torch.no_grad():
            eps = diffusion.model(x, t, cond)
        x0  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0)
        x   = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        snapshots.append(x.detach().cpu().numpy().copy())

    return snapshots   # list of (N, Tp, act_dim), length = n_steps + 1

print("Running DDIM denoising...")
snapshots = run_ddim_with_capture(diffusion, global_cond, N_DDIM_STEPS)
frames = snapshots + [snapshots[-1]] * HOLD_FRAMES  # hold on final result
n_frames = len(frames)
print(f"  {n_frames} animation frames ready ({N_DDIM_STEPS + 1} denoising + {HOLD_FRAMES} hold)")

# ── Axis limits (stable across all frames) ────────────────────────────────────
all_xy = np.concatenate([f[:, :, :2] for f in snapshots])
pad = 0.4
xlim = (all_xy[:, :, 0].min() - pad, all_xy[:, :, 0].max() + pad)
ylim = (all_xy[:, :, 1].min() - pad, all_xy[:, :, 1].max() + pad)

seq_all = np.concatenate([f[0, :, :3] for f in snapshots])
slim = (seq_all.min() - 0.15, seq_all.max() + 0.15)

# ── Colours ───────────────────────────────────────────────────────────────────
SAMPLE_COLORS = plt.cm.tab10(np.linspace(0, 0.9, N_SAMPLES))
DIM_COLORS    = ["#f78166", "#79c0ff", "#56d364"]
DIM_LABELS    = ["ΔX", "ΔY", "ΔZ"]
BG            = "#0d1117"
PANEL_BG      = "#161b22"
GRID_C        = "#21262d"
TEXT_C        = "#c9d1d9"
MUTED_C       = "#8b949e"
ACCENT_GREEN  = "#56d364"
ACCENT_RED    = "#f78166"

# ── Figure & axes ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.5), facecolor=BG)
gs  = gridspec.GridSpec(
    2, 2,
    left=0.06, right=0.97, top=0.88, bottom=0.10,
    wspace=0.30, hspace=0.45,
    width_ratios=[1.6, 1],
    height_ratios=[1.6, 1],
)
ax_traj = fig.add_subplot(gs[:, 0])   # left: spans both rows
ax_seq  = fig.add_subplot(gs[0, 1])   # right-top
ax_bar  = fig.add_subplot(gs[1, 1])   # right-bottom

for ax in (ax_traj, ax_seq, ax_bar):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_C)
    ax.tick_params(colors=MUTED_C, labelsize=7)

# Static labels
ax_traj.set_xlabel("Action dim 0  (ΔX)", color=MUTED_C, fontsize=8)
ax_traj.set_ylabel("Action dim 1  (ΔY)", color=MUTED_C, fontsize=8)
ax_traj.set_xlim(*xlim); ax_traj.set_ylim(*ylim)
ax_traj.grid(True, color=GRID_C, linewidth=0.5, alpha=0.8)

ax_seq.set_xlim(0, Tp - 1); ax_seq.set_ylim(*slim)
ax_seq.set_xlabel("Horizon step  t", color=MUTED_C, fontsize=7)
ax_seq.grid(True, color=GRID_C, linewidth=0.4, alpha=0.6)
ax_seq.axhline(0, color=GRID_C, linewidth=0.6)

ax_bar.set_xlim(0, N_DDIM_STEPS); ax_bar.set_ylim(-0.5, 0.5)
ax_bar.set_yticks([])
ax_bar.set_xticks([0, N_DDIM_STEPS // 2, N_DDIM_STEPS])
ax_bar.set_xticklabels(["noise", "", "clean"], color=MUTED_C, fontsize=7)

# Header
fig.text(0.50, 0.96, "DiffuseManip — Diffusion Policy DDIM Denoising",
         ha="center", va="top", color=TEXT_C, fontsize=13, fontweight="bold")
fig.text(0.50, 0.915,
         f"{N_SAMPLES} samples · {act_dim}-dim OSC_POSE · {N_DDIM_STEPS}-step DDIM · same obs, different noise seeds",
         ha="center", va="top", color=MUTED_C, fontsize=8.5)

# ── Pre-create artists (update in draw_frame instead of clearing) ─────────────
# Noise ghost lines (frame 0 shown faintly in background after step 0)
ghost_lines = [ax_traj.plot([], [], color=GRID_C, lw=0.7, alpha=0.45, zorder=1)[0]
               for _ in range(N_SAMPLES)]

# Main trajectory lines + markers
traj_lines  = [ax_traj.plot([], [], color=SAMPLE_COLORS[s], lw=1.7, alpha=0.9, zorder=3)[0]
               for s in range(N_SAMPLES)]
traj_starts = [ax_traj.plot([], [], "o", color=SAMPLE_COLORS[s], ms=4, zorder=4)[0]
               for s in range(N_SAMPLES)]
traj_ends   = [ax_traj.plot([], [], "*", color=SAMPLE_COLORS[s], ms=8, zorder=5)[0]
               for s in range(N_SAMPLES)]

traj_title = ax_traj.set_title("", color=TEXT_C, fontsize=10, fontweight="bold", pad=6)

# Action sequence lines (sample 0, dims 0-2)
seq_lines = [ax_seq.plot([], [], color=DIM_COLORS[d], lw=1.3, label=DIM_LABELS[d])[0]
             for d in range(3)]
ax_seq.legend(fontsize=7, loc="upper right", framealpha=0.25,
              labelcolor=TEXT_C, frameon=True, edgecolor=GRID_C)
seq_title = ax_seq.set_title("", color=MUTED_C, fontsize=8, pad=3)

# Progress bar
bar_done  = ax_bar.barh([0], [0], color=ACCENT_GREEN, height=0.45, alpha=0.85)[0]
bar_left  = ax_bar.barh([0], [N_DDIM_STEPS], left=[0], color=ACCENT_RED, height=0.45, alpha=0.30)[0]
bar_title = ax_bar.set_title("", color=MUTED_C, fontsize=8, pad=3)

t_axis = np.arange(Tp)

def draw_frame(fi):
    data = frames[fi]           # (N, Tp, act_dim)
    step = min(fi, N_DDIM_STEPS)
    noise_data = frames[0]
    is_final = fi >= N_DDIM_STEPS

    label = "Pure Noise" if step == 0 else ("Clean Actions ✓" if is_final else f"Denoising…")
    color = "#56d364" if is_final else ("#f78166" if step == 0 else "#e3b341")
    traj_title.set_text(f"DDIM Step {step}/{N_DDIM_STEPS}  —  {label}")
    traj_title.set_color(color)

    alpha_frac = 0.35 + 0.65 * (step / N_DDIM_STEPS)

    for s in range(N_SAMPLES):
        # Ghost (noise frame)
        if step > 0:
            ghost_lines[s].set_data(noise_data[s, :, 0], noise_data[s, :, 1])
        else:
            ghost_lines[s].set_data([], [])

        # Current trajectories
        traj_lines[s].set_data(data[s, :, 0], data[s, :, 1])
        traj_lines[s].set_alpha(alpha_frac)
        traj_starts[s].set_data([data[s, 0, 0]], [data[s, 0, 1]])
        traj_starts[s].set_alpha(alpha_frac)
        traj_ends[s].set_data([data[s, -1, 0]], [data[s, -1, 1]])
        traj_ends[s].set_alpha(alpha_frac)

    # Action sequence panel (sample 0)
    for d in range(3):
        seq_lines[d].set_data(t_axis, data[0, :, d])
    seq_title.set_text(f"Sample 0 — action dims 0-2  (step {step})")

    # Progress bar
    bar_done.set_width(step)
    bar_left.set_x(step)
    bar_left.set_width(N_DDIM_STEPS - step)
    bar_title.set_text(f"Denoising progress  ({step}/{N_DDIM_STEPS})")

    return (ghost_lines + traj_lines + traj_starts + traj_ends +
            seq_lines + [bar_done, bar_left, traj_title, seq_title, bar_title])

print("Rendering animation...")
ani = animation.FuncAnimation(
    fig, draw_frame, frames=n_frames,
    interval=1000 // FPS, blit=False, repeat=True
)

writer = animation.PillowWriter(fps=FPS, metadata={"loop": 0})
ani.save(OUT_PATH, writer=writer, dpi=DPI)
plt.close(fig)
print(f"Saved: {OUT_PATH}  ({n_frames} frames @ {FPS} fps, {DPI} DPI)")
