# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the windowing unit tests (no deps beyond numpy; torch optional)
python test_windowing.py

# Smoke-test dataset loading against a real robomimic HDF5
python datasets.py --hdf5 data/lift/ph/low_dim.hdf5 --filter-key train
```

Once `train.py` and `eval.py` exist:
```bash
python diffuse_manip/train.py --config configs/lift_dp.yaml
python diffuse_manip/eval.py  --checkpoint runs/<name>/best.ckpt --n-rollouts 50
```

## Architecture

### Data pipeline (`datasets.py`)

The core abstraction is a **sliding window** over flat, episode-concatenated arrays. An episode boundary array (`episode_ends`) tracks where each episode ends; `create_sample_indices` emits `(buffer_start, buffer_end, sample_start, sample_end)` tuples for every valid window. `sample_sequence` materializes one window with repeat-first / repeat-last padding at episode edges.

`RobomimicSequenceDataset.__getitem__` returns:
- `obs`: `(To, obs_dim)` — the last `To` normalized observations (conditioning)
- `action`: `(Tp, act_dim)` — the full normalized action sequence (diffusion target)

**Horizon constants** (paper defaults): `Tp=16`, `To=2`, `Ta=8`. Predict 16 steps, execute 8, re-plan. `pad_before = To-1 = 1`, `pad_after = Ta-1 = 7`.

`Normalizer` fits per-dimension min/max on training data, maps to `[-1, 1]`, and handles constant dims (maps to 0, no NaN).

### Planned modules (not yet implemented)

| Module | Role |
|---|---|
| `obs_encoders.py` | Low-dim passthrough (M1) + spatial-softmax ResNet-18 for image obs (M2) |
| `diffusion_policy.py` | 1-D temporal U-Net with FiLM conditioning; DDPM training loop; DDIM inference |
| `bc_baselines.py` | Config wrappers to run robomimic BC-MLP / BC-RNN |
| `train.py` | Training loop — EMA of weights is critical for DP stability |
| `eval.py` | Rollout harness: 50 rollouts per task, reports success rate |
| `configs/` | Per-task / per-algo YAML configs |

### Diffusion Policy design

The U-Net operates over the **time axis of the action sequence** (length `Tp`), not over image spatial axes — this is the structural reuse of a standard DDPM U-Net. The observation embedding (concatenated last `To` obs) is injected via **FiLM conditioning** at each U-Net residual block. Training: ε-prediction, MSE loss, 100 DDPM steps. Inference: DDIM with 10–16 steps. EMA on model weights is not optional.

### Observation space

Low-dim obs keys (in order — order must match at eval time):
```python
("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
```
Action space: 7-dim `OSC_POSE` (3 pos + 3 rot delta + 1 gripper).

### Task progression

Lift → Can → Square. Square is the headline task: it is multimodal (multiple valid approach trajectories), which is where BC averages into mush and DP wins.
