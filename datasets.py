"""
datasets.py — robomimic demo loading + Diffusion Policy windowing.

Turns robomimic HDF5 demonstration files into (observation-history, action-sequence)
training windows used by both the BC baseline and Diffusion Policy.

Scope: low-dimensional observations (milestone M1). Image observations (M2) are
marked with TODOs but not implemented here — they need lazy HDF5 reads + a CNN
encoder and shouldn't be loaded fully into RAM.

Window convention (Diffusion Policy, Chi et al. 2023):
  - sample a contiguous window of length `pred_horizon` (Tp) from ONE episode
  - observations: the first `obs_horizon` (To) steps   -> conditioning
  - actions:      all `pred_horizon` (Tp) steps         -> diffusion target
  - episode-boundary windows are padded by repeating the first / last frame,
    with pad_before = To - 1 and pad_after = Ta - 1, so the policy can act from
    the very first step and predict through the very last one.

The heavy logic (indexing, padding, normalization) is pure NumPy so it can be
unit-tested without torch/h5py. Torch is only touched when emitting tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # torch is only needed to emit tensors / subclass Dataset
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except ImportError:  # keeps the module importable for numpy-only testing
    torch = None  # type: ignore
    _TorchDataset = object  # type: ignore
    _HAS_TORCH = False


# Standard robomimic low-dim observation keys for Lift / Can / Square.
# ORDER MATTERS: the live env at eval time must concatenate these in the same
# order to reproduce the obs vector the policy was trained on.
DEFAULT_LOWDIM_OBS_KEYS = (
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
@dataclass
class Normalizer:
    """Per-dimension min/max normalization to [-1, 1].

    Fit on the training data; reused at eval time to (a) normalize live
    observations before they hit the network and (b) un-normalize the policy's
    predicted actions back into env action space.
    """

    obs_min: np.ndarray
    obs_max: np.ndarray
    act_min: np.ndarray
    act_max: np.ndarray
    eps: float = 1e-6

    @classmethod
    def fit(cls, obs: np.ndarray, actions: np.ndarray) -> "Normalizer":
        return cls(
            obs_min=obs.min(axis=0),
            obs_max=obs.max(axis=0),
            act_min=actions.min(axis=0),
            act_max=actions.max(axis=0),
        )

    @staticmethod
    def _to_unit(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float) -> np.ndarray:
        rng = hi - lo
        const = rng < eps
        rng_safe = np.where(const, 1.0, rng)
        out = 2.0 * (x - lo) / rng_safe - 1.0
        return np.where(const, 0.0, out)  # constant dims -> neutral 0

    @staticmethod
    def _from_unit(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float) -> np.ndarray:
        rng = hi - lo
        const = rng < eps
        rng_safe = np.where(const, 1.0, rng)
        out = (x + 1.0) / 2.0 * rng_safe + lo
        return np.where(const, lo, out)  # constant dims -> the constant value

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self._to_unit(obs, self.obs_min, self.obs_max, self.eps)

    def normalize_action(self, act: np.ndarray) -> np.ndarray:
        return self._to_unit(act, self.act_min, self.act_max, self.eps)

    def unnormalize_action(self, act: np.ndarray) -> np.ndarray:
        """Map predicted actions in [-1, 1] back to raw env action space."""
        return self._from_unit(act, self.act_min, self.act_max, self.eps)

    def state_dict(self) -> dict:
        return {
            "obs_min": self.obs_min, "obs_max": self.obs_max,
            "act_min": self.act_min, "act_max": self.act_max, "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(**{k: (np.asarray(v) if k != "eps" else v) for k, v in d.items()})


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """Compute window indices over a flat, episode-concatenated buffer.

    Returns an (N, 4) int array of rows
        (buffer_start, buffer_end, sample_start, sample_end)
    where buffer_[start:end] are global indices into the flat data array, and
    sample_[start:end] are positions within the length-`sequence_length` output
    window. Anything outside [sample_start, sample_end) is padding to be filled
    by repeating the first / last valid frame.

    pad_before / pad_after let windows hang off the start / end of an episode.
    """
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = []
    for i in range(len(episode_ends)):
        start = 0 if i == 0 else int(episode_ends[i - 1])
        end = int(episode_ends[i])
        ep_len = end - start

        min_start = -pad_before
        max_start = ep_len - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start = max(idx, 0) + start
            buffer_end = min(idx + sequence_length, ep_len) + start
            sample_start = max(0, -idx)
            sample_end = sample_start + (buffer_end - buffer_start)
            indices.append((buffer_start, buffer_end, sample_start, sample_end))

    return np.asarray(indices, dtype=np.int64)


def sample_sequence(
    data: np.ndarray,
    sequence_length: int,
    buffer_start: int,
    buffer_end: int,
    sample_start: int,
    sample_end: int,
) -> np.ndarray:
    """Extract one window, padding by repeating the first/last valid frame."""
    out = np.zeros((sequence_length, *data.shape[1:]), dtype=data.dtype)
    out[sample_start:sample_end] = data[buffer_start:buffer_end]
    if sample_start > 0:                       # pad start: repeat first real frame
        out[:sample_start] = out[sample_start]
    if sample_end < sequence_length:           # pad end: repeat last real frame
        out[sample_end:] = out[sample_end - 1]
    return out


# --------------------------------------------------------------------------- #
# HDF5 loading
# --------------------------------------------------------------------------- #
def load_robomimic_lowdim(
    path: str,
    obs_keys=DEFAULT_LOWDIM_OBS_KEYS,
    filter_key: Optional[str] = None,
):
    """Load a robomimic .hdf5 into flat arrays.

    Returns (obs, actions, episode_ends, obs_keys):
      obs           : (T_total, obs_dim)  concatenated low-dim obs keys
      actions       : (T_total, act_dim)
      episode_ends  : (n_episodes,) exclusive end index of each episode in the
                      flat arrays (so episode i spans [ends[i-1], ends[i]))
      obs_keys      : the ordered key list actually used (for eval reconstruction)

    filter_key selects a split from the file's `mask/` group, e.g. "train" /
    "valid" (robomimic ships these for PH datasets).
    """
    import h5py  # imported lazily so the module loads without h5py present

    obs_keys = tuple(obs_keys)
    with h5py.File(path, "r") as f:
        data_grp = f["data"]

        if filter_key is not None:
            demo_names = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in f[f"mask/{filter_key}"][:]
            ]
        else:
            demo_names = list(data_grp.keys())

        # sort numerically by the integer suffix of "demo_<n>"
        demo_names = sorted(demo_names, key=lambda n: int(n.split("_")[-1]))

        obs_chunks, act_chunks, ends = [], [], []
        cursor = 0
        for name in demo_names:
            demo = data_grp[name]
            cols = [np.asarray(demo["obs"][k], dtype=np.float32) for k in obs_keys]
            obs_ep = np.concatenate(cols, axis=-1)
            act_ep = np.asarray(demo["actions"], dtype=np.float32)
            assert obs_ep.shape[0] == act_ep.shape[0], f"{name}: obs/action length mismatch"

            obs_chunks.append(obs_ep)
            act_chunks.append(act_ep)
            cursor += obs_ep.shape[0]
            ends.append(cursor)

    obs = np.concatenate(obs_chunks, axis=0)
    actions = np.concatenate(act_chunks, axis=0)
    episode_ends = np.asarray(ends, dtype=np.int64)
    return obs, actions, episode_ends, obs_keys


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class WindowConfig:
    pred_horizon: int = 16   # Tp — diffusion prediction length / window length
    obs_horizon: int = 2     # To — conditioning steps
    action_horizon: int = 8  # Ta — steps actually executed before re-planning


class RobomimicSequenceDataset(_TorchDataset):
    """(obs-history, action-sequence) windows from robomimic low-dim demos.

    __getitem__ returns a dict:
        obs    : (obs_horizon,  obs_dim)   normalized to [-1, 1]
        action : (pred_horizon, act_dim)   normalized to [-1, 1]

    Construct from arrays (testable) or via `from_hdf5(...)`.
    """

    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        cfg: WindowConfig = WindowConfig(),
        normalizer: Optional[Normalizer] = None,
        obs_keys=DEFAULT_LOWDIM_OBS_KEYS,
    ):
        self.cfg = cfg
        self.obs_keys = tuple(obs_keys)
        self.obs_raw = obs.astype(np.float32)
        self.act_raw = actions.astype(np.float32)
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)

        self.normalizer = normalizer or Normalizer.fit(self.obs_raw, self.act_raw)
        self.obs = self.normalizer.normalize_obs(self.obs_raw)
        self.act = self.normalizer.normalize_action(self.act_raw)

        # window length = Tp; pads let us start at step 0 and finish at the last step
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=cfg.pred_horizon,
            pad_before=cfg.obs_horizon - 1,
            pad_after=cfg.action_horizon - 1,
        )

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        cfg: WindowConfig = WindowConfig(),
        obs_keys=DEFAULT_LOWDIM_OBS_KEYS,
        filter_key: Optional[str] = None,
        normalizer: Optional[Normalizer] = None,
    ) -> "RobomimicSequenceDataset":
        obs, actions, ends, keys = load_robomimic_lowdim(path, obs_keys, filter_key)
        return cls(obs, actions, ends, cfg=cfg, normalizer=normalizer, obs_keys=keys)

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[-1]

    @property
    def act_dim(self) -> int:
        return self.act.shape[-1]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        bs, be, ss, se = (int(v) for v in self.indices[i])
        Tp = self.cfg.pred_horizon

        obs_win = sample_sequence(self.obs, Tp, bs, be, ss, se)[: self.cfg.obs_horizon]
        act_win = sample_sequence(self.act, Tp, bs, be, ss, se)

        if _HAS_TORCH:
            return {
                "obs": torch.from_numpy(obs_win).float(),
                "action": torch.from_numpy(act_win).float(),
            }
        return {"obs": obs_win, "action": act_win}  # numpy path (testing)


# --------------------------------------------------------------------------- #
# Smoke test (needs h5py + a real dataset on disk)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="path to a robomimic low_dim .hdf5")
    ap.add_argument("--filter-key", default="train")
    args = ap.parse_args()

    ds = RobomimicSequenceDataset.from_hdf5(args.hdf5, filter_key=args.filter_key)
    print(f"windows={len(ds)}  obs_dim={ds.obs_dim}  act_dim={ds.act_dim}")
    sample = ds[0]
    o, a = sample["obs"], sample["action"]
    print(f"obs={tuple(o.shape)}  action={tuple(a.shape)}")
    print(f"obs range [{float(o.min()):.3f}, {float(o.max()):.3f}]  "
          f"action range [{float(a.min()):.3f}, {float(a.max()):.3f}]")
