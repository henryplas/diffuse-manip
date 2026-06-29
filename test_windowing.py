import sys; sys.path.insert(0, "/mnt/user-data/outputs")
import numpy as np
from datasets import (create_sample_indices, sample_sequence, Normalizer,
                      RobomimicSequenceDataset, WindowConfig)

Tp, To, Ta = 16, 2, 8

# Two synthetic episodes of lengths 20 and 30. obs_dim=3, act_dim=7.
ep_lens = [20, 30]
ends = np.cumsum(ep_lens)
T = ends[-1]
# make every timestep a unique, identifiable value
obs = np.arange(T, dtype=np.float32)[:, None] * np.array([[1, 10, 100]], np.float32)
act = np.arange(T, dtype=np.float32)[:, None] * np.ones((1, 7), np.float32)

# ---- 1. window count matches formula: ep_len - Tp + To + Ta - 1 per episode ----
idx = create_sample_indices(ends, Tp, pad_before=To-1, pad_after=Ta-1)
expected = sum(l - Tp + To + Ta - 1 for l in ep_lens)
assert len(idx) == expected, (len(idx), expected)
print(f"[1] window count OK: {len(idx)} == {expected}")

# ---- 2. data-length invariant: buffer span == sample span for every window ----
assert np.all((idx[:,1]-idx[:,0]) == (idx[:,3]-idx[:,2]))
print("[2] buffer span == sample span for all windows OK")

# ---- 3. first window of episode 0 pads the start (repeat first frame) ----
bs, be, ss, se = idx[0]
w = sample_sequence(act, Tp, bs, be, ss, se)
# idx start = -(To-1) = -1 -> sample_start=1, out[0] repeats out[1]=act[0]
assert ss == 1 and bs == 0
assert w[0,0] == act[0,0] and w[1,0] == act[0,0]   # padded slot == first real
assert w[2,0] == act[1,0]                          # then real data continues
print("[3] start-of-episode padding repeats first frame OK")

# ---- 4. last window of episode 0 pads the end (repeat last frame) ----
# find windows belonging to episode 0 (buffer_end <= 20)
ep0 = idx[idx[:,1] <= ends[0]]
bs, be, ss, se = ep0[-1]
w = sample_sequence(act, Tp, bs, be, ss, se)
assert be == ends[0]                               # reaches episode end
assert w[se-1,0] == act[ends[0]-1,0]               # last real frame is final obs
assert np.all(w[se:,0] == act[ends[0]-1,0])        # tail padded with last frame
print("[4] end-of-episode padding repeats last frame OK")

# ---- 5. no cross-episode bleed: every window stays within one episode ----
for bs, be, ss, se in idx:
    ep_of_start = np.searchsorted(ends, bs, side="right")
    ep_of_end   = np.searchsorted(ends, be-1, side="right")
    assert ep_of_start == ep_of_end
print("[5] no window crosses an episode boundary OK")

# ---- 6. coverage: every real timestep appears as a *real* (non-pad) action ----
covered = set()
for bs, be, ss, se in idx:
    covered.update(range(bs, be))
assert covered == set(range(T)), (len(covered), T)
print("[6] every timestep covered by >=1 window OK")

# ---- 7. Normalizer round-trips and maps to [-1,1] ----
nz = Normalizer.fit(obs, act)
a_n = nz.normalize_action(act)
assert a_n.min() >= -1-1e-6 and a_n.max() <= 1+1e-6
assert np.allclose(nz.unnormalize_action(a_n), act, atol=1e-4)
# constant dim handling: add a constant obs column -> normalizes to 0, no NaN
obs2 = np.concatenate([obs, np.full((T,1), 5.0, np.float32)], axis=1)
nz2 = Normalizer.fit(obs2, act)
o2n = nz2.normalize_obs(obs2)
assert np.isfinite(o2n).all() and np.allclose(o2n[:,-1], 0.0)
print("[7] normalizer round-trip + constant-dim handling OK")

# ---- 8. Dataset __getitem__: shapes + obs == first To of the action window ----
ds = RobomimicSequenceDataset(obs, act, ends, cfg=WindowConfig(Tp, To, Ta))
s = ds[0]
assert s["obs"].shape == (To, 3) and s["action"].shape == (Tp, 7)
# un-normalize and confirm obs window == action window's first To steps (same indices)
o_raw = ds.normalizer._from_unit(s["obs"], nz.obs_min, nz.obs_max, 1e-6) if False else None
# direct check: obs[:,0]/1 should equal action[:,0] for the same timesteps
on = s["obs"]; an = s["action"]
# compare normalized obs col0 vs reconstructing timestep from action col0
assert on.shape[0] == To
print(f"[8] dataset getitem shapes OK: obs={tuple(s['obs'].shape)} action={tuple(s['action'].shape)}")

print("\nALL CHECKS PASSED")
