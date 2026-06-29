"""
eval.py — rollout evaluation harness for Diffusion Policy (low-dim M1).

Loads a trained checkpoint, runs N rollouts in a live robosuite environment,
and reports the success rate. Optionally saves a GIF of every rollout.

Usage:
    python eval.py --checkpoint runs/Lift_20240101/best.ckpt --task Lift
    python eval.py --checkpoint runs/Square_20240101/best.ckpt --task Square --n-rollouts 50
    python eval.py --checkpoint runs/Lift_20240101/best.ckpt --save-videos

Requires robosuite + robomimic (M0 install):
    pip install robosuite robomimic
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from datasets import Normalizer
from train import DiffusionPolicyNet


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def load_policy(ckpt_path: str, device: torch.device, use_ema: bool = True) -> tuple:
    """Load a checkpoint and return (net, normalizer, obs_keys, hparams).

    use_ema=True applies the EMA shadow weights. Only useful when the model was
    trained for many steps (100k+). With short training runs (<10k steps) the
    EMA shadow is dominated by initial random weights (decay^N stays near 1.0),
    so use_ema=False (raw model_state) typically performs better.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp   = ckpt["hparams"]

    net = DiffusionPolicyNet(
        obs_dim      = ckpt["obs_dim"],
        obs_horizon  = hp["obs_horizon"],
        obs_cond_dim = hp["obs_cond_dim"],
        act_dim      = ckpt["act_dim"],
        pred_horizon = hp["pred_horizon"],
        down_dims    = tuple(hp["down_dims"]),
        n_timesteps  = hp["n_timesteps"],
    ).to(device)

    net.load_state_dict(ckpt["model_state"])

    if use_ema and "ema" in ckpt and ckpt["ema"] is not None:
        N = ckpt.get("global_step", 0)
        decay = ckpt["ema"]["decay"]
        initial_frac = decay ** N
        if initial_frac > 0.5:
            print(f"WARNING: EMA shadow is {initial_frac*100:.0f}% initial random weights "
                  f"(decay={decay}, steps={N}). Using raw model weights instead.")
            use_ema = False
        else:
            for k, p in net.named_parameters():
                if k in ckpt["ema"]["shadow"]:
                    p.data.copy_(ckpt["ema"]["shadow"][k].to(p.dtype))
            print(f"EMA weights applied ({(1-initial_frac)*100:.0f}% trained).")

    if not use_ema:
        print("Using raw model weights (no EMA).")

    net.eval()
    normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
    return net, normalizer, ckpt["obs_keys"], hp


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def make_env(task: str, seed: int = 0, render_offscreen: bool = False):
    """Create a robosuite environment matching the training data setup."""
    try:
        import robosuite as suite
    except ImportError:
        raise ImportError(
            "robosuite is not installed.\n"
            "Install with:  pip install robosuite\n"
            "See M0 setup in the roadmap."
        )

    env = suite.make(
        env_name              = task,
        robots                = "Panda",
        has_renderer          = False,
        has_offscreen_renderer= render_offscreen,
        use_camera_obs        = render_offscreen,
        camera_names          = ["agentview"] if render_offscreen else None,
        camera_heights        = 256,
        camera_widths         = 256,
        use_object_obs        = True,
        reward_shaping        = False,
        control_freq          = 20,
        ignore_done           = False,
        horizon               = 500,
        seed                  = seed,
    )
    return env


def extract_obs(env_obs: dict, obs_keys: tuple) -> np.ndarray:
    """Concatenate observation keys from a robosuite obs dict.

    robosuite uses "object-state" for the concatenated object observations;
    robomimic stores this under the key "object" in the HDF5. We map here.

    Version mismatch fix: robomimic v141 (collected with robosuite 1.4.1) stores
    dims 7-9 of `object` as `eef_pos - cube_pos`, but robosuite 1.5 returns
    `cube_pos - eef_pos` in `object-state`. Negate dims 7-9 to match training.
    """
    KEY_MAP = {"object": "object-state"}
    parts = []
    for k in obs_keys:
        env_key = KEY_MAP.get(k, k)
        if env_key not in env_obs:
            raise KeyError(
                f"Obs key '{env_key}' not found in env obs.\n"
                f"Available keys: {list(env_obs.keys())}"
            )
        val = env_obs[env_key].flatten().astype(np.float32)
        if env_key == "object-state" and val.shape[0] == 10:
            val = val.copy()
            val[7:10] = -val[7:10]
        parts.append(val)
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# Single rollout
# --------------------------------------------------------------------------- #
def run_rollout(
    env,
    net: DiffusionPolicyNet,
    normalizer: Normalizer,
    obs_keys: tuple,
    obs_horizon: int,
    action_horizon: int,
    n_ddim_steps: int,
    device: torch.device,
    save_frames: bool = False,
) -> tuple[bool, int, list]:
    """Run one episode. Returns (success, episode_length, frames)."""

    raw_obs = env.reset()
    obs_vec = extract_obs(raw_obs, obs_keys)

    # Initialise obs buffer — pad with the first observation
    obs_buf = deque([obs_vec] * obs_horizon, maxlen=obs_horizon)

    frames   = []
    step_idx = 0
    done     = False
    success  = False

    while not done:
        # ── Build normalised obs tensor ──────────────────────────────────────
        obs_arr = normalizer.normalize_obs(np.stack(list(obs_buf)))   # (To, obs_dim)
        obs_t   = torch.from_numpy(obs_arr).float().unsqueeze(0).to(device)  # (1, To, obs_dim)

        # ── Predict action chunk ─────────────────────────────────────────────
        actions_n = net.predict(obs_t, n_ddim_steps=n_ddim_steps)    # (1, Tp, act_dim)
        actions   = normalizer.unnormalize_action(
            actions_n.squeeze(0).cpu().numpy()
        )  # (Tp, act_dim)

        # ── Execute Ta actions (receding-horizon) ────────────────────────────
        for a in actions[:action_horizon]:
            if done:
                break
            raw_obs, reward, done, info = env.step(a)
            obs_buf.append(extract_obs(raw_obs, obs_keys))
            step_idx += 1

            if save_frames:
                # robosuite 1.5: camera frames are in the obs dict, not env.render()
                frames.append(raw_obs["agentview_image"])

            # robosuite 1.5 puts nothing in info — check success via reward or
            # _check_success() directly (sparse reward = 1.0 on success).
            if reward > 0 or info.get("success", False) or env._check_success():
                success = True
                done = True
                break

    return success, step_idx, frames


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    net, normalizer, obs_keys, hp = load_policy(args.checkpoint, device)

    print(f"Task: {args.task}  |  {args.n_rollouts} rollouts")
    print(f"obs_keys: {obs_keys}")

    results_dir = Path(args.results_dir) / Path(args.checkpoint).parent.name
    if args.save_videos:
        results_dir.mkdir(parents=True, exist_ok=True)

    successes   = []
    ep_lengths  = []

    for i in range(args.n_rollouts):
        env = make_env(args.task, seed=args.seed + i, render_offscreen=args.save_videos)

        success, length, frames = run_rollout(
            env          = env,
            net          = net,
            normalizer   = normalizer,
            obs_keys     = tuple(obs_keys),
            obs_horizon  = hp["obs_horizon"],
            action_horizon = hp["action_horizon"],
            n_ddim_steps = hp["n_ddim_steps"],
            device       = device,
            save_frames  = args.save_videos,
        )
        env.close()

        successes.append(int(success))
        ep_lengths.append(length)

        tag = "SUCCESS" if success else "fail"
        print(f"  rollout {i+1:3d}/{args.n_rollouts}  {tag}  ({length} steps)")

        if args.save_videos and frames:
            _save_gif(frames, results_dir / f"rollout_{i+1:03d}_{tag}.gif")

    success_rate = float(np.mean(successes))
    avg_len      = float(np.mean(ep_lengths))

    summary = {
        "task":         args.task,
        "checkpoint":   args.checkpoint,
        "n_rollouts":   args.n_rollouts,
        "success_rate": success_rate,
        "avg_ep_len":   avg_len,
        "successes":    successes,
    }

    print(f"\nSuccess rate: {success_rate*100:.1f}%  ({sum(successes)}/{args.n_rollouts})")
    print(f"Avg episode length: {avg_len:.0f} steps")

    out_path = results_dir / "eval_results.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {out_path}")

    return summary


def _save_gif(frames: list, path: Path, fps: int = 10) -> None:
    try:
        import imageio
        imageio.mimsave(str(path), frames, fps=fps)
    except ImportError:
        try:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(str(path), save_all=True, append_images=imgs[1:],
                         duration=1000 // fps, loop=0)
        except ImportError:
            print(f"  (skipping GIF save — install imageio or Pillow)")


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--task",       default="Lift",
                   help="robosuite task name (Lift / Can / Square)")
    p.add_argument("--n-rollouts", type=int, default=50)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--save-videos",action="store_true",
                   help="Render and save rollout GIFs")
    p.add_argument("--results-dir",default="results")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    evaluate(parse_args())
