"""
train.py — Diffusion Policy training loop (low-dim M1).

Usage:
    python train.py --hdf5 data/lift/ph/low_dim.hdf5
    python train.py --hdf5 data/square/ph/low_dim.hdf5 --task Square --epochs 300
    python train.py --hdf5 data/lift/ph/low_dim.hdf5 --resume runs/Lift_20240101/last.ckpt

Checkpoints are saved to runs/<task>_<timestamp>/:
    last.ckpt  — end of every epoch (resume from here)
    best.ckpt  — lowest val loss seen so far (use for eval)
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import RobomimicSequenceDataset, WindowConfig
from obs_encoders import LowDimEncoder
from diffusion_policy import build_diffusion_policy


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class EMA:
    """Exponential Moving Average of model parameters.

    Only tracks learnable parameters (not fixed buffers like noise schedules).
    EMA weights are what you use for eval and inference — they are much more
    stable than the live weights during training.

    Typical decay: 0.9999 (updates shadow by 0.01% of new weights each step).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: p.data.clone().float() for k, p in model.named_parameters()}
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, p in model.named_parameters():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * p.data.float()

    def apply(self, model: nn.Module) -> None:
        """Swap live weights for EMA weights (saves live weights first)."""
        self._backup = {k: p.data.clone() for k, p in model.named_parameters()}
        for k, p in model.named_parameters():
            p.data.copy_(self.shadow[k].to(p.dtype))

    def restore(self, model: nn.Module) -> None:
        """Restore live weights after an EMA eval pass."""
        for k, p in model.named_parameters():
            p.data.copy_(self._backup[k])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, d: dict) -> None:
        self.decay = d["decay"]
        self.shadow = d["shadow"]


# --------------------------------------------------------------------------- #
# Combined encoder + diffusion
# --------------------------------------------------------------------------- #
class DiffusionPolicyNet(nn.Module):
    """Encoder + GaussianDiffusion wrapped together for EMA and checkpointing."""

    def __init__(
        self,
        obs_dim: int,
        obs_horizon: int,
        obs_cond_dim: int,
        act_dim: int,
        pred_horizon: int,
        down_dims: tuple,
        n_timesteps: int = 100,
    ):
        super().__init__()
        self.encoder = LowDimEncoder(obs_dim, obs_horizon, obs_cond_dim)
        _, self.diffusion = build_diffusion_policy(
            obs_dim=obs_dim,
            obs_horizon=obs_horizon,
            obs_cond_dim=obs_cond_dim,
            act_dim=act_dim,
            pred_horizon=pred_horizon,
            down_dims=down_dims,
            n_timesteps=n_timesteps,
        )

    def loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """obs: (B, To, obs_dim)  action: (B, Tp, act_dim)  -> scalar loss."""
        return self.diffusion.loss(action, self.encoder(obs))

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, n_ddim_steps: int = 16) -> torch.Tensor:
        """obs: (B, To, obs_dim) normalized -> (B, Tp, act_dim) normalized."""
        return self.diffusion.ddim_sample(self.encoder(obs), n_steps=n_ddim_steps)


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument("--hdf5", required=True, help="Path to robomimic low_dim .hdf5")
    p.add_argument("--filter-key", default="train", help="Dataset split key")
    p.add_argument("--task", default="Lift", help="Task name (for run directory label)")

    # Window (paper defaults)
    p.add_argument("--pred-horizon",   type=int, default=16, help="Tp")
    p.add_argument("--obs-horizon",    type=int, default=2,  help="To")
    p.add_argument("--action-horizon", type=int, default=8,  help="Ta")

    # Model
    p.add_argument("--obs-cond-dim", type=int, default=256)
    p.add_argument("--down-dims",    default="256,512,1024",
                   help="U-Net channel widths (comma-separated)")
    p.add_argument("--n-timesteps",  type=int, default=100, help="DDPM T")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--ema-decay",    type=float, default=0.999)
    p.add_argument("--grad-clip",    type=float, default=1.0)

    # Eval / logging
    p.add_argument("--n-ddim-steps", type=int, default=16)
    p.add_argument("--val-interval", type=int, default=5,
                   help="Run val loss every N epochs (0 = skip)")
    p.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    p.add_argument("--save-dir",     default="runs")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb",        action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--resume",       default=None, metavar="CKPT",
                   help="Checkpoint path to resume from")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    down_dims = tuple(int(d) for d in args.down_dims.split(","))

    # ── Dataset ─────────────────────────────────────────────────────────────
    win_cfg = WindowConfig(args.pred_horizon, args.obs_horizon, args.action_horizon)
    ds_train = RobomimicSequenceDataset.from_hdf5(
        args.hdf5, cfg=win_cfg, filter_key=args.filter_key
    )
    print(f"Train: {len(ds_train)} windows  obs_dim={ds_train.obs_dim}  act_dim={ds_train.act_dim}")

    ds_val = None
    if args.val_interval > 0:
        try:
            ds_val = RobomimicSequenceDataset.from_hdf5(
                args.hdf5, cfg=win_cfg, filter_key="valid",
                normalizer=ds_train.normalizer,
            )
            print(f"Val:   {len(ds_val)} windows")
        except Exception:
            print("No 'valid' split found — skipping val loss.")

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=True,
    )
    loader_val = (
        DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if ds_val else None
    )

    # ── Model + EMA + optimiser ─────────────────────────────────────────────
    net = DiffusionPolicyNet(
        obs_dim=ds_train.obs_dim,
        obs_horizon=args.obs_horizon,
        obs_cond_dim=args.obs_cond_dim,
        act_dim=ds_train.act_dim,
        pred_horizon=args.pred_horizon,
        down_dims=down_dims,
        n_timesteps=args.n_timesteps,
    ).to(device)

    ema = EMA(net, decay=args.ema_decay)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    # ── Run directory ────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.save_dir) / f"{args.task}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run dir: {run_dir}")

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {args.resume}  (epoch {start_epoch}, step {global_step})")

    # ── W&B ─────────────────────────────────────────────────────────────────
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project="diffuse-manip", name=run_dir.name, config=vars(args))
        except ImportError:
            print("wandb not installed — skipping. pip install wandb to enable.")

    # ── Checkpoint helper ────────────────────────────────────────────────────
    def save_ckpt(path: Path, val_loss: float | None = None) -> None:
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": net.state_dict(),
            "ema": ema.state_dict(),
            "optimizer_state": opt.state_dict(),
            "normalizer": ds_train.normalizer.state_dict(),
            "obs_keys": list(ds_train.obs_keys),
            "obs_dim": ds_train.obs_dim,
            "act_dim": ds_train.act_dim,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "hparams": {
                "obs_horizon": args.obs_horizon,
                "pred_horizon": args.pred_horizon,
                "action_horizon": args.action_horizon,
                "obs_cond_dim": args.obs_cond_dim,
                "down_dims": list(down_dims),
                "n_timesteps": args.n_timesteps,
                "n_ddim_steps": args.n_ddim_steps,
            },
        }, path)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        net.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader_train:
            obs    = batch["obs"].to(device)
            action = batch["action"].to(device)

            loss = net.loss(obs, action)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            ema.update(net)

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                print(f"  step {global_step:6d}  loss={loss.item():.4f}")
                if wb:
                    wb.log({"train/loss": loss.item()}, step=global_step)

        avg_loss = epoch_loss / len(loader_train)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={avg_loss:.4f}  ({elapsed:.0f}s)")
        if wb:
            wb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1}, step=global_step)

        # ── Val loss (EMA model) ─────────────────────────────────────────────
        val_loss = None
        if loader_val and args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            ema.apply(net)
            net.eval()
            total = 0.0
            with torch.no_grad():
                for batch in loader_val:
                    total += net.loss(
                        batch["obs"].to(device), batch["action"].to(device)
                    ).item()
            val_loss = total / len(loader_val)
            ema.restore(net)
            print(f"          val_loss={val_loss:.4f}  (EMA)")
            if wb:
                wb.log({"val/loss": val_loss, "epoch": epoch + 1}, step=global_step)

        # ── Checkpoint ────────────────────────────────────────────────────────
        save_ckpt(run_dir / "last.ckpt", val_loss)
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(run_dir / "best.ckpt", val_loss)
            print(f"          -> best.ckpt  (val_loss={best_val_loss:.4f})")

    # Final checkpoint (EMA weights baked in — ready for eval without --ema flag)
    ema.apply(net)
    save_ckpt(run_dir / "final.ckpt", val_loss)
    ema.restore(net)

    if wb:
        wb.finish()

    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Artifacts: {run_dir}")
    return run_dir


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    train(parse_args())
