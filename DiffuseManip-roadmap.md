# DiffuseManip — Implementation Roadmap

*Working title; rename to taste (e.g. something in the MiniMotionLM family).*

A manipulation **imitation-learning** project built on MuJoCo. Headline deliverable: a from-scratch **Diffusion Policy**, benchmarked against behavioral-cloning baselines on standardized manipulation tasks, with a clean success-rate comparison and rollout videos.

---

## 1. Why this project, and how it maps to the Agility JD

The JD's *core* asks (not the bonus list) are: 3+ yrs learning-from-demonstration, DiffusionPolicy specifically, robot data collection/training/testing for manipulation, RL infra + sim environments. This project directly answers the first two and the sim-infra piece; it deliberately does **not** claim hardware data collection (your honest gap — see §11).

| JD bullet | What in this project answers it |
|---|---|
| "modern learning-from-demonstration tools like DiffusionPolicy" | Diffusion Policy implemented yourself (§5), the centerpiece |
| "develop, design, test imitation learning methods" | BC baseline + DP + ablations (§5, M1b) |
| "core RL infrastructure, scalable training + evaluation frameworks" | Training loop, config system, batched eval-rollout harness (§7) |
| "design and implement new simulation environments and tasks" | robosuite task configs; stretch humanoid env (M3) |
| "robust policies for manipulation" | Success-rate table across tasks (§7) |
| Bonus: MuJoCo, RL for whole-body control | Built on MuJoCo; optional RL fine-tune (M4) |

The narrative you can tell in the cover letter / README: *"I built a DDPM from scratch, then applied the same denoising machinery to robot control — implementing Diffusion Policy and showing it beats strong BC baselines on multimodal manipulation tasks."* That reuse of your existing diffusion work is the thing that makes this fast for **you** specifically.

---

## 2. Tech stack

- **Sim:** MuJoCo (CPU sim is fine for IL — see §8)
- **Envs + tasks:** `robosuite` (manipulation tasks on a Panda arm)
- **Demos + baselines + eval harness:** `robomimic` (ships human demo datasets, BC/BC-RNN baselines, and a rollout evaluator — don't rebuild these)
- **DL:** PyTorch
- **Diffusion:** your own DDPM code as the starting point; `diffusers` schedulers (DDPM/DDIM) optional for sanity-checking
- **Logging:** Weights & Biases or TensorBoard
- **Video:** imageio / robosuite's offscreen renderer for rollout GIFs

Honest split: **baselines come from robomimic** (run them, cite the numbers). Your **own implementation effort goes into Diffusion Policy** — that's what you want to be able to defend in an interview.

---

## 3. Tasks & data

robomimic ships standardized datasets: **PH** (proficient human) and **MH** (multi-human). Use PH first.

Recommended task progression:

1. **Lift** (pick up a cube) — trivial; use it purely to validate the pipeline end-to-end.
2. **Can** (pick-and-place a can into a bin) — moderate; first "real" result.
3. **Square** (nut-on-peg assembly) — **the showcase**. It's multimodal (several valid approach trajectories), which is exactly where BC averages into mush and Diffusion Policy wins. This is the task your headline number should come from.

Optional harder tasks once the above work: **Transport** (two-arm) or **Tool Hang** (long-horizon, precise).

---

## 4. Observation & action spaces

**Two observation regimes — do low-dim first, then image:**

- **Low-dim** (start here): object pose(s) + robot end-effector pose + gripper state → a vector of a few tens of dims. Trains in minutes-to-hours, lets you debug everything cheaply.
- **Image** (M2): `agentview` RGB + `robot0_eye_in_hand` (wrist) RGB at 84×84 or 128×128, plus proprioception. Encode each camera with a small CNN — robomimic's spatial-softmax ResNet-18, or a plain ResNet-18. This is the deployment-realistic version and gives you the compelling rollout videos.

**Action space:** robosuite default `OSC_POSE` → **7-dim** = 6-DoF delta end-effector pose (3 position + 3 orientation) + 1 gripper. Diffusion Policy predicts an **action *sequence*** over a horizon, not a single action (see §5).

---

## 5. Algorithms

### 5a. BC baselines (use robomimic, don't reimplement)
- **BC-MLP:** obs → action. The floor.
- **BC-RNN:** LSTM over observation history → action. robomimic's *strong* IL baseline; this is the number Diffusion Policy needs to beat to make the project worth anything.

Run both from robomimic configs. Report their success rates as your baselines.

### 5b. Diffusion Policy (your implementation — the centerpiece)

The idea: instead of regressing a single action, model the **conditional distribution over a short sequence of future actions** as a denoising diffusion process. This is why it handles multimodality that BC can't.

Key design choices (paper defaults that work):
- **Horizons (receding-horizon control):** observation horizon `To = 2`, prediction horizon `Tp = 16`, action-execution horizon `Ta = 8`. Predict 16 actions, execute 8, re-plan.
- **Architecture:** **1-D temporal U-Net** over the action sequence, with the observation embedding injected via FiLM conditioning. This is the CNN variant — more stable and less hyperparameter-sensitive than the transformer variant in the paper. **This is structurally your DDPM U-Net, but convolving over the time axis of an action sequence instead of the spatial axes of an image.** That's the reuse.
- **Training:** standard DDPM — add noise to the ground-truth action sequence, predict the noise (ε), MSE loss. 100 train steps. **Use an EMA of the weights** (matters a lot for stability).
- **Inference:** DDIM with ~10–16 steps for fast rollouts.
- **Conditioning:** encode the last `To` observations (low-dim vector, or CNN features for images) → conditioning vector for the U-Net.

If you later want a transformer flavor that ties to MiniMotionLM, **ACT** (Action Chunking Transformer) is the natural second method — but DP is the direct hit on their wording, so lead with it.

---

## 6. Milestones

Structured like your MiniMotionLM M1a/M1b/M2 plan.

### M0 — Pipeline & baseline reproduction
- Install robosuite + robomimic; render headless (EGL) on whatever box you use.
- Download Lift (PH); run robomimic **BC** end-to-end; confirm eval rollouts produce a success rate.
- **Done when:** you reproduce a published robomimic BC number on Lift (±a few %).

### M1 — Diffusion Policy on low-dim
- **M1a:** Your DP (1-D U-Net) on **low-dim** obs, on Lift → Can → Square. Target: **match/beat BC-RNN on Square.**
- **M1b — ablations (the "I understand it" evidence):** DDPM vs DDIM inference steps; prediction/action-horizon sweep; with/without EMA; CNN vs transformer backbone if time. A small table + short writeup.
- **Done when:** DP > BC-RNN success rate on Square, with ablations documented.

### M2 — Image-based DP (the deployment-realistic version)
- Add the vision encoder; train DP from camera obs on one task (Square or Can).
- Produce **rollout GIFs** for the README.
- **Done when:** image-based DP completes the task at a respectable success rate + you have videos.

### M3 — Humanoid gesture *(stretch; Agility's actual domain)*
- One loco-manipulation IL task on **HumanoidBench** (Unitree H1 + hands) or **LocoMujoco** (ships expert datasets), or robosuite two-arm **Transport**.
- Frame explicitly as "extending the same IL machinery toward whole-body / humanoid settings." Don't over-invest — a partial result here is still a strong signal.

### M4 — RL fine-tuning *(optional; ties to your RL strength + mirrors your MiniMotionLM M2)*
- Initialize from the IL policy, improve online with RL. **DPPO** (Diffusion Policy Policy Optimization) is the reference for RL-finetuning a diffusion policy.
- This mirrors your "IL then RL fine-tune" structure from the SMART-R1-style MiniMotionLM plan — nice cross-project coherence.

---

## 7. Evaluation protocol

- **Metric:** task success rate over **50 rollouts** per task (robomimic's evaluator).
- Report **both** best-checkpoint and last-checkpoint success rate (matches DP/robomimic convention), averaged over ≥3 seeds if compute allows.
- **Headline artifact:** a table — rows = tasks (Lift/Can/Square), columns = {BC-MLP, BC-RNN, DP (yours)}, split low-dim vs image. The Square multimodal win is the story.

---

## 8. Compute & cost

- **Low-dim DP (M1):** trains on a single modest GPU in **hours**. MuJoCo runs the physics on CPU; the GPU only carries the network. Your own machine may well be enough.
- **Image DP (M2):** heavier — roughly **~a day on one decent GPU** (4090 / A10 / A6000-class).
- **Rent only if needed:** Lambda / RunPod / Vast for an A10 or 4090 at a few cents to ~$0.50/hr. You are **not** in Isaac Sim territory — none of this needs a multi-GPU parallel-sim rig, which is exactly why MuJoCo was the right call.

*(Numbers are rough — depends on image resolution, horizons, and your hardware. Validate on Lift before committing to long image runs.)*

---

## 9. Suggested repo structure

```
diffuse-manip/
  README.md
  configs/            # task + algo YAMLs
  data/               # robomimic datasets (gitignored)
  diffuse_manip/
    datasets.py       # (obs, action-sequence) windows from demos
    obs_encoders.py   # low-dim + CNN vision encoders
    diffusion_policy.py  # YOUR 1-D U-Net + DDPM/DDIM loop
    bc_baselines.py   # thin wrappers / configs for robomimic BC
    train.py
    eval.py           # rollout success-rate harness
  notebooks/          # ablation plots
  results/            # success-rate tables, GIFs
```

---

## 10. README / portfolio framing

Lead with:
1. One-line pitch + a Square rollout GIF (BC failing / DP succeeding side by side is *chef's kiss* if you can get it).
2. The DDPM→Diffusion Policy narrative (reuse of your generative work).
3. The success-rate table.
4. Ablations.
5. Explicit "what this does and doesn't show" (the hardware honesty — turns a weakness into a credibility signal).

---

## 11. Risks & gotchas (read before starting)

- **Honest scope limit:** this is sim-only. It does **not** demonstrate hardware data collection, which is a core Agility ask. It narrows the gap, doesn't close it. Decide consciously whether this is a primary bet or secondary to your Waymo roadmap before sinking weeks in.
- **Install pain:** robosuite/robomimic + MuJoCo bindings + headless EGL rendering on a rented box is the single most likely time-sink. Budget M0 for it; get rendering working before anything else.
- **Image eval is slow:** rollouts render frames; 50 rollouts × several tasks adds up. Keep eval frequency sane.
- **EMA + horizons matter:** the two most common reasons a DP reimplementation underperforms are missing weight EMA and a bad prediction/action-horizon ratio. Don't skip them.
- **Scope creep:** M0–M2 is a complete, defensible project on its own. M3/M4 are bonus. Ship M2 before touching them.
