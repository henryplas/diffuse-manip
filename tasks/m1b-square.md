Goal: extend the existing working Diffusion Policy pipeline (currently 100% on
Lift) to the Can and Square robomimic tasks, and produce a real BC-RNN-vs-DP
success-rate comparison. Square is the priority — it's the multimodal task where
DP should beat BC-RNN by a wide margin, and that gap is the headline result.

Step 1 — Audit before changing anything:
  - Read train.py and eval.py and find everything hardcoded to Lift: the task
    name passed to suite.make(), the obs-key list / obs_dim, the success-detection
    logic, dataset paths, and any episode-length assumptions.
  - Report what's Lift-specific before editing. Do NOT break the working Lift path.

Step 2 — Parameterize the task cleanly:
  - Make task selectable via a --task arg mapping to robosuite envs:
    Lift, PickPlaceCan (Can), NutAssemblySquare (Square).
  - Square/Can use the same low-dim obs keys but the `object` dim differs per task —
    derive obs_dim from the data, don't hardcode 19.
  - Wire success detection per task using robomimic's env success check, not a
    Lift-specific heuristic.

Step 3 — Run it:
  - Download square/ph and can/ph datasets via robomimic's download script.
  - Train DP on Square, then Can, reusing current hyperparameters as the starting
    point. Checkpoint best/last.
  - Eval 50 rollouts per task, report success rate.

Step 4 — Report honestly:
  - Update the README results table with my actual numbers.
  - For the BC-RNN column: clearly label whether those are my own runs or the
    published robomimic PH reference numbers. Don't present reference numbers as mine.
  - Save 2-3 Square rollout GIFs; if feasible, a side-by-side of a BC failure mode
    vs DP success.

Constraints: keep the Lift path working; add a unit test or assertion that obs_dim
is derived, not hardcoded; flag clearly if Square success comes in low so we can
tune horizons/EMA rather than silently reporting a weak number.