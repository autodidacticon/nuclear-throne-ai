# Agent 05 — Behavioural Cloning (Imitation Learning)

## Role
You are an ML engineer implementing a behavioural cloning pipeline that trains an initial policy from human gameplay demonstrations. Your output is a saved policy checkpoint that Agent 06 will use as a warm-start for PPO fine-tuning.

## Context
A human has played Nuclear Throne for 3–5 hours with an input logger running. The logger captured synchronised pairs of (game state, action) at each frame. Your job is to load that dataset, preprocess it, train a neural network policy via supervised learning, validate convergence, and save the result in a format SB3's PPO can load directly.

Read these documents before writing any code:
- `ENVIRONMENT_HANDOFF.md` — observation vector layout, action space encoding, known limitations
- `GML_BRIDGE_SUMMARY.md` — canonical state and action JSON schemas the logger used

## Assumptions About the Dataset
Agent 04's input logger produced a dataset in one of these formats. Check which exists:
- `demonstrations/` — directory of per-episode `.npz` files, each containing arrays `obs`, `actions`, `rewards`, `dones`
- `demonstrations.hdf5` — single HDF5 file with the same structure under episode keys
- `demonstrations.pkl` — pickled list of `imitation`-compatible `Trajectory` objects

If none of these exist, halt and report: "No demonstration dataset found at expected paths. Human intervention required."

## Project Structure to Create
```
nt_rl/
├── bc/
│   ├── __init__.py
│   ├── dataset.py          # Dataset loading and preprocessing
│   ├── train.py            # BC training loop
│   ├── evaluate.py         # Policy evaluation against mock/live env
│   └── config.py           # All BC hyperparameters
└── checkpoints/
    └── bc_policy/          # Saved policy output directory
```

## Task 1 — Dataset Loading (`bc/dataset.py`)

Implement `DemonstrationDataset` that:
- Detects and loads whichever of the three dataset formats exists
- Validates that observation shape matches `EnvConfig` from `nt_rl/config.py`
- Validates that actions are valid samples from the `MultiDiscrete([9, 24, 2, 2])` space
- Reports dataset statistics on load:
  - Total transitions
  - Number of episodes
  - Mean episode length
  - Mean reward per episode
  - Action distribution (what % of steps include shooting, dodging, etc.)
  - Class balance warning if any action dimension is >95% one value

Implement `split(train_ratio=0.9)` returning train and validation subsets.

Implement a `to_imitation_trajectories()` method that converts the dataset to a list of `imitation.data.types.Trajectory` objects, which the `imitation` library's BC trainer expects.

Flag any episodes where `done` is never True — these are truncated episodes and should be handled as `terminal=False` in the trajectory objects.

## Task 2 — BC Configuration (`bc/config.py`)

```python
from dataclasses import dataclass

@dataclass
class BCConfig:
    # Training
    n_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 3e-4
    l2_reg: float = 1e-5
    grad_clip: float = 0.5

    # LR schedule
    lr_schedule: str = "cosine"     # "cosine" | "linear" | "constant"
    warmup_steps: int = 500

    # Policy architecture (must match what PPO will use in Agent 06)
    net_arch: list = None           # None = SB3 default [64, 64]
    activation_fn: str = "tanh"     # "tanh" | "relu"

    # Evaluation
    eval_every_n_epochs: int = 1
    n_eval_episodes: int = 20       # Against mock env
    eval_port: int = 17777          # Mock server port for eval

    # Checkpointing
    checkpoint_dir: str = "checkpoints/bc_policy"
    save_best_only: bool = True     # Save only when val loss improves

    # Logging
    use_wandb: bool = True
    wandb_project: str = "nt-rl"
    wandb_run_name: str = "bc-training"
    log_every_n_steps: int = 50
```

## Task 3 — Training Loop (`bc/train.py`)

Use the `imitation` library's `BC` trainer. Do not reimplement supervised training from scratch.

```python
from imitation.algorithms import bc
from imitation.data import rollout
from stable_baselines3.common.policies import ActorCriticPolicy
```

The training script must:

**Setup:**
- Load dataset via `DemonstrationDataset`
- Print dataset statistics
- Initialise a `NuclearThroneEnv` against the mock server (not Docker) for evaluation
- Build a `BC` trainer with an `ActorCriticPolicy` matching the observation and action spaces
- Initialise W&B run if `BCConfig.use_wandb` is True

**Training loop:**
- Train for `n_epochs`, evaluating on the validation split each epoch
- Log to W&B each epoch:
  - `train/loss` — mean training loss
  - `val/loss` — validation loss
  - `val/action_accuracy` — % of actions matching demonstration exactly
  - `eval/mean_episode_reward` — from mock env rollouts
  - `eval/mean_episode_length`
- Save checkpoint when val loss improves (if `save_best_only`) or every epoch otherwise
- Print a convergence warning if val loss has not improved for 3 consecutive epochs

**Early stopping:**
- Stop training if val loss has not improved for 5 consecutive epochs
- Print "Early stopping triggered at epoch N" and proceed to final save

**Final save:**
- Save the best checkpoint to `BCConfig.checkpoint_dir`
- Save in SB3 `.zip` format: `policy.save("checkpoints/bc_policy/final")`
- Also export the raw PyTorch state dict: `torch.save(policy.state_dict(), "checkpoints/bc_policy/final_state_dict.pt")`
- This dual format ensures Agent 06 can load it regardless of SB3 version differences

## Task 4 — Policy Evaluation (`bc/evaluate.py`)

Implement `evaluate_policy(policy, env, n_episodes)` that:
- Runs `n_episodes` episodes using the policy deterministically (`deterministic=True`)
- Returns a dict:
  ```python
  {
    "mean_reward": float,
    "std_reward": float,
    "mean_length": float,
    "mean_kills": float,           # from info dict if available
    "mean_levels_reached": float,  # from info dict if available
    "episodes_completed": int,
  }
  ```
- Logs a warning if mean episode length < 50 steps — this suggests the policy is dying immediately and BC has not learned even basic survival behaviour

Also implement `action_distribution_report(policy, dataset)` that samples 1000 observations from the dataset, runs them through the policy, and reports the marginal distribution of each action dimension. This catches degenerate policies that always output the same action.

## Task 5 — Convergence Validation

Before writing `BC_TRAINING_SUMMARY.md`, run the following checks and report results:

**Check 1 — Loss convergence:**
Final val loss must be < 80% of initial val loss. If not: flag as "TRAINING DID NOT CONVERGE" and suggest increasing n_epochs or decreasing learning_rate.

**Check 2 — Action diversity:**
No single action value should account for >80% of the policy's outputs on the validation set. If it does: flag as "DEGENERATE POLICY — action collapse detected".

**Check 3 — Non-random baseline:**
Mean eval episode length from the BC policy must exceed mean episode length of a random policy by at least 20%. Compute random policy baseline by running 20 episodes with random actions on the mock env.

**Check 4 — Observation coverage:**
Verify the demonstration dataset contains episodes from at least 3 distinct levels (check the `level` field in observations). A policy trained only on level 1 data will fail in later levels.

## Output Artifacts

```
nt_rl/bc/                          # All source code
checkpoints/bc_policy/
├── final.zip                      # SB3-loadable policy
├── final_state_dict.pt            # Raw PyTorch weights
└── training_log.json              # Per-epoch metrics
BC_TRAINING_SUMMARY.md
```

`BC_TRAINING_SUMMARY.md` must contain:
- Dataset statistics (transitions, episodes, action distribution)
- Training configuration used
- Per-epoch loss table (train and val)
- Final evaluation results (mean reward, length, kills, levels reached)
- All convergence check results (pass/fail with values)
- Recommendation for Agent 06: suggested PPO starting learning rate based on BC convergence behaviour
- Any anomalies or warnings encountered

## Completion Criteria
You are done when:
- All source files in `nt_rl/bc/` exist and are fully implemented
- Training runs to completion without error
- `checkpoints/bc_policy/final.zip` exists and is loadable by SB3
- All 4 convergence checks are reported (pass or fail — do not silently skip failures)
- `BC_TRAINING_SUMMARY.md` is complete

## Failure Handling

**Dataset format unrecognised:**
Halt immediately. Do not attempt to infer format from file contents beyond the three documented formats. Report exact file listing of `demonstrations/` to `BC_TRAINING_SUMMARY.md` and request human clarification.

**CUDA / MPS not available:**
Fall back to CPU training. Log a warning. On the M4 Max host, MPS should be available — if it is not, check that `torch.backends.mps.is_available()` and report the result. Do not fail silently.

**imitation library version mismatch:**
The `imitation` library API changes between versions. If import errors occur, check `pip show imitation` for the installed version and adjust API calls accordingly. Document any API differences from the expected version in `BC_TRAINING_SUMMARY.md`.

**Degenerate policy (Check 2 fails):**
Do not pass a degenerate checkpoint to Agent 06. Instead:
1. Reduce learning rate by 10× and retrain
2. If still degenerate, add label smoothing (0.1) to the BC loss
3. If still degenerate after both mitigations, save the checkpoint anyway but mark it "DEGENERATE — USE WITH CAUTION" in the summary and notify that human review is needed before proceeding to PPO

## Do Not
- Train on the full dataset without a validation split — overfitting to demonstrations is a known BC failure mode
- Use the live Docker game environment for evaluation — use the mock server only
- Hardcode the observation dimension — read it from `EnvConfig` or the loaded dataset shape
- Assume MPS is available — always check and fall back gracefully
