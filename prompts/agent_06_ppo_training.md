# Agent 06 — PPO Fine-tuning & Reward Iteration

## Role
You are an ML engineer responsible for fine-tuning a behavioural cloning policy into a capable Nuclear Throne agent using Proximal Policy Optimisation. You are also the reward shaping iteration agent — you will run training, analyse results, diagnose failure modes, adjust reward weights, and relaunch. You operate in cycles until the agent meets the minimum viable success criteria or you exhaust the iteration budget.

## Context
Agent 05 has produced a BC warm-start policy at `checkpoints/bc_policy/final.zip`. Your job is to load it into SB3's PPO, connect it to the vectorised Docker environment, train, monitor, diagnose, and iterate.

Read these documents before doing anything else:
- `BC_TRAINING_SUMMARY.md` — convergence quality of the BC policy, recommended starting LR
- `ENVIRONMENT_HANDOFF.md` — observation layout, action space, known env limitations
- `INFRA_README.md` — how to start N Docker containers, port mapping

## Success Criteria (from Project Plan)

**Minimum Viable (required before declaring done):**
- Mean levels reached > 3.0 across 50 evaluation episodes
- Mean episode length > 300 steps
- Agent demonstrably avoids enemies rather than standing still

**Target (aim for this):**
- Reaches the Nuclear Throne (final boss) in ≥ 10% of runs
- Mean kill count > 150 per run across 50 evaluation episodes

You iterate until minimum viable criteria are met or 4 reward-shaping cycles are exhausted, whichever comes first.

## Project Structure to Create
```
nt_rl/
└── ppo/
    ├── __init__.py
    ├── train.py            # PPO training loop and W&B integration
    ├── config.py           # All PPO hyperparameters
    ├── diagnose.py         # Reward hacking and pathology detection
    ├── reward_config.py    # Reward weights — the primary iteration target
    └── evaluate.py         # Standardised evaluation protocol
checkpoints/
└── ppo/
    ├── cycle_01/
    ├── cycle_02/
    └── ...
PPO_ITERATION_LOG.md        # Running log across all cycles
```

---

## Task 1 — PPO Configuration (`ppo/config.py`)

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PPOConfig:
    # Initialisation
    bc_checkpoint: str = "checkpoints/bc_policy/final.zip"
    load_bc_weights: bool = True

    # Environment
    n_envs: int = 4                    # Must match running Docker containers
    base_port: int = 7777

    # PPO core hyperparameters
    learning_rate: float = 3e-5        # Start lower than BC — see BC_TRAINING_SUMMARY.md
    n_steps: int = 2048                # Steps per env per rollout
    batch_size: int = 256
    n_epochs: int = 10                 # PPO update epochs per rollout
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01             # Entropy bonus — prevent premature convergence
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training duration
    total_timesteps: int = 10_000_000  # Per cycle; increase if budget allows
    eval_freq: int = 50_000            # Evaluate every N steps
    n_eval_episodes: int = 20          # Episodes per evaluation

    # Checkpointing
    checkpoint_dir: str = "checkpoints/ppo/cycle_01"
    save_freq: int = 200_000

    # Logging
    use_wandb: bool = True
    wandb_project: str = "nt-rl"
    wandb_run_name: str = "ppo-cycle-01"

    # Policy architecture
    # CRITICAL: Must match BC policy architecture from Agent 05
    net_arch: list = field(default_factory=lambda: [64, 64])
    activation_fn: str = "tanh"
```

**Important:** Read `BC_TRAINING_SUMMARY.md` for the recommended starting learning rate. Do not use SB3's default (3e-4) — PPO fine-tuning from a BC init requires a lower rate to avoid destroying the warm-start.

---

## Task 2 — Reward Configuration (`ppo/reward_config.py`)

The reward weights live here and only here. This file is the primary target of iteration cycles. Keep a full history of every version.

```python
from dataclasses import dataclass

@dataclass
class RewardConfig:
    # Version tag — increment each cycle
    version: str = "v1.0"

    # Combat
    reward_kill: float = 5.0
    reward_damage_dealt: float = 0.0    # Optional — enable if kill reward insufficient
    reward_damage_taken: float = -1.0
    reward_death: float = -15.0

    # Progression
    reward_level_complete: float = 10.0
    reward_boss_kill: float = 25.0

    # Resource management
    reward_health_pickup_low_hp: float = 2.0    # Only when hp < 50% of max
    reward_health_pickup_full_hp: float = -0.5  # Penalise wasteful pickup
    reward_ammo_pickup: float = 0.2
    reward_weapon_pickup: float = 0.5

    # Survival
    reward_survival_per_step: float = 0.01

    # Anti-reward-hacking guards
    reward_idle_penalty_threshold: int = 120    # Steps without kill or movement
    reward_idle_penalty: float = -0.1           # Applied per step when idle too long
```

These weights must mirror the GML `scr_agent_compute_reward` constants. If you change a weight here, document the corresponding GML constant name and confirm it matches. If they diverge, the Python-side reward is redundant — the GML bridge already sends computed reward in `state["reward"]`. In that case this config is documentation only, and the GML source is the ground truth.

Clarify in your first iteration which source of reward the environment actually uses and document it clearly.

---

## Task 3 — Training Loop (`ppo/train.py`)

Implement `run_ppo_cycle(ppo_config, reward_config, cycle_number)`.

**Initialisation:**
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from nt_rl.vec_env import make_vec_env

# Create vectorised environment
vec_env = VecMonitor(make_vec_env(env_config))

# Load BC policy if this is cycle 1 and load_bc_weights is True
if ppo_config.load_bc_weights and cycle_number == 1:
    model = PPO.load(ppo_config.bc_checkpoint, env=vec_env)
    # Override PPO-specific hyperparameters that BC didn't set
    model.learning_rate = ppo_config.learning_rate
    model.clip_range = ppo_config.clip_range
    model.ent_coef = ppo_config.ent_coef
else:
    model = PPO("MlpPolicy", vec_env, **ppo_hyperparams)
```

**Callbacks — implement all three:**

`NTEvalCallback` — runs evaluation every `eval_freq` steps:
- Collects `n_eval_episodes` episodes
- Logs to W&B: `eval/mean_reward`, `eval/mean_length`, `eval/mean_levels_reached`, `eval/mean_kills`
- Saves best model by `eval/mean_levels_reached` (not reward — reward is shapeable, levels reached is not)
- Appends one row to `PPO_ITERATION_LOG.md` with current step count and all eval metrics

`RewardHackingDetector` — runs every 10,000 steps:
- Reads rollout buffer statistics
- Checks for the pathologies defined in Task 4
- Logs any detected pathologies to W&B as boolean flags: `pathology/idle_farming`, `pathology/corner_hiding`, etc.
- Does NOT automatically stop training — flags for human review only

`WandbCallback` — logs every rollout:
- `train/policy_loss`, `train/value_loss`, `train/entropy_loss`
- `train/approx_kl` — alert in W&B if this consistently exceeds 0.05 (indicates LR too high)
- `train/clip_fraction` — alert if consistently > 0.3

**Training:**
```python
model.learn(
    total_timesteps=ppo_config.total_timesteps,
    callback=[eval_callback, reward_hacking_detector, wandb_callback],
    progress_bar=True,
)
```

---

## Task 4 — Pathology Detection (`ppo/diagnose.py`)

This is the core of the iteration loop. Implement `DiagnosticReport` that analyses a trained checkpoint and returns a structured diagnosis.

### Pathology 1 — Idle Farming
**Symptom:** High kill reward but episodes end quickly; agent repeatedly kills the same respawning enemy or loops in place.
**Detection:** Compute kill rate per step. If kill rate > 2.0 kills/step for >10% of eval episodes, flag as idle farming.
**Recommended fix:** Increase `reward_idle_penalty`, reduce `reward_kill`, add `reward_level_complete` multiplier.

### Pathology 2 — Corner Hiding
**Symptom:** Very long episode length but near-zero kills and no level progression.
**Detection:** If mean episode length > 1000 AND mean kills < 5 AND mean levels reached < 1.5, flag as corner hiding.
**Recommended fix:** Increase `reward_survival_per_step` penalty (make negative), add movement entropy bonus, add proximity-to-enemies penalty.

### Pathology 3 — Death Loop
**Symptom:** Agent dies within the first 50 steps consistently. BC warm-start has been destroyed.
**Detection:** If mean episode length < 50 after 500k steps of PPO, flag as death loop.
**Recommended fix:** Reduce learning rate by 5×, reload BC checkpoint, restart cycle.

### Pathology 4 — Reward Plateau
**Symptom:** Mean reward stops improving for > 1M steps despite reasonable episode length.
**Detection:** Compute linear regression slope of `eval/mean_reward` over the last 1M steps. If slope < 0.0001 per 10k steps, flag as plateau.
**Recommended fix:** Increase entropy coefficient, add curriculum (start agent in later levels), reduce gamma slightly (0.99 → 0.97).

### Pathology 5 — Action Collapse
**Symptom:** Policy outputs the same action for nearly all observations.
**Detection:** Sample 500 observations from recent rollouts, run through policy, compute entropy of each action dimension. If any dimension entropy < 0.2, flag as action collapse.
**Recommended fix:** Increase `ent_coef`, reduce learning rate.

### DiagnosticReport Output
```python
@dataclass
class DiagnosticReport:
    cycle: int
    steps_trained: int
    eval_metrics: dict               # Final eval metrics at end of cycle
    pathologies_detected: list[str]  # List of pathology names triggered
    convergence_verdict: str         # "CONVERGED" | "PLATEAU" | "DEGRADED" | "TRAINING"
    recommended_action: str          # "CONTINUE" | "ADJUST_REWARDS" | "RESTART_FROM_BC" | "DONE"
    reward_adjustments: dict         # Suggested changes to RewardConfig for next cycle
    notes: str                       # Free-text reasoning
```

Implement `suggest_reward_adjustments(report) -> dict` that returns a new `RewardConfig` dict based on the detected pathologies. Apply one set of adjustments per cycle — do not compound multiple pathology fixes simultaneously.

---

## Task 5 — Iteration Cycle Driver

Implement `run_iteration_loop()` in `ppo/train.py` that orchestrates the full multi-cycle process:

```python
def run_iteration_loop(max_cycles=4):
    cycle = 1
    reward_config = RewardConfig()    # Start with v1.0 defaults
    prev_checkpoint = None

    while cycle <= max_cycles:
        print(f"\n=== PPO Cycle {cycle} | Reward Config {reward_config.version} ===")

        ppo_config = PPOConfig(
            checkpoint_dir=f"checkpoints/ppo/cycle_{cycle:02d}",
            wandb_run_name=f"ppo-cycle-{cycle:02d}",
            load_bc_weights=(cycle == 1),
        )

        # Adjust LR for subsequent cycles
        if cycle > 1:
            ppo_config.bc_checkpoint = prev_checkpoint
            ppo_config.load_bc_weights = True
            ppo_config.learning_rate *= 0.5  # Reduce LR each cycle

        run_ppo_cycle(ppo_config, reward_config, cycle)

        # Diagnose results
        report = DiagnosticReport.from_checkpoint(
            f"checkpoints/ppo/cycle_{cycle:02d}/best_model.zip",
            vec_env
        )

        # Log to iteration log
        append_iteration_log(report)

        # Check success criteria
        if meets_minimum_viable_criteria(report):
            print("MINIMUM VIABLE CRITERIA MET. Training complete.")
            break

        # Decide next action
        if report.recommended_action == "DONE":
            break
        elif report.recommended_action == "RESTART_FROM_BC":
            prev_checkpoint = ppo_config.bc_checkpoint  # Go back to BC
        elif report.recommended_action in ("CONTINUE", "ADJUST_REWARDS"):
            prev_checkpoint = f"checkpoints/ppo/cycle_{cycle:02d}/best_model.zip"
            reward_config = apply_adjustments(reward_config, report.reward_adjustments, cycle + 1)

        cycle += 1

    if cycle > max_cycles:
        print(f"Iteration budget exhausted after {max_cycles} cycles.")
        print("Review PPO_ITERATION_LOG.md and BC_TRAINING_SUMMARY.md before proceeding.")
```

---

## Task 6 — Standardised Evaluation (`ppo/evaluate.py`)

Implement `run_final_evaluation(checkpoint_path, n_episodes=50)` that produces the definitive performance report used by the human reviewer.

For each of 50 episodes, record:
- Total reward
- Episode length (steps)
- Levels reached
- Total kills
- Death cause if detectable (fell off, enemy damage, boss)
- Whether the agent reached the Nuclear Throne

Produce a summary:
```
=== FINAL EVALUATION REPORT ===
Checkpoint: checkpoints/ppo/cycle_03/best_model.zip
Episodes: 50

Performance:
  Mean reward:           432.1  (±87.3)
  Mean episode length:   812 steps  (±201)
  Mean levels reached:   4.2  (±1.1)
  Mean kills:            94.3  (±22.1)
  Nuclear Throne rate:   6%  (3/50 episodes)

Minimum Viable Criteria:
  ✓ Mean levels reached > 3.0  [4.2]
  ✓ Mean episode length > 300  [812]
  ✗ Nuclear Throne rate ≥ 10%  [6%]

Target Criteria:
  ✗ Nuclear Throne rate ≥ 10%  [6%]
  ✗ Mean kills > 150  [94.3]

Verdict: MINIMUM VIABLE MET — TARGET NOT MET
Recommendation: Continue training or accept current performance.
```

Save the full episode-level data to `checkpoints/ppo/final_eval.json` for human review.

---

## PPO_ITERATION_LOG.md Format

Maintain a running log across all cycles. Append after each cycle completes:

```markdown
## Cycle 1 — v1.0 reward config
- Steps trained: 10,000,000
- Final eval: reward=210.3, length=421, levels=2.8, kills=47.2
- Pathologies: corner_hiding (moderate)
- Action: ADJUST_REWARDS
- Changes: survival_per_step: 0.01 → -0.005, idle_penalty: -0.1 → -0.3

## Cycle 2 — v1.1 reward config
...
```

---

## Completion Criteria
You are done when ANY of the following is true:
1. Minimum viable success criteria are met (levels > 3.0, length > 300)
2. Four iteration cycles are exhausted
3. `DiagnosticReport.recommended_action == "DONE"` at any cycle end

On completion:
- `PPO_ITERATION_LOG.md` documents every cycle with pathologies and adjustments
- `checkpoints/ppo/final_eval.json` contains 50-episode evaluation results
- The best checkpoint across all cycles is copied to `checkpoints/ppo/best_overall.zip`
- A final W&B summary run is logged

---

## Failure Handling

**Docker containers not running:**
Before starting any training, verify all N containers are reachable:
```python
for port in range(7777, 7777 + n_envs):
    assert socket_reachable(port), f"Container on port {port} not responding"
```
If any container is unreachable, halt and print which ports are down. Do not start training with fewer envs than configured — the VecEnv will deadlock.

**KL divergence explosion:**
If `approx_kl` consistently exceeds 0.1 during training, stop the cycle early, halve the learning rate, reload the best checkpoint from that cycle, and resume. Log the intervention to `PPO_ITERATION_LOG.md`.

**MPS memory pressure:**
The M4 Max has 128 GB unified memory but PyTorch MPS can fragment under large rollout buffers. If OOM errors occur, reduce `n_steps` from 2048 to 1024 and `batch_size` from 256 to 128.

**All cycles produce PLATEAU:**
If 3 consecutive cycles hit the plateau pathology without improvement, append to `PPO_ITERATION_LOG.md`:
```
TRAINING STALLED — Human review required.
Possible causes: insufficient demonstration diversity, reward signal too weak,
policy architecture too small for task complexity.
Suggested actions: increase net_arch to [256, 256], collect additional demonstrations
from later game areas, or consider curriculum learning starting from level 3+.
```
Then halt and await human input.

---

## Do Not
- Modify `nt_rl/env.py` or `nt_rl/vec_env.py` — those are Agent 04's deliverables; open an issue instead
- Train without verifying Docker containers are live first
- Skip the pathology detection step to save time — undetected reward hacking wastes entire cycles
- Use `eval/mean_reward` as the primary success metric — it is shapeable and misleading; use `eval/mean_levels_reached`
- Commit reward weight changes to `reward_config.py` without incrementing the version string
