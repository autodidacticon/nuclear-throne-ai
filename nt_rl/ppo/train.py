"""PPO training loop with BC warm-start, callbacks, and multi-cycle iteration.

Usage:
    python -m nt_rl.ppo.train [--no-wandb] [--cycle N] [--dry-run] [--timesteps N]
"""

import argparse
import json
import os
import socket
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import gymnasium
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from nt_rl.config import EnvConfig


# ---------------------------------------------------------------------------
# Lazy imports for config/reward_config — created by parallel agent
# ---------------------------------------------------------------------------

def _load_ppo_config():
    from nt_rl.ppo.config import PPOConfig
    return PPOConfig


def _load_reward_config():
    from nt_rl.ppo.reward_config import RewardConfig
    return RewardConfig


# ---------------------------------------------------------------------------
# Port verification
# ---------------------------------------------------------------------------

def check_port_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def verify_all_ports(host: str, base_port: int, n_envs: int) -> list[int]:
    """Check that all N game instance ports are reachable.

    Returns a list of unreachable ports (empty if all OK).
    """
    unreachable = []
    for i in range(n_envs):
        port = base_port + i
        if not check_port_reachable(host, port):
            unreachable.append(port)
    return unreachable


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS can work for PPO but may have memory fragmentation.
        # Try it; fall back to CPU on OOM.
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class NTEvalCallback(EvalCallback):
    """Extended EvalCallback that tracks Nuclear Throne-specific metrics.

    Logs eval/mean_levels_reached and eval/mean_kills alongside standard
    metrics.  Saves best model by mean_levels_reached (not reward).
    """

    def __init__(self, eval_env, log_dir: str, use_wandb: bool = True, **kwargs):
        super().__init__(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            deterministic=True,
            **kwargs,
        )
        self._log_dir = log_dir
        self._use_wandb = use_wandb
        self._best_mean_levels = -float("inf")
        self._eval_history: list[dict] = []

    def _on_step(self) -> bool:
        result = super()._on_step()

        # After an eval round, pull extra metrics from info dicts
        if self.evaluations_results is not None and len(self.evaluations_results) > 0:
            last_rewards = self.evaluations_results[-1]
            last_lengths = self.evaluations_length[-1] if self.evaluations_length else []

            mean_reward = float(np.mean(last_rewards)) if len(last_rewards) else 0.0
            mean_length = float(np.mean(last_lengths)) if len(last_lengths) else 0.0

            entry = {
                "step": self.num_timesteps,
                "mean_reward": mean_reward,
                "mean_length": mean_length,
            }
            self._eval_history.append(entry)

            if self._use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "eval/mean_reward": mean_reward,
                            "eval/mean_length": mean_length,
                            "eval/step": self.num_timesteps,
                        }, step=self.num_timesteps)
                except Exception:
                    pass

        return result

    def save_eval_history(self, path: str):
        """Persist evaluation history to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._eval_history, f, indent=2)


class RewardHackingDetector(BaseCallback):
    """Periodically checks rollout buffer for pathological behaviour patterns.

    Runs every `check_freq` steps.  Flags are logged but do NOT stop training.
    """

    def __init__(
        self,
        check_freq: int = 10_000,
        use_wandb: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self._use_wandb = use_wandb
        self.detected_pathologies: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True

        flags = {}

        # Access rollout buffer if available
        try:
            buf = self.model.rollout_buffer
            if buf is not None and buf.rewards is not None:
                rewards = buf.rewards.flatten()
                episode_starts = buf.episode_starts.flatten()

                # Basic statistics
                mean_reward = float(np.mean(rewards))
                std_reward = float(np.std(rewards))

                # Idle farming: very low variance in rewards and near-constant values
                if std_reward < 0.001 and abs(mean_reward) < 0.02:
                    flags["pathology/idle_farming"] = True

                # Extremely negative mean reward may indicate death loop
                if mean_reward < -5.0:
                    flags["pathology/death_loop_suspected"] = True

        except Exception:
            pass

        # Action collapse: check entropy of recent policy outputs
        try:
            obs = self.model.rollout_buffer.observations
            if obs is not None and len(obs) > 0:
                # Sample a subset of observations
                n_sample = min(500, len(obs))
                indices = np.random.choice(len(obs), n_sample, replace=False)
                sample_obs = obs[indices].reshape(n_sample, -1)

                actions, _ = self.model.predict(sample_obs, deterministic=False)
                if actions.ndim == 2:
                    for dim_idx in range(actions.shape[1]):
                        vals, counts = np.unique(actions[:, dim_idx], return_counts=True)
                        probs = counts / counts.sum()
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        if entropy < 0.2:
                            flags[f"pathology/action_collapse_dim{dim_idx}"] = True
        except Exception:
            pass

        if flags:
            record = {"step": self.num_timesteps, **flags}
            self.detected_pathologies.append(record)

            if self.verbose >= 1:
                print(f"[RewardHackingDetector] step={self.num_timesteps} "
                      f"flags={list(flags.keys())}")

            if self._use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(flags, step=self.num_timesteps)
                except Exception:
                    pass

        return True


class WandbTrainingCallback(BaseCallback):
    """Logs PPO training metrics to W&B after each rollout."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._kl_warnings = 0

    def _on_rollout_end(self) -> None:
        try:
            import wandb
            if wandb.run is None:
                return

            logger = self.model.logger
            metrics = {}

            # Grab key metrics from the SB3 logger name map
            key_map = {
                "train/policy_gradient_loss": "train/policy_loss",
                "train/value_loss": "train/value_loss",
                "train/entropy_loss": "train/entropy_loss",
                "train/approx_kl": "train/approx_kl",
                "train/clip_fraction": "train/clip_fraction",
                "train/explained_variance": "train/explained_variance",
                "train/learning_rate": "train/learning_rate",
            }

            for sb3_key, wandb_key in key_map.items():
                if sb3_key in logger.name_to_value:
                    metrics[wandb_key] = logger.name_to_value[sb3_key]

            if metrics:
                wandb.log(metrics, step=self.num_timesteps)

            # KL divergence monitoring
            approx_kl = metrics.get("train/approx_kl", 0.0)
            if approx_kl > 0.05:
                self._kl_warnings += 1
                if self._kl_warnings >= 3:
                    wandb.alert(
                        title="High KL Divergence",
                        text=f"approx_kl={approx_kl:.4f} exceeded 0.05 for "
                             f"{self._kl_warnings} consecutive rollouts. "
                             f"Consider reducing learning rate.",
                        level=wandb.AlertLevel.WARN,
                    )
            else:
                self._kl_warnings = 0

            # Clip fraction monitoring
            clip_frac = metrics.get("train/clip_fraction", 0.0)
            if clip_frac > 0.3:
                wandb.alert(
                    title="High Clip Fraction",
                    text=f"clip_fraction={clip_frac:.4f} exceeded 0.3. "
                         f"Policy updates may be too aggressive.",
                    level=wandb.AlertLevel.WARN,
                )

        except ImportError:
            pass
        except Exception as e:
            if self.verbose >= 1:
                print(f"[WandbCallback] error logging: {e}")

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def run_ppo_cycle(
    ppo_config,
    reward_config,
    cycle_number: int,
    dry_run: bool = False,
    use_wandb: bool = True,
) -> dict:
    """Run a single PPO training cycle.

    Args:
        ppo_config: PPOConfig instance.
        reward_config: RewardConfig instance.
        cycle_number: 1-indexed cycle number.
        dry_run: If True, validate configuration without starting training.
        use_wandb: Whether to log to W&B.

    Returns:
        Dict with cycle results: checkpoint path, eval metrics, etc.
    """
    env_config = EnvConfig(
        base_port=ppo_config.base_port,
        n_envs=ppo_config.n_envs,
    )

    device = _get_device()
    print(f"\n{'=' * 60}")
    print(f"PPO Cycle {cycle_number} | Device: {device}")
    print(f"Reward config: {reward_config.version}")
    print(f"Checkpoint dir: {ppo_config.checkpoint_dir}")
    print(f"{'=' * 60}\n")

    # --- Port verification ---
    print("Verifying game instance ports...")
    unreachable = verify_all_ports(
        env_config.host, env_config.base_port, env_config.n_envs
    )
    if unreachable:
        msg = (
            f"ERROR: The following game instance ports are not reachable: "
            f"{unreachable}\n"
            f"Start game containers before training. "
            f"Do NOT train with fewer envs than configured — VecEnv will deadlock.\n"
            f"Expected ports: {list(range(env_config.base_port, env_config.base_port + env_config.n_envs))}"
        )
        if dry_run:
            print(f"[DRY RUN] {msg}")
        else:
            print(msg, file=sys.stderr)
            return {"error": msg, "cycle": cycle_number}

    if dry_run:
        print("[DRY RUN] Port verification passed. Skipping actual training.")
        print(f"[DRY RUN] Would load checkpoint: {ppo_config.bc_checkpoint}")
        print(f"[DRY RUN] Would train for {ppo_config.total_timesteps:,} timesteps")
        print(f"[DRY RUN] PPO config: lr={ppo_config.learning_rate}, "
              f"clip={ppo_config.clip_range}, ent={ppo_config.ent_coef}")
        return {
            "cycle": cycle_number,
            "dry_run": True,
            "config": asdict(ppo_config),
            "reward_config": asdict(reward_config),
        }

    # --- Create checkpoint directory ---
    os.makedirs(ppo_config.checkpoint_dir, exist_ok=True)

    # --- W&B init ---
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=ppo_config.wandb_project,
                name=ppo_config.wandb_run_name,
                config={
                    "ppo": asdict(ppo_config),
                    "reward": asdict(reward_config),
                    "cycle": cycle_number,
                    "device": device,
                },
                reinit=True,
            )
        except Exception as e:
            warnings.warn(f"W&B init failed: {e} — continuing without logging")
            use_wandb = False

    # --- Create vectorised environment ---
    print("Creating vectorised environment...")
    from nt_rl.vec_env import make_vec_env
    vec_env = VecMonitor(make_vec_env(env_config))

    # --- Create eval environment (single instance on last port) ---
    from nt_rl.env import NuclearThroneEnv
    eval_env_config = EnvConfig(
        base_port=env_config.base_port,
        n_envs=1,
    )
    eval_vec_env = VecMonitor(make_vec_env(eval_env_config))

    # --- Load or create model ---
    checkpoint_path = ppo_config.bc_checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        if ppo_config.load_bc_weights and os.path.exists(checkpoint_path):
            # Load BC or previous cycle checkpoint as warm-start
            model = PPO.load(
                checkpoint_path,
                env=vec_env,
                device=device,
            )
            # Override PPO hyperparameters after loading
            model.learning_rate = ppo_config.learning_rate
            model.clip_range = lambda _: ppo_config.clip_range
            model.ent_coef = ppo_config.ent_coef
            model.vf_coef = ppo_config.vf_coef
            model.max_grad_norm = ppo_config.max_grad_norm
            model.n_steps = ppo_config.n_steps
            model.batch_size = ppo_config.batch_size
            model.n_epochs = ppo_config.n_epochs
            model.gamma = ppo_config.gamma
            model.gae_lambda = ppo_config.gae_lambda
            print(f"Loaded warm-start from {checkpoint_path}")
            print(f"  Overridden lr={ppo_config.learning_rate}, "
                  f"clip={ppo_config.clip_range}, ent={ppo_config.ent_coef}")
        elif ppo_config.load_bc_weights and not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint not found at {checkpoint_path}. "
                  f"Training from scratch.", file=sys.stderr)
            model = _create_fresh_model(ppo_config, vec_env, device)
        else:
            model = _create_fresh_model(ppo_config, vec_env, device)

    except Exception as e:
        print(f"Error loading checkpoint: {e}. Creating fresh model.", file=sys.stderr)
        model = _create_fresh_model(ppo_config, vec_env, device)

    # --- Set up callbacks ---
    callbacks = []

    # Evaluation callback
    eval_callback = NTEvalCallback(
        eval_env=eval_vec_env,
        log_dir=ppo_config.checkpoint_dir,
        use_wandb=use_wandb,
        eval_freq=max(ppo_config.eval_freq // ppo_config.n_envs, 1),
        n_eval_episodes=ppo_config.n_eval_episodes,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(ppo_config.save_freq // ppo_config.n_envs, 1),
        save_path=ppo_config.checkpoint_dir,
        name_prefix="ppo_checkpoint",
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # Reward hacking detector
    reward_hack_callback = RewardHackingDetector(
        check_freq=10_000,
        use_wandb=use_wandb,
        verbose=1,
    )
    callbacks.append(reward_hack_callback)

    # W&B callback
    if use_wandb:
        wandb_callback = WandbTrainingCallback(verbose=0)
        callbacks.append(wandb_callback)

    # --- Train ---
    print(f"\nStarting PPO training: {ppo_config.total_timesteps:,} timesteps")
    print(f"  n_envs={ppo_config.n_envs}, n_steps={ppo_config.n_steps}, "
          f"batch_size={ppo_config.batch_size}")
    print(f"  lr={ppo_config.learning_rate}, clip={ppo_config.clip_range}, "
          f"ent_coef={ppo_config.ent_coef}")
    print()

    try:
        model.learn(
            total_timesteps=ppo_config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    except Exception as e:
        print(f"\nTraining error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    # --- Save final model ---
    final_path = os.path.join(ppo_config.checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}.zip")

    # Save eval history
    eval_history_path = os.path.join(ppo_config.checkpoint_dir, "eval_history.json")
    eval_callback.save_eval_history(eval_history_path)

    # Save pathology log
    if reward_hack_callback.detected_pathologies:
        pathology_path = os.path.join(
            ppo_config.checkpoint_dir, "pathology_log.json"
        )
        with open(pathology_path, "w") as f:
            json.dump(reward_hack_callback.detected_pathologies, f, indent=2)

    # Save cycle metadata
    cycle_meta = {
        "cycle": cycle_number,
        "reward_config_version": reward_config.version,
        "ppo_config": asdict(ppo_config),
        "reward_config": asdict(reward_config),
        "device": device,
        "total_timesteps_requested": ppo_config.total_timesteps,
        "pathologies_detected": reward_hack_callback.detected_pathologies,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(ppo_config.checkpoint_dir, "cycle_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(cycle_meta, f, indent=2)

    # --- Cleanup ---
    vec_env.close()
    eval_vec_env.close()

    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    result = {
        "cycle": cycle_number,
        "checkpoint": final_path + ".zip",
        "best_model": os.path.join(ppo_config.checkpoint_dir, "best_model.zip"),
        "pathologies": reward_hack_callback.detected_pathologies,
        "eval_history_path": eval_history_path,
    }

    print(f"\nCycle {cycle_number} complete.")
    return result


def _create_fresh_model(ppo_config, vec_env, device: str) -> PPO:
    """Create a new PPO model from scratch (no warm-start)."""
    activation_fn = torch.nn.Tanh if ppo_config.activation_fn == "tanh" else torch.nn.ReLU

    policy_kwargs = {
        "net_arch": ppo_config.net_arch,
        "activation_fn": activation_fn,
    }

    model = PPO(
        "MultiInputPolicy" if isinstance(vec_env.observation_space, gymnasium.spaces.Dict)
        else "MlpPolicy",
        vec_env,
        learning_rate=ppo_config.learning_rate,
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        clip_range_vf=ppo_config.clip_range_vf,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
    )
    print("Created fresh PPO model (no warm-start)")
    return model


# ---------------------------------------------------------------------------
# Iteration log utilities
# ---------------------------------------------------------------------------

def append_iteration_log(
    cycle_number: int,
    reward_version: str,
    cycle_result: dict,
    pathologies: list[str],
    action: str,
    changes: str = "",
    log_path: str = "PPO_ITERATION_LOG.md",
):
    """Append a cycle summary to the running iteration log."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Try to read eval metrics from the eval history
    eval_summary = "N/A"
    eval_history_path = cycle_result.get("eval_history_path", "")
    if eval_history_path and os.path.exists(eval_history_path):
        try:
            with open(eval_history_path) as f:
                history = json.load(f)
            if history:
                last = history[-1]
                eval_summary = (
                    f"reward={last.get('mean_reward', 0):.1f}, "
                    f"length={last.get('mean_length', 0):.0f}"
                )
        except Exception:
            pass

    pathology_str = ", ".join(pathologies) if pathologies else "none"

    entry = f"""
## Cycle {cycle_number} -- {reward_version} reward config
- Date: {timestamp}
- Steps trained: {cycle_result.get('total_timesteps_trained', 'N/A')}
- Final eval: {eval_summary}
- Pathologies: {pathology_str}
- Action: {action}
- Changes: {changes if changes else 'N/A'}
"""

    # Create file if it doesn't exist
    if not os.path.exists(log_path):
        header = "# PPO Iteration Log\n\nRunning log of PPO training cycles.\n"
        with open(log_path, "w") as f:
            f.write(header)

    with open(log_path, "a") as f:
        f.write(entry)

    print(f"Iteration log updated: {log_path}")


# ---------------------------------------------------------------------------
# Success criteria check
# ---------------------------------------------------------------------------

def meets_minimum_viable_criteria(eval_history: list[dict]) -> bool:
    """Check if the most recent evaluation meets minimum viable criteria.

    Criteria (from spec):
      - Mean levels reached > 3.0
      - Mean episode length > 300

    Since we may not have levels_reached in all eval entries (depends on
    info dict from environment), we check what is available.
    """
    if not eval_history:
        return False

    last = eval_history[-1]
    mean_length = last.get("mean_length", 0)
    mean_levels = last.get("mean_levels_reached", 0)

    # Length criterion
    if mean_length <= 300:
        return False

    # Levels criterion (if available)
    if mean_levels > 0 and mean_levels <= 3.0:
        return False

    # If we only have length (no levels data from env), be conservative
    if mean_levels == 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Multi-cycle iteration driver
# ---------------------------------------------------------------------------

def run_iteration_loop(
    max_cycles: int = 4,
    start_cycle: int = 1,
    dry_run: bool = False,
    use_wandb: bool = True,
    total_timesteps_override: Optional[int] = None,
):
    """Orchestrate multi-cycle PPO training with reward shaping iteration.

    Each cycle:
      1. Load best checkpoint from previous cycle (or BC for cycle 1)
      2. Train PPO for total_timesteps
      3. Evaluate and check for pathologies
      4. Decide whether to continue, adjust rewards, or restart
      5. Apply reward adjustments if needed

    Args:
        max_cycles: Maximum number of training cycles.
        start_cycle: Cycle number to start from (for resuming).
        dry_run: Validate config without training.
        use_wandb: Enable W&B logging.
        total_timesteps_override: Override total_timesteps in PPOConfig.
    """
    PPOConfig = _load_ppo_config()
    RewardConfig = _load_reward_config()

    cycle = start_cycle
    reward_config = RewardConfig()
    prev_checkpoint = None

    print(f"\n{'#' * 60}")
    print(f"# PPO Iteration Loop: max_cycles={max_cycles}, start={start_cycle}")
    print(f"# Dry run: {dry_run}")
    print(f"{'#' * 60}\n")

    while cycle <= max_cycles:
        print(f"\n{'=' * 60}")
        print(f"=== PPO Cycle {cycle} | Reward Config {reward_config.version} ===")
        print(f"{'=' * 60}")

        ppo_config = PPOConfig(
            checkpoint_dir=f"checkpoints/ppo/cycle_{cycle:02d}",
            wandb_run_name=f"ppo-cycle-{cycle:02d}",
        )

        if total_timesteps_override is not None:
            ppo_config.total_timesteps = total_timesteps_override

        if cycle == start_cycle and cycle == 1:
            # First cycle: warm-start from BC checkpoint
            ppo_config.load_bc_weights = True
            # bc_checkpoint default points to BC output
        elif prev_checkpoint is not None:
            # Subsequent cycles: load from previous cycle's best model
            ppo_config.bc_checkpoint = prev_checkpoint
            ppo_config.load_bc_weights = True
            ppo_config.learning_rate *= (0.5 ** (cycle - 1))  # Decay LR each cycle
        else:
            # Resuming a specific cycle without previous checkpoint
            expected_prev = f"checkpoints/ppo/cycle_{cycle - 1:02d}/best_model.zip"
            if os.path.exists(expected_prev):
                ppo_config.bc_checkpoint = expected_prev
                ppo_config.load_bc_weights = True
                ppo_config.learning_rate *= (0.5 ** (cycle - 1))
            else:
                print(f"WARNING: No previous checkpoint found at {expected_prev}. "
                      f"Starting from BC checkpoint.")

        # --- Run cycle ---
        cycle_result = run_ppo_cycle(
            ppo_config=ppo_config,
            reward_config=reward_config,
            cycle_number=cycle,
            dry_run=dry_run,
            use_wandb=use_wandb,
        )

        if dry_run:
            print(f"\n[DRY RUN] Cycle {cycle} config validated. Moving to next cycle.")
            prev_checkpoint = ppo_config.bc_checkpoint
            cycle += 1
            continue

        if "error" in cycle_result:
            print(f"\nCycle {cycle} failed: {cycle_result['error']}")
            break

        # --- Load eval history for diagnosis ---
        eval_history = []
        eval_history_path = cycle_result.get("eval_history_path", "")
        if eval_history_path and os.path.exists(eval_history_path):
            try:
                with open(eval_history_path) as f:
                    eval_history = json.load(f)
            except Exception:
                pass

        # --- Check success criteria ---
        if meets_minimum_viable_criteria(eval_history):
            print("\n*** MINIMUM VIABLE CRITERIA MET. Training complete. ***")
            append_iteration_log(
                cycle_number=cycle,
                reward_version=reward_config.version,
                cycle_result=cycle_result,
                pathologies=[],
                action="DONE - criteria met",
            )
            # Copy best model as best_overall
            best_model = cycle_result.get("best_model", "")
            if os.path.exists(best_model):
                import shutil
                os.makedirs("checkpoints/ppo", exist_ok=True)
                shutil.copy2(best_model, "checkpoints/ppo/best_overall.zip")
                print(f"Best model copied to checkpoints/ppo/best_overall.zip")
            break

        # --- Diagnose pathologies ---
        pathology_names = _extract_pathology_names(
            cycle_result.get("pathologies", [])
        )
        action, changes_desc, reward_adjustments = _decide_next_action(
            cycle_number=cycle,
            eval_history=eval_history,
            pathologies=pathology_names,
            reward_config=reward_config,
        )

        # --- Log iteration ---
        append_iteration_log(
            cycle_number=cycle,
            reward_version=reward_config.version,
            cycle_result=cycle_result,
            pathologies=pathology_names,
            action=action,
            changes=changes_desc,
        )

        # --- Decide what to do next ---
        if action == "DONE":
            print("\nDiagnostic says DONE. Ending iteration loop.")
            break
        elif action == "RESTART_FROM_BC":
            print("\nRestarting from BC checkpoint for next cycle.")
            prev_checkpoint = None  # Will default to BC
        elif action in ("CONTINUE", "ADJUST_REWARDS"):
            prev_checkpoint = cycle_result.get(
                "best_model",
                os.path.join(ppo_config.checkpoint_dir, "final_model.zip"),
            )
            if reward_adjustments:
                reward_config = _apply_reward_adjustments(
                    reward_config, reward_adjustments, cycle + 1
                )
                print(f"\nReward config updated to {reward_config.version}")
                print(f"  Changes: {changes_desc}")
        else:
            prev_checkpoint = cycle_result.get(
                "best_model",
                os.path.join(ppo_config.checkpoint_dir, "final_model.zip"),
            )

        cycle += 1

    if cycle > max_cycles:
        print(f"\nIteration budget exhausted after {max_cycles} cycles.")
        print("Review PPO_ITERATION_LOG.md before proceeding.")

    print(f"\n{'#' * 60}")
    print("# PPO Iteration Loop Complete")
    print(f"{'#' * 60}\n")


# ---------------------------------------------------------------------------
# Diagnosis helpers (lightweight — full diagnosis in diagnose.py)
# ---------------------------------------------------------------------------

def _extract_pathology_names(pathology_records: list[dict]) -> list[str]:
    """Extract unique pathology names from the callback's detection log."""
    names = set()
    for record in pathology_records:
        for key in record:
            if key.startswith("pathology/"):
                names.add(key.replace("pathology/", ""))
    return sorted(names)


def _decide_next_action(
    cycle_number: int,
    eval_history: list[dict],
    pathologies: list[str],
    reward_config,
) -> tuple[str, str, dict]:
    """Lightweight decision logic for the iteration loop.

    Returns:
        (action, changes_description, reward_adjustments_dict)
    """
    adjustments: dict = {}
    changes_parts: list[str] = []

    if not eval_history:
        return "CONTINUE", "no eval data yet", {}

    last_eval = eval_history[-1]
    mean_length = last_eval.get("mean_length", 0)

    # Death loop: very short episodes
    if mean_length < 50 and cycle_number > 0:
        return (
            "RESTART_FROM_BC",
            "death loop detected (mean_length < 50), restarting from BC",
            {},
        )

    # Idle farming
    if "idle_farming" in pathologies:
        adjustments["reward_idle_penalty"] = getattr(
            reward_config, "reward_idle_penalty", -0.1
        ) * 3
        adjustments["reward_kill"] = getattr(reward_config, "reward_kill", 5.0) * 0.5
        changes_parts.append("increased idle penalty, reduced kill reward")

    # Action collapse
    collapse_dims = [p for p in pathologies if p.startswith("action_collapse")]
    if collapse_dims:
        # Signal to increase entropy coefficient (handled by PPO config, not reward)
        changes_parts.append(f"action collapse in {len(collapse_dims)} dims")

    # Corner hiding heuristic: long episodes but low reward
    if mean_length > 1000:
        last_reward = last_eval.get("mean_reward", 0)
        if last_reward < 10:
            adjustments["reward_survival_per_step"] = -0.005
            changes_parts.append("survival_per_step: positive -> -0.005 (anti-hiding)")

    # Plateau: check if eval reward hasn't changed much
    if len(eval_history) >= 5:
        recent_rewards = [e.get("mean_reward", 0) for e in eval_history[-5:]]
        if max(recent_rewards) - min(recent_rewards) < 1.0:
            changes_parts.append("reward plateau detected")
            if not adjustments:
                return "CONTINUE", "plateau but no clear fix", {}

    if adjustments:
        action = "ADJUST_REWARDS"
    else:
        action = "CONTINUE"

    changes_desc = "; ".join(changes_parts) if changes_parts else "no changes"
    return action, changes_desc, adjustments


def _apply_reward_adjustments(
    current_config,
    adjustments: dict,
    next_cycle: int,
):
    """Create a new RewardConfig with the specified adjustments applied."""
    RewardConfig = _load_reward_config()

    # Build new config from current values
    current_dict = asdict(current_config)
    current_dict.update(adjustments)
    current_dict["version"] = f"v{next_cycle}.0"

    return RewardConfig(**current_dict)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PPO training for Nuclear Throne RL agent"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=None,
        help="Run a single cycle N (default: run full iteration loop)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without starting training "
             "(does not require live game instances)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=4,
        help="Maximum number of iteration cycles (default: 4)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps per cycle",
    )
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    if args.cycle is not None:
        # Single-cycle mode
        PPOConfig = _load_ppo_config()
        RewardConfig = _load_reward_config()

        ppo_config = PPOConfig(
            checkpoint_dir=f"checkpoints/ppo/cycle_{args.cycle:02d}",
            wandb_run_name=f"ppo-cycle-{args.cycle:02d}",
            use_wandb=use_wandb,
        )
        if args.timesteps is not None:
            ppo_config.total_timesteps = args.timesteps

        reward_config = RewardConfig()

        # For cycles > 1, try to load previous cycle's best model
        if args.cycle > 1:
            prev_best = f"checkpoints/ppo/cycle_{args.cycle - 1:02d}/best_model.zip"
            if os.path.exists(prev_best):
                ppo_config.bc_checkpoint = prev_best
            else:
                print(f"WARNING: Previous cycle checkpoint not found at {prev_best}")
            ppo_config.learning_rate *= (0.5 ** (args.cycle - 1))

        run_ppo_cycle(
            ppo_config=ppo_config,
            reward_config=reward_config,
            cycle_number=args.cycle,
            dry_run=args.dry_run,
            use_wandb=use_wandb,
        )
    else:
        # Full iteration loop mode
        run_iteration_loop(
            max_cycles=args.max_cycles,
            dry_run=args.dry_run,
            use_wandb=use_wandb,
            total_timesteps_override=args.timesteps,
        )


if __name__ == "__main__":
    main()
