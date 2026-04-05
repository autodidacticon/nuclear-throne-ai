"""Behavioural cloning training loop using the imitation library.

Usage:
    python -m nt_rl.bc.train [--no-wandb] [--epochs 10] [--demo-dir demonstrations]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

from nt_rl.bc.config import BCConfig
from nt_rl.bc.dataset import DemonstrationDataset
from nt_rl.bc.evaluate import evaluate_policy, action_distribution_report
from nt_rl.config import EnvConfig
from nt_rl.env import NuclearThroneEnv


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    warnings.warn("Neither CUDA nor MPS available — training on CPU")
    return "cpu"


def _make_eval_env(bc_config: BCConfig, env_config: EnvConfig) -> NuclearThroneEnv:
    return NuclearThroneEnv(port=bc_config.eval_port, config=env_config)


def train(bc_config: BCConfig | None = None, env_config: EnvConfig | None = None):
    """Run the full BC training pipeline."""
    bc_config = bc_config or BCConfig()
    env_config = env_config or EnvConfig()
    device = _get_device()

    # --- Load dataset ---
    print("=" * 60)
    print("Loading demonstrations...")
    try:
        dataset = DemonstrationDataset(bc_config.demonstrations_dir, env_config)
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    dataset.print_statistics()
    train_ds, val_ds = dataset.split(bc_config.train_ratio)
    print(f"\nTrain: {train_ds.n_transitions:,} transitions ({train_ds.n_episodes} episodes)")
    print(f"Val:   {val_ds.n_transitions:,} transitions ({val_ds.n_episodes} episodes)")

    # --- Convert to imitation trajectories ---
    print("\nConverting to imitation trajectories...")
    train_trajectories = train_ds.to_imitation_trajectories()
    val_trajectories = val_ds.to_imitation_trajectories()

    # --- Build BC trainer ---
    from imitation.algorithms.bc import BC
    from imitation.data.types import TransitionsMinimal
    from stable_baselines3.common.policies import ActorCriticPolicy
    import gymnasium

    obs_space = gymnasium.spaces.Box(
        low=-1.0, high=1.0, shape=(env_config.obs_dim,), dtype=np.float32
    )
    act_space = gymnasium.spaces.MultiDiscrete(
        [env_config.n_move_dirs, env_config.n_aim_angles, 2, 2]
    )

    activation_fn = torch.nn.Tanh if bc_config.activation_fn == "tanh" else torch.nn.ReLU

    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: bc_config.learning_rate,
        net_arch=bc_config.net_arch,
        activation_fn=activation_fn,
    )

    # Build transitions from train trajectories
    from imitation.data.rollout import flatten_trajectories
    train_transitions = flatten_trajectories(train_trajectories)

    bc_trainer = BC(
        observation_space=obs_space,
        action_space=act_space,
        demonstrations=train_transitions,
        policy=policy,
        device=device,
        batch_size=bc_config.batch_size,
    )

    # --- W&B init ---
    wandb_run = None
    if bc_config.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=bc_config.wandb_project,
                name=bc_config.wandb_run_name,
                config={
                    "bc": vars(bc_config),
                    "env": vars(env_config),
                    "device": device,
                    "n_train_transitions": train_ds.n_transitions,
                    "n_val_transitions": val_ds.n_transitions,
                },
            )
        except Exception as e:
            warnings.warn(f"W&B init failed: {e} — continuing without logging")
            bc_config.use_wandb = False

    # --- Evaluation env ---
    eval_env = None
    try:
        eval_env = _make_eval_env(bc_config, env_config)
    except Exception as e:
        warnings.warn(f"Could not create eval env: {e} — skipping live evaluation")

    # --- Training loop ---
    print(f"\n{'=' * 60}")
    print(f"Training BC for {bc_config.n_epochs} epochs on {device}")
    print(f"{'=' * 60}\n")

    os.makedirs(bc_config.checkpoint_dir, exist_ok=True)
    training_log = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Compute initial validation loss
    val_transitions = flatten_trajectories(val_trajectories)
    initial_val_loss = _compute_val_loss(bc_trainer, val_transitions, bc_config.batch_size)
    print(f"Initial val loss: {initial_val_loss:.4f}\n")

    for epoch in range(1, bc_config.n_epochs + 1):
        # Train one epoch
        # imitation's BC.train() takes n_batches, compute from dataset size
        n_batches = max(1, train_ds.n_transitions // bc_config.batch_size)
        bc_trainer.train(n_batches=n_batches)

        # Validation loss
        val_loss = _compute_val_loss(bc_trainer, val_transitions, bc_config.batch_size)

        # Action accuracy on validation set
        val_accuracy = _compute_action_accuracy(bc_trainer.policy, val_ds)

        epoch_log = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "val_action_accuracy": float(val_accuracy),
        }

        # Eval on mock env
        if eval_env is not None and epoch % bc_config.eval_every_n_epochs == 0:
            try:
                eval_results = evaluate_policy(
                    bc_trainer.policy, eval_env, bc_config.n_eval_episodes
                )
                epoch_log["eval_mean_reward"] = eval_results["mean_reward"]
                epoch_log["eval_mean_length"] = eval_results["mean_length"]
            except Exception as e:
                warnings.warn(f"Evaluation failed at epoch {epoch}: {e}")

        training_log.append(epoch_log)

        # Log to W&B
        if wandb_run is not None:
            wandb.log({f"val/{k}" if k != "epoch" else k: v for k, v in epoch_log.items()})

        # Checkpointing
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if bc_config.save_best_only:
                _save_checkpoint(bc_trainer.policy, bc_config.checkpoint_dir)
        else:
            epochs_without_improvement += 1

        if not bc_config.save_best_only:
            _save_checkpoint(bc_trainer.policy, bc_config.checkpoint_dir, suffix=f"_epoch{epoch}")

        status = "* (best)" if improved else ""
        print(f"Epoch {epoch:3d}: val_loss={val_loss:.4f}  "
              f"accuracy={val_accuracy:.1%}  "
              f"no_improve={epochs_without_improvement} {status}")

        # Convergence warning
        if epochs_without_improvement >= 3:
            warnings.warn(f"Val loss has not improved for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= 5:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # --- Final save ---
    print(f"\n{'=' * 60}")
    print("Saving final checkpoint...")
    _save_checkpoint(bc_trainer.policy, bc_config.checkpoint_dir)

    # Save training log
    log_path = os.path.join(bc_config.checkpoint_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    # --- Convergence checks ---
    print(f"\n{'=' * 60}")
    print("Convergence checks:")
    checks = _run_convergence_checks(
        bc_trainer.policy, dataset, val_ds, initial_val_loss, best_val_loss,
        eval_env, bc_config, env_config,
    )
    for name, result in checks.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {name}: {result['detail']}")

    if eval_env is not None:
        eval_env.close()

    if wandb_run is not None:
        wandb.finish()

    # Save summary
    _write_summary(dataset, bc_config, training_log, checks, device)
    print(f"\nDone. Checkpoint: {bc_config.checkpoint_dir}/final.zip")


def _compute_val_loss(bc_trainer, val_transitions, batch_size: int) -> float:
    """Compute BC loss on validation transitions."""
    policy = bc_trainer.policy
    policy.eval()

    obs = torch.as_tensor(val_transitions.obs, device=policy.device).float()
    acts = torch.as_tensor(val_transitions.acts, device=policy.device).long()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(obs), batch_size):
            batch_obs = obs[i:i + batch_size]
            batch_acts = acts[i:i + batch_size]
            _, log_prob, _ = policy.evaluate_actions(batch_obs, batch_acts)
            total_loss += -log_prob.mean().item()
            n_batches += 1

    policy.train()
    return total_loss / max(n_batches, 1)


def _compute_action_accuracy(policy, dataset: DemonstrationDataset) -> float:
    """Fraction of actions matching demonstrations exactly (all 4 dims)."""
    policy.eval()
    n_samples = min(len(dataset.obs), 5000)
    indices = np.random.choice(len(dataset.obs), n_samples, replace=False)

    correct = 0
    for idx in indices:
        pred, _ = policy.predict(dataset.obs[idx], deterministic=True)
        if np.array_equal(pred, dataset.actions[idx]):
            correct += 1

    policy.train()
    return correct / n_samples


def _save_checkpoint(policy, checkpoint_dir: str, suffix: str = ""):
    """Save policy in both SB3 .zip and raw PyTorch formats."""
    policy.save(os.path.join(checkpoint_dir, f"final{suffix}"))
    torch.save(
        policy.state_dict(),
        os.path.join(checkpoint_dir, f"final_state_dict{suffix}.pt"),
    )


def _run_convergence_checks(
    policy, full_dataset, val_dataset, initial_val_loss, best_val_loss,
    eval_env, bc_config, env_config,
) -> dict:
    """Run all 4 convergence checks from the agent_05 spec."""
    checks = {}

    # Check 1: Loss convergence (final < 80% of initial)
    ratio = best_val_loss / initial_val_loss if initial_val_loss > 0 else 1.0
    checks["loss_convergence"] = {
        "passed": ratio < 0.8,
        "detail": f"best/initial = {ratio:.2%} (threshold: <80%)",
    }

    # Check 2: Action diversity (no single value >80%)
    n_samples = min(len(val_dataset.obs), 1000)
    rng = np.random.RandomState(123)
    sample_indices = rng.choice(len(val_dataset.obs), n_samples, replace=False)
    report = action_distribution_report(policy, val_dataset.obs[sample_indices])
    max_dominant = max(r["dominant_pct"] for r in report.values())
    checks["action_diversity"] = {
        "passed": max_dominant <= 80,
        "detail": f"max dominant = {max_dominant:.0f}% (threshold: <=80%)",
    }

    # Check 3: Non-random baseline (BC mean length > random * 1.2)
    if eval_env is not None:
        try:
            bc_eval = evaluate_policy(policy, eval_env, n_episodes=20)
            # Random baseline
            random_lengths = []
            for _ in range(20):
                obs, _ = eval_env.reset()
                length = 0
                done = False
                while not done:
                    action = eval_env.action_space.sample()
                    obs, _, term, trunc, _ = eval_env.step(action)
                    length += 1
                    done = term or trunc
                random_lengths.append(length)

            random_mean = float(np.mean(random_lengths))
            bc_mean = bc_eval["mean_length"]
            threshold = random_mean * 1.2
            checks["non_random_baseline"] = {
                "passed": bc_mean > threshold,
                "detail": f"BC={bc_mean:.0f} vs random={random_mean:.0f}*1.2={threshold:.0f}",
            }
        except Exception as e:
            checks["non_random_baseline"] = {
                "passed": False,
                "detail": f"Evaluation failed: {e}",
            }
    else:
        checks["non_random_baseline"] = {
            "passed": False,
            "detail": "No eval env available — skipped",
        }

    # Check 4: Observation coverage (>=3 distinct levels)
    level_idx = 11
    levels = np.unique(np.round(full_dataset.obs[:, level_idx] * env_config.max_level))
    n_levels = len(levels)
    checks["observation_coverage"] = {
        "passed": n_levels >= 3,
        "detail": f"{n_levels} distinct level(s) in dataset: {sorted(levels.astype(int).tolist())}",
    }

    return checks


def _write_summary(dataset, bc_config, training_log, checks, device):
    """Write BC_TRAINING_SUMMARY.md."""
    path = "BC_TRAINING_SUMMARY.md"
    lines = [
        "# BC Training Summary",
        "",
        "## Dataset Statistics",
        f"- Transitions: {dataset.n_transitions:,}",
        f"- Episodes: {dataset.n_episodes}",
        "",
        "## Training Configuration",
        f"- Device: {device}",
        f"- Epochs: {bc_config.n_epochs}",
        f"- Batch size: {bc_config.batch_size}",
        f"- Learning rate: {bc_config.learning_rate}",
        f"- Architecture: {bc_config.net_arch}",
        f"- Activation: {bc_config.activation_fn}",
        "",
        "## Per-Epoch Metrics",
        "",
        "| Epoch | Val Loss | Accuracy |",
        "|-------|----------|----------|",
    ]
    for entry in training_log:
        lines.append(
            f"| {entry['epoch']} | {entry['val_loss']:.4f} | "
            f"{entry.get('val_action_accuracy', 0):.1%} |"
        )
    lines.extend([
        "",
        "## Convergence Checks",
        "",
    ])
    for name, result in checks.items():
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"- **[{status}]** {name}: {result['detail']}")

    lines.extend([
        "",
        "## Recommendation for Agent 06",
        f"- Suggested PPO starting learning rate: {bc_config.learning_rate}",
        "",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train BC policy from demonstrations")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--epochs", type=int, default=None, help="Override n_epochs")
    parser.add_argument("--demo-dir", type=str, default=None, help="Demonstrations directory")
    args = parser.parse_args()

    bc_config = BCConfig()
    if args.no_wandb:
        bc_config.use_wandb = False
    if args.epochs is not None:
        bc_config.n_epochs = args.epochs
    if args.demo_dir is not None:
        bc_config.demonstrations_dir = args.demo_dir

    train(bc_config)


if __name__ == "__main__":
    main()
