"""BC training with class-balanced per-dimension loss.

The imitation library's BC trainer uses sum-of-log-probs which lets the policy
collapse to majority-class predictions on imbalanced action dimensions.

This script trains the same ActorCriticPolicy but with:
1. Per-dimension cross-entropy loss with inverse-frequency class weights
2. Per-dimension accuracy tracking
3. Action diversity monitoring every epoch
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gymnasium
from stable_baselines3.common.policies import ActorCriticPolicy

from nt_rl.bc.dataset import DemonstrationDataset
from nt_rl.config import EnvConfig
from nt_rl.deepsets_policy import DeepSetsExtractor, DEEPSETS_FEATURES_DIM


# --- Config ---
NET_ARCH = [256, 256]
ACTIVATION = torch.nn.Tanh
LR = 1e-4
N_EPOCHS = 20
BATCH_SIZE = 512
CHECKPOINT_DIR = "checkpoints/bc_balanced"
DEMO_DIR = "demonstrations"
DEVICE = "cpu"

# Action space dimensions
DIM_NAMES = ["move_dir", "aim_bin", "shoot", "special"]
DIM_SIZES = [9, 24, 2, 2]


def compute_class_weights(actions: np.ndarray) -> list[torch.Tensor]:
    """Compute class weights for each action dimension.

    Strategy:
    - move_dir (9 classes): uniform — the 28% stationary rate is real gameplay
    - aim_bin (24 classes): sqrt inverse-frequency — the only dimension with
      meaningful class imbalance that should be corrected (some angles under-represented)
    - shoot (2 classes): uniform — the 86/14 split is real gameplay
    - special (2 classes): uniform — the 99/1 split is intentional
    """
    weights = []
    for dim, size in enumerate(DIM_SIZES):
        counts = np.bincount(actions[:, dim], minlength=size).astype(np.float32)
        freq = counts / counts.sum()

        if DIM_NAMES[dim] == "aim_bin":
            # Only aim needs reweighting — some angles are under-represented
            w = 1.0 / np.sqrt(freq + 1e-3)
            w = w / w.sum() * size
        else:
            # All other dimensions: uniform weights — imbalance is meaningful
            w = np.ones(size, dtype=np.float32)

        weights.append(torch.tensor(w, device=DEVICE))
        dominant = freq.max() * 100
        print(f"  {DIM_NAMES[dim]:>10}: dominant={dominant:.0f}%, weights=[{', '.join(f'{x:.2f}' for x in w[:4])}{'...' if len(w)>4 else ''}]")
    return weights


def extract_action_logits(policy, obs):
    """Run obs through the policy network and get per-dimension logits.

    SB3's MultiCategoricalDistribution splits the action_net output into
    chunks of [9, 24, 2, 2] for each dimension.
    """
    features = policy.extract_features(obs, policy.features_extractor)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    logits = policy.action_net(latent_pi)
    # Split into per-dimension logits
    split_logits = torch.split(logits, DIM_SIZES, dim=-1)
    return split_logits


def balanced_loss(policy, obs, actions, class_weights):
    """Per-dimension weighted cross-entropy loss."""
    split_logits = extract_action_logits(policy, obs)
    total_loss = 0.0
    for dim, (logits, weight) in enumerate(zip(split_logits, class_weights)):
        targets = actions[:, dim].long()
        loss = F.cross_entropy(logits, targets, weight=weight)
        total_loss += loss
    return total_loss


def compute_per_dim_accuracy(policy, obs, actions):
    """Accuracy for each action dimension separately."""
    split_logits = extract_action_logits(policy, obs)
    accs = {}
    for dim, logits in enumerate(split_logits):
        preds = logits.argmax(dim=-1)
        targets = actions[:, dim].long()
        accs[DIM_NAMES[dim]] = (preds == targets).float().mean().item()
    return accs


def check_action_diversity(policy, obs_sample):
    """Check if the policy outputs diverse actions."""
    with torch.no_grad():
        split_logits = extract_action_logits(policy, obs_sample)
        for dim, logits in enumerate(split_logits):
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            counts = torch.bincount(preds, minlength=DIM_SIZES[dim])
            dominant = counts.max().item() / len(preds) * 100
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
            print(f"    {DIM_NAMES[dim]:>10}: dominant={dominant:.0f}%, entropy={entropy:.2f}")


def main():
    print("=" * 60)
    print("BC Training with Class-Balanced Loss")
    print("=" * 60)

    env_config = EnvConfig()

    # Load data
    print("\nLoading demonstrations...")
    dataset = DemonstrationDataset(DEMO_DIR, env_config)
    dataset.print_statistics()
    train_ds, val_ds = dataset.split(0.9)
    print(f"Train: {train_ds.n_transitions:,}, Val: {val_ds.n_transitions:,}")

    # Class weights
    print("\nClass weights:")
    class_weights = compute_class_weights(train_ds.actions)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_ds.obs, device=DEVICE),
            torch.tensor(train_ds.actions, device=DEVICE),
        ),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_obs = torch.tensor(val_ds.obs, device=DEVICE)
    val_acts = torch.tensor(val_ds.actions, device=DEVICE)

    # Create policy
    obs_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(env_config.obs_dim,), dtype=np.float32)
    act_space = gymnasium.spaces.MultiDiscrete(DIM_SIZES)

    policy_kwargs = {
        "features_extractor_class": DeepSetsExtractor,
        "features_extractor_kwargs": {"features_dim": DEEPSETS_FEATURES_DIM},
        "net_arch": NET_ARCH,
        "activation_fn": ACTIVATION,
    }

    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: LR,
        **policy_kwargs,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    training_log = []

    print(f"\nTraining for {N_EPOCHS} epochs...")
    print()

    for epoch in range(1, N_EPOCHS + 1):
        # --- Train ---
        policy.train()
        epoch_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in train_loader:
            loss = balanced_loss(policy, obs_batch, act_batch, class_weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        # --- Validate ---
        policy.eval()
        with torch.no_grad():
            val_loss = balanced_loss(policy, val_obs, val_acts, class_weights).item()
            accs = compute_per_dim_accuracy(policy, val_obs, val_acts)

        scheduler.step()

        # Checkpoint
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            policy.save(os.path.join(CHECKPOINT_DIR, "best"))
            torch.save(policy.state_dict(), os.path.join(CHECKPOINT_DIR, "best_state_dict.pt"))

        # Log
        acc_str = " ".join(f"{k}={v:.0%}" for k, v in accs.items())
        marker = "* (best)" if improved else ""
        print(f"Epoch {epoch:2d}: train={train_loss:.3f} val={val_loss:.3f} {acc_str} {marker}")

        # Action diversity check
        if epoch % 5 == 0 or epoch == 1:
            print("  Diversity check:")
            with torch.no_grad():
                sample_idx = np.random.choice(len(val_obs), min(1000, len(val_obs)), replace=False)
                check_action_diversity(policy, val_obs[sample_idx])

        training_log.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **{f"acc_{k}": v for k, v in accs.items()},
        })

    # Save final
    policy.save(os.path.join(CHECKPOINT_DIR, "final"))
    torch.save(policy.state_dict(), os.path.join(CHECKPOINT_DIR, "final_state_dict.pt"))
    with open(os.path.join(CHECKPOINT_DIR, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.3f}")
    print(f"Checkpoint: {CHECKPOINT_DIR}/best_state_dict.pt")


if __name__ == "__main__":
    main()
