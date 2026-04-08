"""PPO training with BC warm-start against live Nuclear Throne.

Supports single or multiple game instances via UDP bridge.

Usage:
    .venv/bin/python3 scripts/ppo_single.py                    # 1 instance, localhost
    .venv/bin/python3 scripts/ppo_single.py --n-envs 4         # 4 instances, ports 7777-7780
    NT_HOST=host.docker.internal .venv/bin/python3 scripts/ppo_single.py  # from devcontainer
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from nt_rl.config import EnvConfig
from nt_rl.env import NuclearThroneEnv
from nt_rl.deepsets_policy import DeepSetsExtractor, DEEPSETS_FEATURES_DIM
from nt_rl.kl_ppo import KLRegularizedPPO


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for Nuclear Throne")
    parser.add_argument("--host", default=os.environ.get("NT_HOST", "127.0.0.1"))
    parser.add_argument("--base-port", type=int, default=7777)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--bc-checkpoint", default="checkpoints/bc_balanced/best")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ppo/cycle_01")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--kl-reg",
        action="store_true",
        help="Use KL-regularized PPO against a frozen BC reference policy.",
    )
    parser.add_argument(
        "--kl-coef-start",
        type=float,
        default=0.5,
        help="Initial KL coefficient (only used with --kl-reg).",
    )
    parser.add_argument(
        "--kl-coef-end",
        type=float,
        default=0.05,
        help="Final KL coefficient after annealing (only used with --kl-reg).",
    )
    parser.add_argument(
        "--kl-anneal-steps",
        type=int,
        default=2_000_000,
        help="Number of env steps over which to anneal the KL coefficient.",
    )
    return parser.parse_args()


def make_env(host, port):
    """Factory for a single NuclearThroneEnv."""
    def _init():
        config = EnvConfig(host=host, base_port=port)
        return NuclearThroneEnv(port=port, config=config)
    return _init


def verify_game(host, port):
    """Check if game instance is responding on UDP."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(3)
        s.sendto(b'{"type":"action","move_dir":0,"moving":false,"aim_dir":0,"fire":false,"spec":false}', (host, port))
        s.recvfrom(65536)
        s.close()
        return True
    except Exception:
        return False


def main():
    args = parse_args()

    # N_STEPS scales with n_envs: total rollout = n_steps * n_envs
    n_steps = 128
    batch_size = min(64, n_steps * args.n_envs)

    print("=" * 60)
    print("PPO Nuclear Throne Training")
    print(f"  Host: {args.host}, Ports: {args.base_port}-{args.base_port + args.n_envs - 1}")
    print(f"  Envs: {args.n_envs}, Device: {args.device}")
    print(f"  BC checkpoint: {args.bc_checkpoint}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  LR: {args.lr}, N_STEPS: {n_steps}, BATCH: {batch_size}")
    print(f"  Rollout buffer: {n_steps * args.n_envs} steps/rollout")
    print("=" * 60)

    # Verify all game instances
    print("\nVerifying game instances...")
    all_ok = True
    for i in range(args.n_envs):
        port = args.base_port + i
        ok = verify_game(args.host, port)
        status = "OK" if ok else "UNREACHABLE"
        print(f"  Port {port}: {status}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nERROR: Not all game instances are reachable.")
        print("Start the game with agent_mode.txt before running this script.")
        sys.exit(1)

    # Create vectorized environment
    env_fns = [make_env(args.host, args.base_port + i) for i in range(args.n_envs)]
    # SubprocVecEnv runs each env in its own process — true parallelism
    # DummyVecEnv fallback for single env (subprocess overhead not worth it)
    if args.n_envs > 1:
        vec_env = VecMonitor(SubprocVecEnv(env_fns))
    else:
        vec_env = VecMonitor(DummyVecEnv(env_fns))

    # Shared PPO / policy kwargs (used by both vanilla PPO and KL-regularized PPO)
    policy_kwargs = {
        "features_extractor_class": DeepSetsExtractor,
        "features_extractor_kwargs": {"features_dim": DEEPSETS_FEATURES_DIM},
        "net_arch": [256, 256],
        "activation_fn": torch.nn.Tanh,
    }
    ppo_kwargs = dict(
        policy=ActorCriticPolicy,
        env=vec_env,
        learning_rate=args.lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=args.device,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    # Load BC weights up-front so we can use them for both the trainer policy
    # and (optionally) the frozen reference policy.
    bc_weights = args.bc_checkpoint + "_state_dict.pt"
    bc_sd = None
    if os.path.exists(bc_weights):
        print(f"Loading BC policy from {bc_weights}...")
        bc_sd = torch.load(bc_weights, map_location=args.device, weights_only=True)
    else:
        print(f"  BC weights not found at {bc_weights} — training from scratch")
        if args.kl_reg:
            print("  --kl-reg requires a BC checkpoint; aborting.")
            sys.exit(1)

    if args.kl_reg:
        print(
            f"\nCreating KL-regularized PPO model with [256,256] tanh policy "
            f"(kl_coef {args.kl_coef_start} -> {args.kl_coef_end} over "
            f"{args.kl_anneal_steps:,} steps)..."
        )

        # Build a frozen reference policy with the same architecture as the
        # trainer policy, load BC weights into it, then hand it to
        # KLRegularizedPPO. The reference policy must match the trainer
        # architecture and live on the same device.
        ref_policy = ActorCriticPolicy(
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            lr_schedule=lambda _: 0.0,  # Frozen — no optimization
            **policy_kwargs,
        )
        ref_policy.load_state_dict(bc_sd, strict=False)
        ref_policy.to(args.device)

        model = KLRegularizedPPO(
            reference_policy=ref_policy,
            kl_coef_start=args.kl_coef_start,
            kl_coef_end=args.kl_coef_end,
            kl_anneal_steps=args.kl_anneal_steps,
            **ppo_kwargs,
        )
    else:
        print(f"\nCreating PPO model with [256,256] tanh policy...")
        model = PPO(**ppo_kwargs)

    if bc_sd is not None:
        model.policy.load_state_dict(bc_sd, strict=False)
        print(f"  Loaded {len(bc_sd)} weights into trainer policy")

    # Callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1000),
        save_path=args.checkpoint_dir,
        name_prefix="ppo",
    )

    # Train
    print(f"\nStarting PPO training for {args.timesteps:,} timesteps...")
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")

    vec_env.close()


if __name__ == "__main__":
    main()
