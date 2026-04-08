"""PPO training configuration. All hyperparameters live here."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PPOConfig:
    # Initialisation
    # BC Run C: [256,256] tanh, val_loss=0.636, 70.4% accuracy
    bc_checkpoint: str = "checkpoints/bc_run_c/final.zip"
    load_bc_weights: bool = True

    # Environment
    n_envs: int = 4                    # Must match running Docker containers
    base_port: int = 7777

    # PPO core hyperparameters
    learning_rate: float = 3e-5        # 10x lower than BC's 1e-4 to preserve warm-start
    n_steps: int = 2048                # Steps per env per rollout
    batch_size: int = 256
    n_epochs: int = 10                 # PPO update epochs per rollout
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01            # Entropy bonus — prevent premature convergence
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
    # CRITICAL: Must match BC Run C architecture ([256,256] tanh)
    net_arch: list = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"
