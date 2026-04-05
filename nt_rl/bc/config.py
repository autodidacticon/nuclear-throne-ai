"""BC training configuration. All hyperparameters live here."""

from dataclasses import dataclass, field


@dataclass
class BCConfig:
    # Training
    n_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 3e-4
    l2_reg: float = 1e-5
    grad_clip: float = 0.5

    # LR schedule
    lr_schedule: str = "cosine"  # "cosine" | "linear" | "constant"
    warmup_steps: int = 500

    # Policy architecture (must match what PPO will use in Agent 06)
    net_arch: list = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"  # "tanh" | "relu"

    # Evaluation
    eval_every_n_epochs: int = 1
    n_eval_episodes: int = 20
    eval_port: int = 17777  # Mock server port for eval

    # Checkpointing
    checkpoint_dir: str = "checkpoints/bc_policy"
    save_best_only: bool = True

    # Logging
    use_wandb: bool = True
    wandb_project: str = "nt-rl"
    wandb_run_name: str = "bc-training"
    log_every_n_steps: int = 50

    # Dataset
    demonstrations_dir: str = "demonstrations"
    train_ratio: float = 0.9
