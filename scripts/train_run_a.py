"""Run A: BC training with [64,64] tanh network, lr=3e-4, batch=256, 10 epochs."""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nt_rl.bc.config import BCConfig
from nt_rl.bc.train import train

config = BCConfig(
    net_arch=[64, 64],
    learning_rate=3e-4,
    batch_size=256,
    activation_fn="tanh",
    n_epochs=10,
    checkpoint_dir="checkpoints/bc_run_a",
    demonstrations_dir="demonstrations",
    use_wandb=False,
)

if __name__ == "__main__":
    train(config)
