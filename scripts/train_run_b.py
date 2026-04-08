"""BC training Run B: medium net [128,128], lr=1e-3, batch=512"""
import sys; sys.path.insert(0, ".")
from nt_rl.bc.config import BCConfig
from nt_rl.bc.train import train

cfg = BCConfig()
cfg.net_arch = [128, 128]
cfg.learning_rate = 1e-3
cfg.batch_size = 512
cfg.activation_fn = "tanh"
cfg.n_epochs = 10
cfg.use_wandb = False
cfg.checkpoint_dir = "checkpoints/bc_run_b"
cfg.demonstrations_dir = "demonstrations"

train(cfg)
