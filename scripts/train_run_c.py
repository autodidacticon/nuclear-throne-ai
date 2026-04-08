"""BC training Run C: large net [256,256], lr=1e-4, batch=256"""
import sys; sys.path.insert(0, ".")
from nt_rl.bc.config import BCConfig
from nt_rl.bc.train import train

cfg = BCConfig()
cfg.net_arch = [256, 256]
cfg.learning_rate = 1e-4
cfg.batch_size = 256
cfg.activation_fn = "tanh"
cfg.n_epochs = 10
cfg.use_wandb = False
cfg.checkpoint_dir = "checkpoints/bc_run_c"
cfg.demonstrations_dir = "demonstrations"

train(cfg)
