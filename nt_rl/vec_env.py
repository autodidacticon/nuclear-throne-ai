"""Vectorized environment factory for parallel Nuclear Throne training."""

from stable_baselines3.common.vec_env import SubprocVecEnv

from nt_rl.config import EnvConfig
from nt_rl.env import NuclearThroneEnv


def make_env(port: int, config: EnvConfig):
    """Create a factory function for a single NuclearThroneEnv instance."""
    def _init():
        return NuclearThroneEnv(port=port, config=config)
    return _init


def make_vec_env(config: EnvConfig | None = None) -> SubprocVecEnv:
    """Create a SubprocVecEnv with N parallel Nuclear Throne environments.

    Each environment connects to a different Docker container via its
    unique port (base_port, base_port+1, ..., base_port+n_envs-1).
    """
    if config is None:
        config = EnvConfig()

    env_fns = [
        make_env(port=config.base_port + i, config=config)
        for i in range(config.n_envs)
    ]
    return SubprocVecEnv(env_fns)
