"""Configuration for Nuclear Throne RL environment. All tunable values live here."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    # Socket
    host: str = "localhost"
    base_port: int = 7777
    socket_timeout: float = 10.0
    step_timeout: float = 2.0

    # Observation space
    max_enemies: int = 20  # Matches GML global.agent_max_enemies
    enemy_features: int = 5  # x, y, hp_ratio, max_hp, hitid_norm
    player_features: int = 12  # See obs_utils.py for layout

    # Action space
    n_move_dirs: int = 9  # 8 directions + no-move (index 8)
    n_aim_angles: int = 24  # 360/24 = 15 degree bins

    # Reward weights (mirror GML scr_agent_config values — for reference only,
    # actual reward comes from the GML bridge)
    reward_kill: float = 5.0
    reward_level_complete: float = 10.0
    reward_health_pickup: float = 2.0
    reward_damage_taken: float = -1.0
    reward_death: float = -15.0
    reward_survival_per_step: float = 0.01

    # Episode
    max_steps: int = 100_000

    # Parallelism
    n_envs: int = 4

    # Normalization constants
    room_width: float = 10080.0  # Approximate max room dimension
    room_height: float = 10080.0
    max_speed: float = 10.0
    max_hp: float = 12.0
    max_weapon_id: float = 128.0
    max_ammo: float = 99.0
    max_level: float = 20.0
    max_hitid: float = 120.0

    @property
    def obs_dim(self) -> int:
        return self.player_features + (self.max_enemies * self.enemy_features)
