"""Observation encoding utilities for Nuclear Throne RL environment.

Observation vector layout (with default config, 112 floats total):

Player features (indices 0-11):
  [0]  player_x_norm         — x / room_width, clipped [0, 1]
  [1]  player_y_norm         — y / room_height, clipped [0, 1]
  [2]  player_hp_ratio       — hp / max_hp, clipped [0, 1]
  [3]  player_hspeed_norm    — hspeed / max_speed, clipped [-1, 1]
  [4]  player_vspeed_norm    — vspeed / max_speed, clipped [-1, 1]
  [5]  gunangle_norm         — gunangle / 360, [0, 1]
  [6]  weapon_id_norm        — wep / max_weapon_id, [0, 1]
  [7]  ammo_total_norm       — sum(ammo) / (max_ammo * 6), [0, 1]
  [8]  reload_norm           — reload / 60 (max reload ~60 frames), clipped [0, 1]
  [9]  can_shoot             — 0.0 or 1.0
  [10] is_rolling            — 0.0 or 1.0
  [11] level_norm            — game.level / max_level, [0, 1]

Enemy features (indices 12 to 12 + max_enemies * 5, zero-padded):
  Per enemy (5 floats):
    [0] enemy_x_norm         — x / room_width
    [1] enemy_y_norm         — y / room_height
    [2] enemy_hp_ratio       — hp / max_hp (enemy's own max_hp)
    [3] enemy_max_hp_norm    — max_hp / 100, clipped [0, 1]
    [4] enemy_hitid_norm     — hitid / max_hitid
"""

import numpy as np

from nt_rl.config import EnvConfig


def encode_observation(state: dict, config: EnvConfig) -> np.ndarray:
    """Convert a state dict from the GML bridge into a flat float32 observation vector."""
    obs = np.zeros(config.obs_dim, dtype=np.float32)

    player = state.get("player", {})
    game = state.get("game", {})
    enemies = state.get("enemies", [])

    # Player features
    obs[0] = _clip_norm(player.get("x", 0), config.room_width)
    obs[1] = _clip_norm(player.get("y", 0), config.room_height)

    max_hp = max(player.get("max_hp", 1), 1)
    obs[2] = np.clip(player.get("hp", 0) / max_hp, 0.0, 1.0)

    obs[3] = np.clip(player.get("hspeed", 0) / config.max_speed, -1.0, 1.0)
    obs[4] = np.clip(player.get("vspeed", 0) / config.max_speed, -1.0, 1.0)

    obs[5] = player.get("gunangle", 0) / 360.0

    obs[6] = _clip_norm(player.get("wep", 0), config.max_weapon_id)

    ammo = player.get("ammo", [0, 0, 0, 0, 0, 0])
    if isinstance(ammo, list):
        obs[7] = np.clip(sum(ammo) / (config.max_ammo * 6), 0.0, 1.0)
    else:
        obs[7] = 0.0

    obs[8] = np.clip(player.get("reload", 0) / 60.0, 0.0, 1.0)
    obs[9] = 1.0 if player.get("can_shoot", False) else 0.0
    obs[10] = 1.0 if player.get("roll", False) else 0.0
    obs[11] = _clip_norm(game.get("level", 0), config.max_level)

    # Enemy features (sorted by distance in GML, already nearest-first)
    offset = config.player_features
    n_enemies = min(len(enemies), config.max_enemies)

    for i in range(n_enemies):
        e = enemies[i]
        base = offset + i * config.enemy_features
        obs[base + 0] = _clip_norm(e.get("x", 0), config.room_width)
        obs[base + 1] = _clip_norm(e.get("y", 0), config.room_height)
        e_max_hp = max(e.get("max_hp", 1), 1)
        obs[base + 2] = np.clip(e.get("hp", 0) / e_max_hp, 0.0, 1.0)
        obs[base + 3] = np.clip(e.get("max_hp", 0) / 100.0, 0.0, 1.0)
        obs[base + 4] = _clip_norm(e.get("hitid", 0), config.max_hitid)

    return obs


def _clip_norm(value: float, max_val: float) -> float:
    """Normalize value to [0, 1] and clip."""
    if max_val <= 0:
        return 0.0
    return float(np.clip(value / max_val, 0.0, 1.0))
