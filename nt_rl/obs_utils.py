"""Observation encoding utilities for Nuclear Throne RL environment.

Observation vector layout (with default config, 239 floats total):

Player features (indices 0-18):
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
  [12] wall_dist_e           — distance to east wall, already normalized [0, 1]
  [13] wall_dist_n           — distance to north wall, [0, 1]
  [14] wall_dist_w           — distance to west wall, [0, 1]
  [15] wall_dist_s           — distance to south wall, [0, 1]
  [16] enemies_remaining_norm — game.enemies_remaining / max_enemies_on_level, clipped [0, 1]
  [17] portal_dir_norm       — game.portal_dir, already [0, 1]
  [18] portal_dist_norm      — game.portal_dist, already [0, 1]

Enemy features (indices 19 to 118, zero-padded to 20 enemies):
  Per enemy (5 floats):
    [0] enemy_x_norm         — x / room_width
    [1] enemy_y_norm         — y / room_height
    [2] enemy_hp_ratio       — hp / max_hp (enemy's own max_hp)
    [3] enemy_max_hp_norm    — max_hp / 100, clipped [0, 1]
    [4] enemy_hitid_norm     — hitid / max_hitid

Projectile features (indices 119 to 238, zero-padded to 20 projectiles):
  Per projectile (6 floats):
    [0] proj_x_norm          — x / room_width, clipped [0, 1]
    [1] proj_y_norm          — y / room_height, clipped [0, 1]
    [2] proj_hspeed_norm     — hspeed / max_projectile_speed, clipped [-1, 1]
    [3] proj_vspeed_norm     — vspeed / max_projectile_speed, clipped [-1, 1]
    [4] proj_damage_norm     — damage / max_projectile_damage, clipped [0, 1]
    [5] proj_lifetime_norm   — lifetime / max_projectile_lifetime, clipped [0, 1]

  Empty slots are zero-filled (absence == no threat).
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

    # Wall distance raycasts (already normalized [0, 1] by GML; default 0.5 for
    # BC data that doesn't include wall distances — avoids teaching "walls everywhere")
    obs[12] = float(np.clip(player.get("wall_dist_e", 0.5), 0.0, 1.0))
    obs[13] = float(np.clip(player.get("wall_dist_n", 0.5), 0.0, 1.0))
    obs[14] = float(np.clip(player.get("wall_dist_w", 0.5), 0.0, 1.0))
    obs[15] = float(np.clip(player.get("wall_dist_s", 0.5), 0.0, 1.0))

    # Strategic features (from game struct; defaults handle BC/NTT data that
    # doesn't include these fields)
    obs[16] = float(np.clip(
        game.get("enemies_remaining", 15) / config.max_enemies_on_level, 0.0, 1.0))
    obs[17] = float(np.clip(game.get("portal_dir", 0.5), 0.0, 1.0))
    obs[18] = float(np.clip(game.get("portal_dist", 1.0), 0.0, 1.0))

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

    # Projectile features (sorted by distance in GML, nearest-first)
    # Empty slots stay at 0.0 (absence == no threat).
    projectiles = state.get("projectiles", [])
    proj_offset = (
        config.player_features + config.max_enemies * config.enemy_features
    )
    n_projectiles = min(len(projectiles), config.max_projectiles)

    for i in range(n_projectiles):
        p = projectiles[i]
        base = proj_offset + i * config.projectile_features
        # Position: clip-normalize by room dimensions. If GML sends already-
        # normalized values in [0,1], dividing again would underflow, so we
        # only normalize when value > 1 (heuristic for "raw pixel coords").
        obs[base + 0] = _maybe_norm(p.get("x", 0.0), config.room_width)
        obs[base + 1] = _maybe_norm(p.get("y", 0.0), config.room_height)
        # Speeds: clip-normalize to [-1, 1]
        obs[base + 2] = _maybe_signed_norm(
            p.get("hspeed", 0.0), config.max_projectile_speed)
        obs[base + 3] = _maybe_signed_norm(
            p.get("vspeed", 0.0), config.max_projectile_speed)
        # Damage: clip-normalize to [0, 1]
        obs[base + 4] = _maybe_norm(
            p.get("damage", 0.0), config.max_projectile_damage)
        # Lifetime: clip-normalize to [0, 1]
        obs[base + 5] = _maybe_norm(
            p.get("lifetime", 0.0), config.max_projectile_lifetime)

    return obs


def _clip_norm(value: float, max_val: float) -> float:
    """Normalize value to [0, 1] and clip."""
    if max_val <= 0:
        return 0.0
    return float(np.clip(value / max_val, 0.0, 1.0))


def _maybe_norm(value: float, max_val: float) -> float:
    """Normalize to [0, 1], handling both raw and pre-normalized inputs.

    If value already lies in [0, 1], it's assumed to be pre-normalized by GML.
    Otherwise it's divided by max_val. Always clipped to [0, 1].
    """
    if max_val <= 0:
        return 0.0
    v = float(value)
    if 0.0 <= v <= 1.0:
        return v
    return float(np.clip(v / max_val, 0.0, 1.0))


def _maybe_signed_norm(value: float, max_val: float) -> float:
    """Normalize to [-1, 1], handling both raw and pre-normalized inputs.

    If value already lies in [-1, 1], it's assumed to be pre-normalized.
    Otherwise it's divided by max_val. Always clipped to [-1, 1].
    """
    if max_val <= 0:
        return 0.0
    v = float(value)
    if -1.0 <= v <= 1.0:
        return v
    return float(np.clip(v / max_val, -1.0, 1.0))
