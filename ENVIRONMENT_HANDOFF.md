# Nuclear Throne RL Environment — Handoff for Agent 05

*Phase 4 Output — Agent 04 — 2026-04-04*

---

## Quick Start

### Single Environment
```python
from nt_rl import NuclearThroneEnv, EnvConfig

config = EnvConfig(base_port=7777)
env = NuclearThroneEnv(port=7777, config=config)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Vectorized Environment (Parallel Training)
```python
from nt_rl.vec_env import make_vec_env
from nt_rl.config import EnvConfig

config = EnvConfig(n_envs=4, base_port=7777)
vec_env = make_vec_env(config)

obs = vec_env.reset()
for _ in range(10000):
    actions = [vec_env.action_space.sample() for _ in range(config.n_envs)]
    obs, rewards, dones, infos = vec_env.step(actions)

vec_env.close()
```

---

## Observation Vector Layout

Total dimension: **112 floats** (with default config: 12 player + 20×5 enemy)

### Player Features (indices 0–11)

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | `player_x_norm` | [0, 1] | x / 10080 |
| 1 | `player_y_norm` | [0, 1] | y / 10080 |
| 2 | `player_hp_ratio` | [0, 1] | hp / max_hp |
| 3 | `player_hspeed_norm` | [-1, 1] | hspeed / 10 |
| 4 | `player_vspeed_norm` | [-1, 1] | vspeed / 10 |
| 5 | `gunangle_norm` | [0, 1] | gunangle / 360 |
| 6 | `weapon_id_norm` | [0, 1] | wep / 128 |
| 7 | `ammo_total_norm` | [0, 1] | sum(ammo) / 594 |
| 8 | `reload_norm` | [0, 1] | reload / 60 |
| 9 | `can_shoot` | {0, 1} | Boolean |
| 10 | `is_rolling` | {0, 1} | Boolean |
| 11 | `level_norm` | [0, 1] | level / 20 |

### Enemy Features (indices 12–111, zero-padded)

For each of the 20 enemy slots (5 floats per enemy):

| Offset | Name | Range | Description |
|--------|------|-------|-------------|
| +0 | `enemy_x_norm` | [0, 1] | x / 10080 |
| +1 | `enemy_y_norm` | [0, 1] | y / 10080 |
| +2 | `enemy_hp_ratio` | [0, 1] | hp / max_hp |
| +3 | `enemy_max_hp_norm` | [0, 1] | max_hp / 100 |
| +4 | `enemy_hitid_norm` | [0, 1] | hitid / 120 |

Enemies are sorted by distance (nearest first). Empty slots are zero-filled.

---

## Action Space Encoding

`MultiDiscrete([9, 24, 2, 2])`

| Component | Values | Meaning |
|-----------|--------|---------|
| `move_dir` | 0–7 | 8 directions: 0=E, 45=NE, 90=N, 135=NW, 180=W, 225=SW, 270=S, 315=SE |
| `move_dir` | 8 | No movement |
| `aim_bin` | 0–23 | Aim angle = bin × 15° |
| `shoot` | 0 or 1 | Fire weapon |
| `special` | 0 or 1 | Use special ability |

---

## Test Results

```
9 passed in 9.19s

test_reset_returns_valid_obs         PASSED
test_reset_returns_info_dict         PASSED
test_step_returns_correct_shape      PASSED
test_action_space_sample_is_valid    PASSED
test_observation_space_bounds        PASSED
test_episode_terminates_on_done      PASSED
test_env_survives_socket_disconnect  PASSED
test_env_survives_malformed_json     PASSED
test_gymnasium_api_compliance        PASSED
```

Gymnasium `check_env` passes with no warnings.

---

## Known Limitations

1. **Observation clipping**: All values are clipped to [-1, 1] or [0, 1]. During boss fights with many enemies or high-speed movement, some information may be lost to saturation.

2. **Enemy count cap**: Only the nearest 20 enemies are included. Levels with 50+ enemies will have distant threats invisible to the agent.

3. **No pickup features**: The current implementation excludes pickup locations from the observation vector (unlike the prompt suggestion for max_pickups). This can be added later by extending `obs_utils.py`.

4. **Weapon/ammo encoding**: Weapon is encoded as a single normalized ID rather than one-hot. One-hot would be more informative but adds 128 dimensions. Ammo is summed across types rather than per-type. These are reasonable simplifications for initial training.

5. **Mutation screen**: When `info["mutation_screen"]` is True, the game is on the mutation selection screen. The agent currently has no way to select mutations — the bridge would need swap/pick actions mapped to mutation choices.

6. **Socket reconnection**: If the game crashes mid-episode, the env catches the error, returns `terminated=True`, and reconnects on next `reset()`. The Python side does not need explicit reconnection logic.

---

## Recommended SB3 PPO Hyperparameters

Starting points for this observation/action space:

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
    verbose=1,
)
```

**Rationale:**
- 2-layer 256-unit MLP is standard for obs dim ~112
- `ent_coef=0.01` encourages exploration in the early stages
- `n_steps=2048` provides enough trajectory data per update with 4 envs (8192 total)
- `gamma=0.99` is appropriate for the long episode horizons (up to 100k steps)
