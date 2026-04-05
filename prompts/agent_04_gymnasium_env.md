# Agent 04 — Python Gymnasium Environment

## Role
You are a Python ML engineer implementing a Gymnasium-compatible reinforcement learning environment that wraps the Nuclear Throne GML socket bridge produced by Agents 02 and 03.

## Context
The game runs in Docker containers under Xvfb with a TCP socket server on port 7777. Each step, the game sends a JSON state message and awaits a JSON action message. Your environment must conform to the Gymnasium API so it integrates directly with Stable Baselines3's `SubprocVecEnv` for parallel training.

Read these documents before writing any code:
- `GML_BRIDGE_SUMMARY.md` — the exact JSON schemas for state and action messages
- `INFRA_README.md` — port mapping for N parallel containers

## Project Structure to Create
```
nt_rl/
├── __init__.py
├── env.py                  # Core NuclearThroneEnv Gymnasium class
├── vec_env.py              # Multi-instance SubprocVecEnv factory
├── config.py               # All hyperparameters and env settings
├── rewards.py              # Reward computation (mirrors GML weights)
├── obs_utils.py            # Observation normalization helpers
└── tests/
    ├── test_env.py         # Tests against mock socket server
    ├── mock_server.py      # Fake GML socket server for testing
    └── test_vec_env.py     # Parallel env tests
```

## Task 1 — Configuration Module (`config.py`)
All tunable values live here. No magic numbers anywhere else.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class EnvConfig:
    # Socket
    host: str = "localhost"
    base_port: int = 7777          # Port of first instance; +1 per additional
    socket_timeout: float = 10.0   # Seconds to wait for game connection
    step_timeout: float = 2.0      # Seconds to wait for state message per step

    # Observation space
    max_enemies: int = 32          # Must match GML cap in scr_agent_build_state
    max_pickups: int = 16
    obs_normalize: bool = True

    # Action space
    n_move_dirs: int = 9           # 8 directions + no-move (index 8)
    n_aim_angles: int = 24         # Quantized aim: 360/24 = 15 degree bins
    # shoot and dodge are binary — handled as MultiDiscrete components

    # Reward weights (must mirror GML scr_agent_compute_reward constants)
    reward_kill: float = 5.0
    reward_level_complete: float = 10.0
    reward_health_pickup: float = 2.0
    reward_damage_taken: float = -1.0
    reward_death: float = -15.0
    reward_survival_per_step: float = 0.01

    # Episode
    max_steps: int = 100_000       # Hard cap per episode regardless of survival

    # Parallelism
    n_envs: int = 4                # Number of parallel Docker containers
```

## Task 2 — Core Environment (`env.py`)
Implement `NuclearThroneEnv(gymnasium.Env)`.

### Observation Space
Construct a flat `Box` observation space. The observation vector must encode:

**Player features (fixed-length):**
- player_x, player_y (normalized to [0, 1] using room dimensions ~1000x750)
- player_hp_ratio (hp / hp_max)
- player_vel_x, player_vel_y (normalized by max speed, approx ±10)
- weapon_id (one-hot encoded across ~30 weapon types, or integer / 30)
- ammo_ratio (ammo / max_ammo, clipped to [0, 1])
- level (normalized by max level, ~20)
- is_dodging (0.0 or 1.0)

**Enemy features (max_enemies × 4, zero-padded):**
For each enemy slot: [norm_x, norm_y, hp_ratio, type_id_norm]
Sort enemies by distance to player before packing — closest enemies in first slots.

**Pickup features (max_pickups × 3, zero-padded):**
For each pickup slot: [norm_x, norm_y, type_id_norm]

Total observation dimension: 9 + (max_enemies × 4) + (max_pickups × 3)
With defaults: 9 + 128 + 48 = **185 floats**

Use `gymnasium.spaces.Box(low=0.0, high=1.0, shape=(185,), dtype=np.float32)`.
Note: velocity and some values can exceed [0,1] — clip aggressively and document it.

### Action Space
Use `gymnasium.spaces.MultiDiscrete`:
```python
# [move_dir (0-8), aim_bin (0-23), shoot (0-1), dodge (0-1)]
gymnasium.spaces.MultiDiscrete([9, 24, 2, 2])
```

Implement a method `_decode_action(action) -> dict` that converts this to the GML JSON format:
- `move_dir`: 0–7 maps to directions, 8 maps to -1 (no move)
- `aim_angle`: bin × 15.0 degrees
- `shoot`: bool
- `dodge`: bool

### Core Methods

**`__init__(self, port, config)`:**
- Store config
- Initialize socket to None (don't connect here)
- Define observation and action spaces

**`reset(self, seed=None, options=None) -> (obs, info)`:**
- If socket is None or disconnected: call `_connect()` with retry logic
- Send `{"command": "reset"}` to the game
- Wait for the first state message (up to `socket_timeout` seconds)
- Return `(_parse_obs(state), {"level": state["level"]})`

**`step(self, action) -> (obs, reward, terminated, truncated, info)`:**
- Encode action to JSON via `_decode_action()`
- Send action JSON to socket
- Wait for state JSON response (up to `step_timeout`)
- Parse state into obs vector
- Extract reward from `state["reward"]` (computed by GML)
- `terminated = state["done"]`
- `truncated = self._step_count >= config.max_steps`
- Increment `self._step_count`
- Return tuple

**`close(self)`:**
- Send `{"command": "quit"}` if socket is open
- Close socket gracefully

**`_connect(self, max_retries=10, retry_delay=3.0)`:**
- Attempt `socket.connect((config.host, self.port))`
- Retry up to max_retries times with retry_delay seconds between attempts
- Raise `RuntimeError` with clear message if all retries fail
- Log each retry attempt to stderr with attempt number

**`_recv_json(self) -> dict`:**
- Read from socket until newline delimiter `\n`
- Handle partial reads correctly using a receive buffer
- Raise `TimeoutError` if no data received within `step_timeout`
- Raise `json.JSONDecodeError` on malformed messages (log raw bytes for debugging)

**`_parse_obs(self, state: dict) -> np.ndarray`:**
- Implement the full observation encoding described above
- All values must be clipped to reasonable ranges before inclusion
- Return `np.float32` array

### Socket Protocol Assumption
The GML bridge sends one JSON message per step terminated by `\n`. Verify this matches `GML_BRIDGE_SUMMARY.md`. If the delimiter differs, adjust `_recv_json` accordingly and document the deviation.

### Error Resilience
The environment must survive:
- Game process crash (socket disconnect mid-episode): catch `ConnectionResetError`, call `_connect()` to wait for game restart (Agent 03's entrypoint.sh restarts it), then call `reset()`
- Malformed JSON from game: log warning, return zero observation, `reward=0`, `terminated=True`
- Step timeout: log warning, return zero observation, `terminated=True`

Never let an exception propagate out of `step()` or `reset()` — SB3's VecEnv cannot handle uncaught env exceptions gracefully.

## Task 3 — Parallel Environment Factory (`vec_env.py`)
```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from nt_rl.env import NuclearThroneEnv
from nt_rl.config import EnvConfig

def make_env(port: int, config: EnvConfig):
    def _init():
        env = NuclearThroneEnv(port=port, config=config)
        return env
    return _init

def make_vec_env(config: EnvConfig) -> SubprocVecEnv:
    env_fns = [
        make_env(port=config.base_port + i, config=config)
        for i in range(config.n_envs)
    ]
    return SubprocVecEnv(env_fns)
```

## Task 4 — Mock Socket Server (`tests/mock_server.py`)
This is the most important testing tool. It simulates the GML game without requiring Docker or a running game.

Implement a `MockNuclearThroneServer` that:
- Listens on a configurable port
- On receiving `{"command": "reset"}`: responds with a valid initial state JSON
- On receiving an action: responds with a plausible next state (move player slightly, maybe spawn/remove an enemy)
- On receiving `{"command": "quit"}`: closes connection
- Tracks step count and terminates the episode after N steps
- Runs in a background thread so tests can interact with it synchronously

The mock state must use the exact same JSON schema as `GML_BRIDGE_SUMMARY.md`.

## Task 5 — Test Suite (`tests/test_env.py`)
Write tests using `pytest`. All tests must pass against the mock server without any Docker dependency.

Required tests:
```python
def test_reset_returns_valid_obs()
def test_step_returns_correct_shape()
def test_action_space_sample_is_valid()
def test_observation_space_bounds()    # obs stays within Box bounds
def test_episode_terminates_on_done()
def test_env_survives_socket_disconnect()  # Mock server closes mid-episode
def test_env_survives_malformed_json()     # Mock server sends garbage
def test_vec_env_creates_n_instances()
def test_gymnasium_api_compliance()    # Use gymnasium.utils.env_checker
```

Run the full test suite and report results before declaring completion:
```bash
pip install pytest gymnasium stable-baselines3 numpy
pytest tests/ -v
```

All tests must pass. Fix any failures before completing.

## Task 6 — Dependency File
Create `requirements.txt`:
```
gymnasium>=0.29.0
stable-baselines3>=2.3.0
numpy>=1.26.0
torch>=2.2.0
imitation>=1.0.0
wandb>=0.16.0
pytest>=8.0.0
```

## Completion Criteria
You are done when:
- All 6 files in `nt_rl/` exist and are fully implemented (no stubs, no TODOs)
- `mock_server.py` accurately reflects the state/action schema from `GML_BRIDGE_SUMMARY.md`
- All 9 tests pass: `pytest tests/ -v` exits 0
- `gymnasium.utils.env_checker.check_env(NuclearThroneEnv(port=PORT, config=EnvConfig()))` passes against the mock server
- `requirements.txt` is complete

## Handoff Note for Agent 05 (Training)
Include a file `ENVIRONMENT_HANDOFF.md` containing:
- How to instantiate a single env vs vectorized env (code snippet)
- The observation vector layout (which indices mean what)
- The action space encoding (what each MultiDiscrete component represents)
- Confirmed test results
- Any known limitations (e.g., "obs is clipped at X, which may saturate during boss fights")
- Recommended SB3 hyperparameter starting points for PPO based on the obs/action dimensions

## Do Not
- Import the real game or Docker anywhere in the test suite — all tests use the mock server
- Use `gym` (old API) — use `gymnasium` (new API) throughout
- Implement your own reward computation — reward comes from the GML bridge via `state["reward"]`
- Assume the socket sends complete messages atomically — always use a buffered reader
