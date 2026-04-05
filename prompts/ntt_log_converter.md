# NTT Log Converter — Transform NTT Recording Logs to Training Data

## Role
You are a Python engineer building a converter that transforms NTT recording mod logs into the `.npz` demonstration format that the BC training pipeline expects.

## Context
A Nuclear Throne Together mod (`ntt_mods/nt_recorder.mod.gml`, see `prompts/ntt_recording_mod.md`) records human gameplay from the official Nuclear Throne game as `.jsonl` files — one JSON object per frame. Your job is to convert these logs into the same `.npz` format that `nt_rl/bc/recorder.py` produces from the rebuild's socket bridge, so the training pipeline (`nt_rl/bc/train.py`) can consume both data sources interchangeably.

Read these files before writing any code:
- `nt_rl/bc/recorder.py` — the rebuild recorder, produces the target `.npz` format
- `nt_rl/obs_utils.py` — `encode_observation()` — the canonical observation encoder
- `nt_rl/bc/dataset.py` — `DemonstrationDataset` — what loads the `.npz` files
- `nt_rl/config.py` — `EnvConfig` — normalization constants and space definitions
- `ENVIRONMENT_HANDOFF.md` — observation vector layout
- `prompts/ntt_recording_mod.md` — the NTT mod's output format

## Input Format

NTT log files: `ntt_demo_TIMESTAMP_NNNN.jsonl`

Each line is a JSON object:
```json
{
  "frame": 42,
  "player": {
    "x": 5040.0, "y": 4980.0,
    "my_health": 6, "maxhealth": 8,
    "hspeed": 2.5, "vspeed": -1.0,
    "gunangle": 135.0,
    "wep": 3, "bwep": 12,
    "ammo": [0, 15, 8, 0, 0, 0],
    "reload": 0, "can_shoot": true,
    "roll": false, "race": 1, "nexthurt": 0
  },
  "enemies": [
    {"x": 5200, "y": 4900, "my_health": 5, "maxhealth": 10, "type_id": 42}
  ],
  "game": {
    "area": 1, "subarea": 1, "level": 2,
    "loops": 0, "kills": 7, "hard": 0
  },
  "human_action": {
    "move_dir": 90.0, "moving": true,
    "aim_dir": 135.0, "fire": true,
    "spec": false, "swap": false, "pick": false
  },
  "reward_signals": {
    "kills_this_frame": 1,
    "damage_this_frame": 0,
    "healed_this_frame": false,
    "level_changed": false
  }
}
```

## Output Format

Per-episode `.npz` files in `demonstrations/`, identical to what `DemonstrationRecorder` produces:
- `obs`: float32 array, shape `(T, 112)` — encoded observation vectors
- `actions`: int32 array, shape `(T, 4)` — `MultiDiscrete([9, 24, 2, 2])`
- `rewards`: float32 array, shape `(T,)` — computed per-step rewards
- `dones`: bool array, shape `(T,)` — terminal flags

## Key Transformations

### 1. Variable Name Mapping

The NTT mod uses official game variable names. Map to the rebuild's names before calling `encode_observation()`:

```python
state["player"]["hp"] = state["player"].pop("my_health")
state["player"]["max_hp"] = state["player"].pop("maxhealth")
for enemy in state["enemies"]:
    enemy["hp"] = enemy.pop("my_health")
    enemy["max_hp"] = enemy.pop("maxhealth")
    enemy["hitid"] = enemy.pop("type_id")  # object_index used as hitid proxy
```

After this mapping, `encode_observation(state, config)` works directly.

### 2. Action Discretization

Use `discretize_action()` from `nt_rl/bc/recorder.py` — it converts the continuous `human_action` dict to `MultiDiscrete([9, 24, 2, 2])`.

### 3. Reward Computation

The NTT mod records raw reward signals. Apply weights from `EnvConfig`:

```python
reward = config.reward_survival_per_step  # 0.01 base
reward += signals["kills_this_frame"] * config.reward_kill
reward += signals["damage_this_frame"] * config.reward_damage_taken  # negative weight
if signals["healed_this_frame"]:
    # Only apply heal reward if HP was below 50%
    if player_hp_ratio < 0.5:
        reward += config.reward_health_pickup
if signals["level_changed"]:
    reward += config.reward_level_complete
# Death: player hp <= 0 on the final frame
if is_terminal:
    reward += config.reward_death
```

### 4. Episode Detection

Each `.jsonl` file from the NTT mod is one episode. However, a single file may contain multiple episodes if the mod flushed mid-session. Detect episode boundaries by:
- `frame` counter resetting to a lower value
- `player.my_health` jumping from 0 back to `maxhealth`

Split multi-episode files into separate `.npz` outputs.

## File to Create

```
nt_rl/bc/ntt_converter.py
```

## Implementation

### Class: `NTTLogConverter`

```python
class NTTLogConverter:
    def __init__(self, config: EnvConfig | None = None):
        ...

    def convert_file(self, jsonl_path: str, output_dir: str = "demonstrations") -> list[str]:
        """Convert one .jsonl log file to one or more .npz episode files.
        Returns list of created file paths."""
        ...

    def convert_directory(self, input_dir: str, output_dir: str = "demonstrations") -> int:
        """Convert all .jsonl files in a directory. Returns total episodes converted."""
        ...
```

### CLI Entry Point

```
python -m nt_rl.bc.ntt_converter --input /path/to/ntt/logs --output demonstrations
```

Arguments:
- `--input`: directory containing `.jsonl` files (or a single file)
- `--output`: output directory for `.npz` files (default: `demonstrations`)
- `--config-file`: optional EnvConfig override (JSON)

### Validation

After conversion, run the same validation as `DemonstrationDataset`:
- Observation shape matches `EnvConfig().obs_dim`
- Actions are in valid ranges for `MultiDiscrete([9, 24, 2, 2])`
- Print statistics: total transitions, episodes, mean length, action distribution

### Tests

Add tests to `nt_rl/tests/test_bc.py`:

1. **`test_ntt_converter_single_episode`**: Create a synthetic `.jsonl` file with NTT variable names, convert it, verify the `.npz` output matches expected shapes and values.

2. **`test_ntt_converter_variable_mapping`**: Verify `my_health` → `hp`, `maxhealth` → `max_hp`, `type_id` → `hitid` mapping produces identical observations to a rebuild-format state.

3. **`test_ntt_converter_multi_episode`**: Create a `.jsonl` file with two episodes (frame counter resets), verify two `.npz` files are produced.

4. **`test_ntt_converter_reward_computation`**: Verify kill reward, damage penalty, heal reward, level completion reward, and death penalty are computed correctly from raw signals.

## Completion Criteria
- `nt_rl/bc/ntt_converter.py` exists with `NTTLogConverter` class and CLI
- Converted `.npz` files are loadable by `DemonstrationDataset` with no validation errors
- Variable name mapping produces observations identical to rebuild-sourced data (same state → same 112-float vector)
- Reward computation matches `EnvConfig` weights
- All tests pass
- Handles edge cases: empty files, truncated episodes, missing fields

## Do Not
- Duplicate observation encoding logic — call `encode_observation()` from `obs_utils.py`
- Duplicate action discretization — call `discretize_action()` from `recorder.py`
- Hardcode reward weights — read from `EnvConfig`
- Assume one episode per file — handle multi-episode files
