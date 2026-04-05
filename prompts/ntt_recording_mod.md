# NTT Recording Mod — Human Gameplay Logger for Official Nuclear Throne

## Role
You are a GameMaker modder writing a Nuclear Throne Together (NTT) mod that records human gameplay demonstrations from the official Nuclear Throne game. The recorded data will be converted to training data for a reinforcement learning agent.

## Context
We have an RL training pipeline that expects demonstrations as per-episode `.npz` files with arrays `obs`, `actions`, `rewards`, `dones`. A separate Python converter (see `prompts/ntt_log_converter.md`) will transform your mod's output into that format. Your job is to capture all the game state and human input needed for that conversion, each frame, and write it to disk in a parseable format.

Read these documents before writing any code:
- `ENVIRONMENT_HANDOFF.md` — observation vector layout (what state variables the RL agent sees)
- `GML_BRIDGE_SUMMARY.md` — canonical state and action JSON schemas
- `nt-recreated-public/scripts/scr_agent_build_state/scr_agent_build_state.gml` — the rebuild's state serialization (your mod must capture equivalent data)
- `nt-recreated-public/scripts/scr_agent_read_human_input/scr_agent_read_human_input.gml` — the rebuild's input reading (your mod must capture equivalent data)

## NTT Scripting Reference

NTT mods use a GML variant with `#define` directives for hooks. Your mod is a `.mod.gml` file.

### Available Hooks
- `#define init` — called once when mod loads
- `#define step` — called every frame
- `#define draw` — called every draw frame
- `#define cleanup` — called when mod unloads / game exits

### Reading Game State
NTT gives full read access to instance variables. Key differences from the rebuild:

| Rebuild variable | NTT equivalent | Notes |
|-----------------|----------------|-------|
| `Player.hp` | `Player.my_health` | |
| `Player.max_hp` | `Player.maxhealth` | |
| `Player.x`, `Player.y` | Same | |
| `Player.hspeed`, `Player.vspeed` | Same | |
| `Player.gunangle` | Same | Aim direction in degrees |
| `Player.wep`, `Player.bwep` | Same | Weapon IDs |
| `Player.ammo` | Same | Array of 6 |
| `Player.reload` | Same | Reload counter |
| `Player.can_shoot` | Same | Boolean |
| `Player.roll` | Same | Roll/dodge active |
| `Player.race` | Same | Character ID |
| `Player.nexthurt` | Same | Iframe counter |
| `GameCont.area` | Same | |
| `GameCont.subarea` | Same | |
| `GameCont.level` | Same | |
| `GameCont.loops` | Same | |
| `GameCont.kills` | Same | |
| `GameCont.hard` | Same | |
| `enemy` | Same | Base enemy type, iterate with `with(enemy)` |
| `enemy.hp` | `enemy.my_health` | |
| `enemy.max_hp` | `enemy.maxhealth` | |
| `enemy.hitid` | May not exist | Use `object_index` as fallback type ID |
| `LevCont` | Same | Mutation screen |

### Reading Human Input
```gml
button_check(0, "north")    // held: up
button_check(0, "south")    // held: down
button_check(0, "east")     // held: right
button_check(0, "west")     // held: left
button_check(0, "fire")     // held: fire
button_check(0, "spec")     // held: special ability
button_pressed(0, "swap")   // just pressed: weapon swap
button_pressed(0, "pick")   // just pressed: pickup
```

Movement direction must be derived from cardinal button states:
```
dx = button_check(0, "east") - button_check(0, "west")
dy = button_check(0, "south") - button_check(0, "north")
moving = (dx != 0 || dy != 0)
move_dir = point_direction(0, 0, dx, -dy)  // GML convention: 0=east, 90=north
```

Aim direction: use `Player.gunangle` (already in degrees, 0-360).

### File I/O
```gml
string_save(string, path)   // write string to file (overwrites)
string_load(path)            // read file as string
file_exists(path)            // check existence
file_delete(path)            // delete file
```

### Limitations
- No `network_*` functions (no sockets)
- No `keyboard_check()` — use `button_check()` / `button_pressed()` instead
- No `game_set_speed()` — game runs at its normal frame rate
- No custom object creation — use `CustomObject` if needed
- Interpreted bytecode — keep per-frame work minimal for performance

## Output Format

Write one JSON-lines file per episode to the game's save directory:
```
ntt_demo_TIMESTAMP_NNNN.jsonl
```

Each line is a JSON object representing one frame:
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

### Design Decisions

**Enemy `type_id`**: The rebuild uses `hitid` for enemy type identification. The official game may or may not expose `hitid` on all enemies. Use `object_index` as the type identifier — it's always available in NTT. The Python converter will handle the mapping.

**Reward signals**: Instead of computing reward weights in the mod (which would duplicate logic), record the raw signals: kills delta, damage delta, heal events, level transitions. The Python converter applies the reward weights from `EnvConfig`.

**Enemy sorting**: Sort enemies by distance to player (nearest first), cap at 20. This matches the rebuild's `scr_agent_build_state` behavior.

**Flush strategy**: Accumulate frames in a string variable. Flush to disk:
- On player death (episode end)
- Every 300 frames (~10 seconds at 30 FPS) as a safety measure against crashes
- On mod cleanup

**Episode boundaries**: A new episode starts when:
- The player spawns after a death
- The game restarts from the menu
- Detect via `GameCont` being recreated or `Player.my_health` jumping from 0 to max

## File to Create

```
ntt_mods/nt_recorder.mod.gml
```

## Implementation Tasks

1. **`#define init`**: Initialize globals for frame accumulation, episode counter, previous-frame state tracking (for reward signal deltas)

2. **`#define step`**: Each frame:
   - Check `instance_exists(Player)` — skip if no player
   - Read player state (all variables listed above)
   - Iterate enemies with `with(enemy)`, collect position/health/type, sort by distance, cap at 20
   - Read game state from `GameCont`
   - Read human input via `button_check` / `button_pressed`
   - Compute reward signal deltas (kills - prev_kills, hp change, area/subarea change)
   - Serialize frame as JSON string, append to accumulator
   - If player is dead (`my_health <= 0`): flush episode to disk, reset accumulators
   - If frame count since last flush >= 300: flush partial data

3. **`#define cleanup`**: Flush any remaining accumulated data

4. **Visual feedback**: In `#define draw`, show a small red recording indicator dot in the corner so the player knows recording is active. Optionally show frame count and episode number.

## Completion Criteria
- `ntt_mods/nt_recorder.mod.gml` exists and follows NTT mod conventions
- Mod captures all state variables needed to reconstruct the 112-float observation vector
- Mod captures human input in the action JSON schema
- Mod captures raw reward signals for the converter to apply weights
- Episodes are saved as `.jsonl` files
- Per-frame overhead is minimal (no file I/O every frame except during flush)
- Recording indicator is visible during play

## Do Not
- Compute the 112-float observation vector in GML — that's the Python converter's job
- Apply reward weights — record raw signals, let the converter apply `EnvConfig` weights
- Use `network_*` functions — they don't exist in NTT
- Use `keyboard_check()` — use `button_check()` / `button_pressed()` instead
- Write to disk every frame — accumulate and flush periodically
