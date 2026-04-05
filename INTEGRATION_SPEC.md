# Nuclear Throne RL Integration Specification

*Generated: 2026-04-04 — Phase 1 (Agent 01)*
*Source: nt-recreated-public, branch `rewrite`*

---

## 1. Game Loop

### Controller Architecture

The game uses a multi-controller architecture within two rooms (`romInit` → `romGame`).

| Object | Role | Key File |
|--------|------|----------|
| **UberCont** | Primary per-frame game loop | `objects/UberCont/Step_0.gml` |
| **GameCont** | Game state manager (kills, area, timers) | `objects/GameCont/Step_0.gml` |
| **Vlambeer** | Room initialisation; creates GameCont, GenCont, etc. | `objects/Vlambeer/Create_0.gml` |
| **GenCont** | Procedural level generation | `objects/GenCont/Create_0.gml` |
| **LevCont** | Level-up / mutation selection screen | `objects/LevCont/Create_0.gml` |

### Frame Execution Order

1. **UberCont::Step_0** — input processing (`input_tick()`, `scrHandleInputsGeneral()`), pause logic, speed control, camera/audio updates, frame counter increment
2. **GameCont::Step_0** — timers, radiance management, level progression tracking
3. **UberCont::Step_1** — pause state transitions, instance activation/deactivation

### Speed Control

**File:** `objects/UberCont/Step_0.gml` (lines 147–165)

- Default: **30 FPS** via `game_set_speed(30, gamespeed_fps)`
- Debug uncap: triggered by `TestCont` instance + `global.__debug_test_framerate_uncapped` flag
- For RL: call `game_set_speed(500, gamespeed_fps)` (or higher) when agent mode is active

### Room Architecture

Only two rooms exist. All levels are generated procedurally within `romGame` via `room_restart()`.

| Room | Purpose | Instances |
|------|---------|-----------|
| `romInit` | Startup, disclaimer, save loading | `MakeGame` |
| `romGame` | All gameplay | `Vlambeer` (creates everything else) |

---

## 2. Player State Schema

**Object:** `Player` (inherits from `hitme`)
**Location:** `objects/Player/`

### Core RL Observation Variables

| Variable | Type | Meaning | Defined In |
|----------|------|---------|------------|
| `x` | real | X position | GML built-in |
| `y` | real | Y position | GML built-in |
| `hp` | real | Current hit points | `Create_0.gml:3` |
| `max_hp` | real | Maximum hit points | `Create_0.gml:4` |
| `hspeed` | real | Horizontal velocity | GML built-in |
| `vspeed` | real | Vertical velocity | GML built-in |
| `speed` | real | Total movement speed magnitude | GML built-in |
| `direction` | real | Movement direction (0–360) | GML built-in |
| `gunangle` | real | Aim direction (0–360) | `Create_0.gml:12` |
| `wep` | real | Primary weapon ID (macro) | `Create_0.gml:22` |
| `bwep` | real | Secondary weapon ID | `Create_0.gml:23` |
| `ammo` | array[real] | Ammo per type, indexed by `Ammo` enum | `scrAmmoInit.gml:27` |
| `reload` | real | Primary weapon reload counter | `Create_0.gml:20` |
| `breload` | real | Secondary weapon reload counter | `Create_0.gml:21` |
| `can_shoot` | bool | Primary weapon can fire | `Create_0.gml:99` |
| `bcan_shoot` | bool | Secondary weapon can fire | `Create_0.gml:100` |
| `nexthurt` | real | Frame number when next damage allowed (iframe check: `current_frame < nexthurt`) | `hitme/Create_0.gml:5` |
| `roll` | bool | Currently rolling/dodging | `Create_0.gml:62` |
| `race` | real | Character class (Race enum) | `Create_0.gml:6` |
| `index` | real | Player instance index (multiplayer) | `Create_0.gml:8` |
| `maxspeed` | real | Maximum movement speed | `Create_0.gml:10` |
| `right` | real | Facing direction (-1 or 1) | `Create_0.gml:58` |
| `totdamagetaken` | real | Cumulative damage taken in run | `Create_0.gml:25` |
| `fainted` | real | Fainted/knocked down counter | `Create_0.gml:109` |
| `spirit` | bool | Strong Spirit passive active | `Create_0.gml:82` |

### Ammo Enum

Defined in `scripts/scrAmmoInit/scrAmmoInit.gml`:

| Index | Name |
|-------|------|
| 0 | `Ammo.None` |
| 1 | `Ammo.Bullets` |
| 2 | `Ammo.Shells` |
| 3 | `Ammo.Bolts` |
| 4 | `Ammo.Explosives` |
| 5 | `Ammo.Energy` |

### Race Enum

Defined in `scripts/scrRaces/scrRaces.gml`:

| ID | Race |
|----|------|
| 1 | Fish |
| 2 | Crystal |
| 3 | Eyes |
| 4 | Melting |
| 5 | Plant |
| 6 | Venuz |
| 7 | Steroids |
| 8 | Robot |
| 9 | Chicken |
| 10 | Rebel |
| 11 | Horror |
| 12 | Rogue |
| 13 | BigDog |
| 14 | Skeleton |
| 15 | Frog |
| 16 | Cuz |

### GameCont State (Global, Not Per-Player)

| Variable | Type | Meaning | Defined In |
|----------|------|---------|------------|
| `GameCont.area` | real | Current area ID | `GameCont/Create_0.gml:7` |
| `GameCont.subarea` | real | Sub-area within area | `GameCont/Create_0.gml:8` |
| `GameCont.level` | real | Current stage/level | `GameCont/Create_0.gml:25` |
| `GameCont.loops` | real | Number of loops completed | `GameCont/Create_0.gml:13` |
| `GameCont.kills` | real | Total kills this run | `GameCont/Create_0.gml` |
| `GameCont.hard` | real | Difficulty modifier | `GameCont/Create_0.gml` |
| `GameCont.skillpoints` | real | Available mutation selections | `GameCont/Create_0.gml` |

---

## 3. Enemy State Schema

### Inheritance Hierarchy

```
hitme (base)
└── enemy (all standard enemies)
    └── bossenemy (all bosses)
```

**70+ enemy types** inherit from `enemy`. All bosses inherit from `bossenemy`.

### Enemy Variables

| Variable | Type | Meaning | Defined In |
|----------|------|---------|------------|
| `x` | real | X position | GML built-in |
| `y` | real | Y position | GML built-in |
| `hp` | real | Current health | `enemy/Create_0.gml` |
| `max_hp` | real | Maximum health (scaled by loops) | `enemy/Create_0.gml` |
| `hitid` | real | Enemy type identifier (HitId enum) | Each enemy's `Create_0.gml` |
| `team` | real | Team affiliation (`team_enemy`) | `hitme/Create_0.gml` |
| `givekill` | bool | Whether kill increments `GameCont.kills` | `enemy/Create_0.gml` |
| `hspeed` | real | Horizontal velocity | GML built-in |
| `vspeed` | real | Vertical velocity | GML built-in |
| `gunangle` | real | Aim angle | `enemy/Create_0.gml` |
| `size` | real | Size multiplier | `hitme/Create_0.gml` |
| `raddrop` | real | Radiation dropped on death | `hitme/Create_0.gml` |
| `object_index` | real | GML object type ID | GML built-in |

### Enumerating Enemies

The codebase uses the `with (enemy) { }` pattern universally:

```gml
with (enemy) {
    // this.x, this.y, this.hp, this.hitid accessible here
}
```

Also available: `instance_number(enemy)`, `instance_nearest(x, y, enemy)`, `instance_find(enemy, n)`.

### Enemy Death Flow

1. **Step check:** `objects/enemy/Step_0.gml:6` — `if (hp <= 0) instance_destroy()`
2. **Destroy event:** `objects/enemy/Destroy_0.gml`:
   - Line 1–2: `if givekill && instance_exists(GameCont) { GameCont.kills++ }`
   - Lines 9–28: Corpse creation
   - Line 34: Radiation drops via `scrRadDrop()`
   - Lines 38–119: Player mutation effects (Lucky Shot, Bloodlust, Trigger Fingers)

### HitId Enum (Enemy Type Identifiers)

Defined in `scripts/scrDeathCauses/scrDeathCauses.gml` (lines 4–119). 76+ entries including:
`HitId.Bandit`, `HitId.Maggot`, `HitId.RadMaggot`, `HitId.Scorpion`, `HitId.Rat`, `HitId.Spider`, `HitId.LaserCrystal`, etc.

---

## 4. Input Interception Points

### Input Architecture

Input flows through three layers:

1. **UberCont::Step_0** calls `scrHandleInputsGeneral(index)` each frame
2. `scrHandleInputsGeneral` populates the **`KeyCont` global struct** via `scrSetKeyboardInputs()` or `scrSetGamepadInputs()`
3. **Player::Step_0** reads from `KeyCont` to compute movement, aiming, firing

### KeyCont Struct (All Input State)

**Defined in:** `scripts/InputHandling/InputHandling.gml` (lines 3–25)

All values are arrays indexed by player `index`:

| Field | Type | Meaning |
|-------|------|---------|
| `KeyCont.moving[i]` | bool | Any movement input active |
| `KeyCont.dir_move[i]` | real | Movement direction (0–360) |
| `KeyCont.hold_east[i]` | bool | Holding right |
| `KeyCont.hold_west[i]` | bool | Holding left |
| `KeyCont.hold_nort[i]` | bool | Holding up |
| `KeyCont.hold_sout[i]` | bool | Holding down |
| `KeyCont.dir_fire[i]` | real | Aim direction (0–360) |
| `KeyCont.dis_fire[i]` | real | Aim distance (analogue) |
| `KeyCont.hold_fire[i]` | bool | Holding fire |
| `KeyCont.press_fire[i]` | bool | Fire just pressed |
| `KeyCont.release_fire[i]` | bool | Fire just released |
| `KeyCont.hold_spec[i]` | bool | Holding special ability |
| `KeyCont.press_spec[i]` | bool | Special just pressed |
| `KeyCont.release_spec[i]` | bool | Special just released |
| `KeyCont.press_swap[i]` | bool | Weapon swap just pressed |
| `KeyCont.hold_swap[i]` | bool | Holding weapon swap |
| `KeyCont.press_pick[i]` | bool | Pickup just pressed |
| `KeyCont.hold_pick[i]` | bool | Holding pickup |

### Recommended Injection Point

**Override `KeyCont` values directly after `scrHandleInputsGeneral()` returns**, before `Player::Step_0` reads them. This is the simplest and least invasive approach:

```gml
// In agent mode, replace human input with agent actions:
if (global.agent_mode) {
    KeyCont.dir_move[0]  = agent_action.move_dir;
    KeyCont.moving[0]    = agent_action.moving;
    KeyCont.dir_fire[0]  = agent_action.aim_dir;
    KeyCont.hold_fire[0] = agent_action.firing;
    KeyCont.press_fire[0]= agent_action.fire_pressed;
    // ... etc
}
```

### Default Key Bindings (Reference)

Defined in `scripts/scrOptionsKeymaps/scrOptionsKeymaps.gml`:

| Action | Keyboard | Gamepad |
|--------|----------|---------|
| Move | WASD | Left stick / D-pad |
| Aim | Mouse position | Right stick |
| Fire | Left mouse | Right shoulder |
| Special | Right mouse | Left shoulder |
| Swap | Space | Right shoulder button |
| Pick up | E | Face button 1 |

---

## 5. Room & Level Structure

### Room List

| Room | Purpose | Instances |
|------|---------|-----------|
| `romInit` | Startup, disclaimer | `MakeGame` |
| `romGame` | All gameplay (procedurally generated) | `Vlambeer` |

### Level Transitions

**Portal object** (`objects/Portal/`) handles level completion:
1. Player collides with Portal → `Collision_Player.gml`
2. Portal `Alarm_1.gml` fires → sets `GameCont.is_level_ended = true`, calls `room_restart()`
3. On room restart, `Vlambeer` re-creates `GenCont` which generates a new level

### Area Definitions

Defined in `scripts/scrArea/scrArea.gml` and `scripts/macros_general/macros_general.gml`:

| Constant | Value | Max Subareas |
|----------|-------|-------------|
| `area_campfire` | 0 | 1 |
| `area_desert` | 1 | 3 |
| `area_sewers` | 2 | 1 |
| `area_scrapyards` | 3 | 3 |
| `area_caves` | 4 | 1 |
| `area_city` | 5 | 3 |
| `area_labs` | 6 | 1 |
| `area_palace` | 7 | 3 |
| `area_vault` | 100 | 1 |
| `area_oasis` | 101 | 1 |
| `area_pizza_sewers` | 102 | 1 |
| `area_mansion` | 103 | 1 |
| `area_cursed_caves` | 104 | 1 |
| `area_jungle` | 105 | 1 |
| `area_hq` | 106 | 3 |
| `area_crib` | 107 | 1 |

### Area Progression Logic

**File:** `objects/GameCont/Other_5.gml` (room end event, lines 91–135)

- When `subarea >= maxsubarea`: advance to next area (`area++`), go to campfire (`area_campfire`) for mutation selection
- Otherwise: `subarea++`

### Procedural Generation Seeding

**File:** `scripts/scrRngStatesInit/scrRngStatesInit.gml`

Seed priority:
1. `global.custom_seed` (if set)
2. `UberCont.daily_seed` (if daily run)
3. `UberCont.weekly_data[? "seed"]` (if weekly run)
4. File: `game_directory + "seed.txt"`
5. Console: `Console.seed`
6. Random: `irandom(2147483647)`

**Per-system RNG states** (LCG implementation):

| Index | System |
|-------|--------|
| 0 | Generation (floors/walls) |
| 1 | Enemies |
| 2 | Props |
| 3 | Weapon drops |
| 4 | Chests |
| 5 | Skills/mutations |
| 8 | Popo |
| 9 | Pickups |

### Programmatic New Run

**Function:** `scrRunStart()` in `scripts/scrRunStart/scrRunStart.gml`
- Creates GameCont if missing, sets seed, creates players, calls `room_restart()`

**Full game restart:** `game_restart()` (reloads from `romInit`)

---

## 6. Reward Signal Locations

| Event | Script / Object | Key Code | Variable / Signal |
|-------|----------------|----------|-------------------|
| **Enemy kill** | `objects/enemy/Destroy_0.gml:1-2` | `GameCont.kills++` | Incremental counter |
| **Player damage** | `scripts/scr_hit/scr_hit.gml:18` | `hp -= _amount` | HP decrement |
| **Total damage** | `scripts/scrPlayerProcTakeDamage/scrPlayerProcTakeDamage.gml:4` | `totdamagetaken += _amount` | Cumulative counter |
| **Player death** | `objects/Player/Destroy_0.gml:1-121` | Instance destroyed when `hp <= 0` | Destroy event fires |
| **Area advance** | `objects/GameCont/Other_5.gml:108` | `area++` | Discrete progression |
| **Subarea advance** | `objects/GameCont/Other_5.gml:135` | `subarea++` | Discrete progression |
| **Loop complete** | `objects/GameCont/Other_5.gml:115-116` | `loops++` | Major milestone |
| **Weapon pickup** | `objects/Player/Collision_WepPickup.gml:54` | `wep = other.wep` | Weapon ID change |
| **Health pickup** | `objects/HPPickup/Collision_Player.gml:12` | `scrPlayerHeal(id, _give_amount, true)` | HP increment |
| **Health restore** | `scripts/scrPlayerHeal/scrPlayerHeal.gml:14` | `hp += _amount` | HP increment |
| **Mutation grant** | `objects/SkillIcon/Other_10.gml:5` | `scr_skill_set(skill, scr_skill_get(skill) + 1)` | Skill level increment |
| **Level ended** | `objects/Portal/Other_7.gml:31` | `GameCont.is_level_ended = true` | Boolean flag |

---

## 7. Existing IPC / Network Code

### TCP/UDP Networking (Multiplayer Co-op)

The codebase has a **fully implemented networking stack** for co-op multiplayer:

| Component | Location | Purpose |
|-----------|----------|---------|
| **CoopController** | `objects/CoopController/` (11 event files) | TCP server/client, frame sync, event replay |
| **CoopMenu** | `objects/CoopMenu/` (6 event files) | UDP broadcast for LAN game discovery |
| **NetworkUtils** | `scripts/NetworkUtils/NetworkUtils.gml` | Packet helpers, socket management |

### Packet Helper Functions

Defined in `scripts/NetworkUtils/NetworkUtils.gml`:

```gml
packet_begin(_event)      // Seek to buffer start, write event type
packet_write(_type, _data) // Write typed data to global.mpbuffer
packet_read(_type)         // Read typed data from global.mpbuffer
packet_send()              // Send to all connected sockets
```

### Buffer Types Used

`buffer_u8`, `buffer_u16`, `buffer_u32`, `buffer_f16`, `buffer_f32`, `buffer_string`

### JSON Functions

Both `json_stringify()` and `json_parse()` are used extensively for:
- Player data transmission
- Event serialisation for network replay
- PlayerInstance list serialisation

### Async Networking Events

**File:** `objects/CoopController/Other_68.gml` — handles `network_type_connect`, `network_type_disconnect`, `network_type_data`

### Network Event Enum

```gml
enum event {
    tcp_handshake, tcp_connect, disconnect, ping, latency,
    inputs, player_connect, leave, refuse, set_config,
    broadcast, start, restart, update_playerinstance,
    ready_state, run_start, brutesync
}
```

### File I/O Utilities

| Function | Location | Purpose |
|----------|----------|---------|
| `file_write(path, str)` | `scripts/File/File.gml` | Write string to text file |
| `file_read(path)` | `scripts/File/File.gml` | Read text file to string |
| `struct_secure_save(path, struct)` | `scripts/extra_functions/Extra_Functions.gml` | JSON → base64 → compress → save |
| `struct_secure_load(path)` | `scripts/extra_functions/Extra_Functions.gml` | Load → decompress → base64 → JSON |

### execute_shell_simple Extension

**Location:** `extensions/execute_shell_simple_ext/`
- Windows-only (ShellExecute wrapper)
- Not useful for the socket bridge (Linux target)

### Lockstep Synchronisation

**File:** `objects/CoopController/Other_10.gml`
- Frame-based input delay buffer (`delay = 4`)
- Checksum verification every 4 frames via `gamestate_get_checksum()`
- Game pauses if network inputs are missing

---

## 8. Recommended Socket Bridge Architecture

### Overview

Create a new **`obj_AgentBridge`** object that owns a TCP socket server. It runs alongside the existing game loop, intercepting input and serialising state each frame. A global `agent_mode` flag gates all bridge behaviour.

### Which Object Should Own the TCP Socket

**New object: `obj_AgentBridge`** — persistent, created in `romGame` by `Vlambeer` when `agent_mode` is active. Do NOT reuse `CoopController` — its lockstep sync and event replay systems would conflict with the RL loop.

### Which Event Handles Socket I/O

| Event | Purpose |
|-------|---------|
| **Create** | `network_create_server(network_socket_tcp, 7777, 1)`, init buffers |
| **Step (Begin Step)** | Read action from Python, write to `KeyCont` before `Player::Step_0` |
| **Step (End Step)** | Build state JSON, compute reward, send to Python |
| **Async Networking (Other_68)** | Handle connect/disconnect events |
| **Clean Up** | `network_destroy()`, buffer cleanup |

### Recommended State Message Schema (GML → Python)

```json
{
  "type": "state",
  "frame": 12345,
  "done": false,
  "reward": 1.5,
  "player": {
    "x": 512.0,
    "y": 384.0,
    "hp": 6,
    "max_hp": 8,
    "hspeed": 2.5,
    "vspeed": -1.0,
    "gunangle": 135.0,
    "wep": 5,
    "bwep": 12,
    "ammo": [0, 24, 8, 12, 0, 0],
    "reload": 0,
    "can_shoot": true,
    "roll": false,
    "race": 1,
    "nexthurt": 0,
    "current_frame": 12345
  },
  "enemies": [
    {
      "x": 600.0,
      "y": 300.0,
      "hp": 3,
      "max_hp": 4,
      "hitid": 2,
      "object_index": 1042
    }
  ],
  "game": {
    "area": 1,
    "subarea": 2,
    "level": 3,
    "loops": 0,
    "kills": 17,
    "hard": 0
  }
}
```

**Field types and notes:**

| Field | Type | Notes |
|-------|------|-------|
| `frame` | int | Monotonically increasing frame counter |
| `done` | bool | `true` when player is dead or run complete |
| `reward` | float | Computed by `scr_agent_compute_reward` |
| `player.*` | mixed | All values from Player instance; `ammo` is array of 6 |
| `enemies` | array | One entry per `enemy` instance; cap at nearest N (e.g., 20) for observation space size |
| `game.*` | mixed | From `GameCont` instance |

### Recommended Action Message Schema (Python → GML)

```json
{
  "type": "action",
  "move_dir": 90.0,
  "moving": true,
  "aim_dir": 45.0,
  "fire": true,
  "spec": false,
  "swap": false,
  "pick": false
}
```

| Field | Type | Maps To |
|-------|------|---------|
| `move_dir` | float (0–360) | `KeyCont.dir_move[0]` |
| `moving` | bool | `KeyCont.moving[0]` |
| `aim_dir` | float (0–360) | `KeyCont.dir_fire[0]` |
| `fire` | bool | `KeyCont.hold_fire[0]` (+ derive `press_fire`/`release_fire`) |
| `spec` | bool | `KeyCont.hold_spec[0]` (+ derive press/release) |
| `swap` | bool | `KeyCont.press_swap[0]` |
| `pick` | bool | `KeyCont.press_pick[0]` |

### Reset Message (Python → GML)

```json
{
  "type": "reset"
}
```

Triggers `scrRunStart()` or `game_restart()` to begin a new episode.

### Gotchas and Risks for Agent 02

1. **Begin Step vs Step ordering:** Agent action injection MUST happen before `Player::Step_0` reads `KeyCont`. Use **Begin Step** event in `obj_AgentBridge`, or inject inside `UberCont::Step_0` after `scrHandleInputsGeneral()` returns.

2. **press/release derivation:** `KeyCont.press_fire` and `release_fire` are edge-triggered (true for one frame only). The bridge must track previous frame's `fire` state and derive press/release:
   ```gml
   KeyCont.press_fire[0]   = (agent_fire && !agent_fire_prev);
   KeyCont.release_fire[0] = (!agent_fire && agent_fire_prev);
   ```

3. **Enemy cap in state message:** `instance_number(enemy)` can exceed 100. Cap the `enemies` array to the nearest N enemies (sorted by distance to player) to keep observation space fixed-size.

4. **`room_restart()` destroys all instances:** The agent bridge object must be **persistent** or re-created by the level generation flow. Mark it persistent in the .yy file.

5. **Audio in agent mode:** On macOS native builds, audio works normally. To silence during training, mute via in-game settings or set `audio_master_gain(0)` in `scr_agent_config_init()`. For Docker fallback, use PulseAudio dummy sink or `SDL_AUDIODRIVER=dummy`.

6. **Non-blocking socket:** Use `network_set_config(network_config_use_non_blocking_socket, 1)` as the existing co-op code does. The bridge must handle partial reads and buffering.

7. **Mutation selection screen:** When `LevCont` is active (between levels), the game expects mutation input, not movement. The bridge must detect this state and either auto-select or expose it in the state message. Check `instance_exists(LevCont)`.

8. **Pause state:** `UberCont` has pause logic that deactivates instances. The bridge must ensure the game is never paused in agent mode, or handle pause/unpause explicitly.

9. **`game_set_speed` timing:** Call `game_set_speed(500, gamespeed_fps)` in `obj_AgentBridge::Create` and reset to 30 in `Clean Up`. The existing `TestCont` uncap logic may interfere — disable it in agent mode.

10. **JSON size:** `json_stringify()` works for state serialisation but may be slow with many enemies. If throughput is a bottleneck, switch to binary buffer serialisation using the existing `packet_write()` helpers.

---

*End of Integration Specification — All object names, variable names, script names, and file paths verified against nt-recreated-public rewrite branch.*
