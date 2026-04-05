# GML Socket Bridge Summary

*Phase 2 Output — Agent 02 — 2026-04-04*

---

## Files Created

| File | Purpose |
|------|---------|
| `objects/AgentBridge/AgentBridge.yy` | Object definition (persistent, Controllers folder) |
| `objects/AgentBridge/Create_0.gml` | TCP server socket creation on port 7777 |
| `objects/AgentBridge/Step_0.gml` | Per-frame: compute reward, build state, send JSON to Python |
| `objects/AgentBridge/Other_68.gml` | Async networking: client connect/disconnect, parse incoming JSON actions |
| `objects/AgentBridge/CleanUp_0.gml` | Destroy sockets and buffers on shutdown |
| `scripts/scr_agent_config/scr_agent_config.gml` | Agent mode init, reward weights, game speed config |
| `scripts/scr_agent_config/scr_agent_config.yy` | Script registration |
| `scripts/scr_agent_build_state/scr_agent_build_state.gml` | Serialise player, enemies, GameCont state to struct |
| `scripts/scr_agent_build_state/scr_agent_build_state.yy` | Script registration |
| `scripts/scr_agent_apply_action/scr_agent_apply_action.gml` | Inject agent actions into KeyCont input struct |
| `scripts/scr_agent_apply_action/scr_agent_apply_action.yy` | Script registration |
| `scripts/scr_agent_compute_reward/scr_agent_compute_reward.gml` | Per-frame reward accumulation + heal reward hook |
| `scripts/scr_agent_compute_reward/scr_agent_compute_reward.yy` | Script registration |
| `scripts/scr_agent_reset_episode/scr_agent_reset_episode.gml` | Programmatic game restart for new RL episode |
| `scripts/scr_agent_reset_episode/scr_agent_reset_episode.yy` | Script registration |

## Files Modified

| File | Change | Reason |
|------|--------|--------|
| `objects/UberCont/Create_0.gml` | Added `scr_agent_config_init()` call | Initialise agent mode early in game startup |
| `objects/UberCont/Step_0.gml` | Added `scr_agent_apply_action()` after `scrHandleInputsGeneral()` | Inject agent actions before Player::Step reads KeyCont |
| `objects/Vlambeer/Create_0.gml` | Added AgentBridge instance creation | Create bridge when entering romGame |
| `objects/HPPickup/Collision_Player.gml` | Added `scr_agent_reward_heal()` call | Reward signal for health pickup |
| `nuclearthronemobile.yyp` | Added 7 resource entries (1 object + 6 scripts) | Register new assets in project |

---

## State JSON Schema (GML → Python)

Sent as newline-delimited JSON every frame:

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
    {"x": 600.0, "y": 300.0, "hp": 3, "max_hp": 4, "hitid": 2}
  ],
  "game": {
    "area": 1,
    "subarea": 2,
    "level": 3,
    "loops": 0,
    "kills": 17,
    "hard": 0
  },
  "mutation_screen": false
}
```

## Action JSON Schema (Python → GML)

Sent as newline-delimited JSON:

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

## Reset Command (Python → GML)

```json
{"type": "reset"}
```

---

## Deviations from INTEGRATION_SPEC.md

| Spec Recommendation | Implementation | Reason |
|---------------------|----------------|--------|
| `agent_mode` as compile-time macro | Runtime flag via `agent_mode.txt` file | More flexible — create/delete a file instead of recompiling. On macOS, place alongside the game binary or in the .app bundle's working directory. Checked once at init. |
| Begin Step event for action injection | Patched into UberCont::Step_0 after `scrHandleInputsGeneral()` | Same effect — UberCont Step runs before Player Step. Keeps all agent logic visible in one location. |
| `object_index` in enemy state | Omitted in favour of `hitid` only | `hitid` is the semantic identifier; `object_index` is an opaque engine integer that varies across builds. |
| Spec suggested `dis_fire` not needed | Set `dis_fire = 100` (constant) | Required for aim assist calculations; constant value simulates mouse far from player. |

## Reward Weights

All configurable in `scr_agent_config.gml`:

| Signal | Weight | Variable |
|--------|--------|----------|
| Enemy killed | +5.0 | `global.agent_reward_kill` |
| Level completed | +10.0 | `global.agent_reward_level` |
| Health pickup (when < 50% HP) | +2.0 | `global.agent_reward_heal_low` |
| Hit taken | -1.0 | `global.agent_reward_hit` |
| Death | -15.0 | `global.agent_reward_death` |
| Per-step survival | +0.01 | `global.agent_reward_survival` |

---

## Static Verification

| Check | Result |
|-------|--------|
| All Player variables (`hp`, `max_hp`, `x`, `y`, `hspeed`, `vspeed`, `gunangle`, `wep`, `bwep`, `ammo`, `reload`, `can_shoot`, `roll`, `race`, `nexthurt`, `current_frame`) verified against `objects/Player/Create_0.gml` | PASS |
| All KeyCont fields (`moving`, `dir_move`, `hold_east/west/nort/sout`, `dir_fire`, `dis_fire`, `hold_fire`, `press_fire`, `release_fire`, `hold_spec`, `press_spec`, `release_spec`, `press_swap`, `hold_swap`, `press_pick`, `hold_pick`) verified against `scripts/InputHandling/InputHandling.gml` | PASS |
| GameCont variables (`area`, `subarea`, `level`, `loops`, `kills`, `hard`) verified against `objects/GameCont/Create_0.gml` | PASS |
| Enemy variables (`x`, `y`, `hp`, `max_hp`, `hitid`) verified against `objects/enemy/Create_0.gml` and `objects/hitme/Create_0.gml` | PASS |
| `network_create_server(type, port, max_client)` — correct GMS2 signature | PASS |
| `network_destroy(socket)` — correct GMS2 signature | PASS |
| `network_send_packet(socket, buffer, size)` — correct GMS2 signature | PASS |
| `json_stringify()` / `json_parse()` — GMS2.3+ functions (not deprecated `json_encode`/`json_decode`) | PASS |
| `buffer_create(size, buffer_grow, alignment)` — correct GMS2 signature | PASS |
| `array_sort()` with comparison function — GMS2.3+ | PASS |
| `variable_struct_exists()` — GMS2.3+ | PASS |
| `instance_create(x, y, obj)` — uses GMS1-compatible version (project uses this throughout, not `instance_create_layer`) | PASS |
| `scrRunStart()` verified to exist at `scripts/scrRunStart/scrRunStart.gml` | PASS |
| `point_distance()`, `lengthdir_x()`, `lengthdir_y()` — standard GML functions | PASS |

---

## Known Limitations & Fragile Points

1. **Mutation screen handling:** When `LevCont` is active (between levels), the game expects mutation selection input. The `mutation_screen` flag is included in state so Python can detect this, but the bridge does not auto-select mutations. Python must either send swap/pick actions to select, or a future enhancement could add auto-selection in agent mode.

2. **TCP fragmentation:** Messages use newline-delimited JSON (`\n`). The async networking handler accumulates partial reads in `recv_string`. If the Python client sends very large messages or sends faster than the game processes, the buffer may grow. In practice, action messages are < 200 bytes so this is not a concern.

3. **`room_restart()` and persistence:** AgentBridge is marked `persistent: true` in its .yy file, so it survives room restarts. However, if `game_restart()` is called (which reloads from `romInit`), all instances including AgentBridge are destroyed and re-created when `Vlambeer::Create_0` runs again. The socket server is destroyed and re-created — Python must reconnect.

4. **`scrRunStart()` creates GameCont:** The `scr_agent_reset_episode()` function destroys the existing GameCont before calling `scrRunStart()`, which creates a fresh one. If `scrRunStart()` changes in future repo updates, the reset path may break.

5. **Agent mode activation:** Agent mode is activated by the presence of a file named `agent_mode.txt` in the game's working directory. For macOS native builds, place the file alongside the game binary or use a launch script that creates it. For Docker, the entrypoint creates it automatically.

6. **No pause in agent mode:** The bridge does not explicitly disable pausing. If the game pauses (e.g., focus loss on macOS), the Step event will still fire (AgentBridge is persistent), but Player instances may be deactivated. On macOS, use App Nap prevention (`defaults write`) or keep the game window focused. For Docker, this is not an issue.
