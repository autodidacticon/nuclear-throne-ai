# Agent 02 — GML Socket Bridge Implementation

## Role
You are a GameMaker Studio 2 / GML developer implementing a TCP socket bridge that exposes game state to an external Python RL agent and accepts synthetic actions in return.

## Context
You are working on the `nt-recreated-public` GML project — an open-source rebuild of Nuclear Throne. A previous agent (Agent 01) has produced `INTEGRATION_SPEC.md` which maps the codebase and specifies the exact socket architecture. You must read that document before writing a single line of code.

Your deliverables will be integrated into the GameMaker project and compiled by Agent 03 (Infrastructure). All code must be correct GML that compiles under GameMaker Studio 2.2+.

## Setup
```bash
# The repo should already be cloned from Agent 01's work
cd nt-recreated-public
# Read the spec before anything else
cat INTEGRATION_SPEC.md
```

## Pre-Implementation Checklist
Before writing code, confirm from `INTEGRATION_SPEC.md`:
- [ ] The name of the controller object that will own the socket
- [ ] The JSON schema for the outgoing state message
- [ ] The JSON schema for the incoming action message
- [ ] The exact variable names for all state fields
- [ ] The input interception points for movement and aim/shoot

If any of these are marked NOT FOUND in the spec, you must search the source yourself before proceeding. Do not write code against variables that have not been verified.

## Implementation Tasks

### Task 1 — Bot Mode Flag
Add a global compile-time or runtime flag `AGENT_MODE` (boolean, default `false`) that enables the entire socket bridge. When `false`, the game runs exactly as before — no socket code executes, no behavior changes. This flag is how Agent 03 will enable bot mode in the Docker build.

Add the flag initialization to the game's primary Create event or macro file:
```gml
// In the appropriate initialization location:
global.agent_mode = false; // Set to true for RL training
```

### Task 2 — Socket Server Object
Create a new GML object: `obj_AgentBridge`

This object is responsible for the full socket lifecycle. Implement the following events:

**Create Event:**
```
- If global.agent_mode is false, exit immediately
- Create a TCP server socket on port 7777 using network_create_server()
- Store socket handle in instance variable
- Initialize a receive buffer
- Log "AgentBridge: listening on port 7777" to stdout
```

**Async - Networking Event:**
```
- Handle ds_map async_load for new client connections (network_type_connect)
- Handle incoming data (network_type_data)
  - Read buffer contents into a string
  - Parse as JSON
  - Validate presence of required action fields
  - Store parsed action in a global struct: global.agent_action
- Handle disconnections gracefully — reset global.agent_action to neutral
```

**Step Event:**
```
- If global.agent_mode is false, exit immediately
- If no client is connected, exit
- Call scr_agent_build_state() to construct current state struct
- Serialize state struct to JSON string
- Send JSON over the client socket
- Apply global.agent_action to the game (call scr_agent_apply_action())
```

**Cleanup / Game End Event:**
```
- Destroy server socket
- Destroy client socket if connected
```

### Task 3 — State Serialization Script
Create script: `scr_agent_build_state`

This script must return a GML struct containing all fields specified in the state JSON schema from `INTEGRATION_SPEC.md`. Use the exact variable names and locations identified by Agent 01.

The struct must include at minimum:
- `player_x`, `player_y` — player position
- `player_hp`, `player_hp_max` — current and max health
- `player_vel_x`, `player_vel_y` — velocity components
- `weapon_id` — current weapon identifier
- `ammo` — current ammo count
- `level` — current level/area number
- `is_dodging` — bool, whether player is in iframe
- `enemies` — array of structs, each with `{x, y, hp, type_id}`
- `pickups` — array of structs for nearby pickups `{x, y, type}`
- `reward` — the reward value computed this step (see Task 5)
- `done` — bool, whether the episode has ended (player is dead)
- `step` — integer step counter, incremented each frame

For the enemies array, use whatever enumeration method Agent 01 identified (likely `with(parent_enemy_object)`). Cap the array at 32 enemies to bound message size.

### Task 4 — Action Application Script
Create script: `scr_agent_apply_action`

This script reads `global.agent_action` and injects it as synthetic input, bypassing or overriding the Input 3 library at the interception points Agent 01 identified.

The action schema (from `INTEGRATION_SPEC.md`) will include:
- `move_dir` — integer 0–7 (8-directional) or -1 for no movement
- `aim_angle` — real 0–359.9 degrees
- `shoot` — bool
- `dodge` — bool

When `global.agent_mode` is true and an action has been received this frame, these values must override whatever the keyboard/gamepad would have provided. The override mechanism depends on what Agent 01 found — if Input 3 uses a virtual input layer, inject there; if input is read directly from keyboard, use a global override struct that the Step event checks first.

Document clearly in code comments which interception approach you used and why.

### Task 5 — Reward Computation Script
Create script: `scr_agent_compute_reward`

This script must be called from the game events Agent 01 identified. It accumulates reward into `global.agent_reward` each frame. Reset the accumulator after it is included in the state message.

Implement the following reward components:
```
+5.0   per enemy killed
+10.0  per level completed
+2.0   for picking up health when hp < hp_max * 0.5
-1.0   per hit taken
-15.0  on death
+0.01  per step survived (survival bonus, discourages passivity)
```

Use a named constant or config struct for all reward weights — do not hardcode magic numbers inline. Agent 03 or future agents will want to tune these without hunting through logic.

### Task 6 — Game Speed Control
In the same location as the `global.agent_mode` flag, add:
```gml
if (global.agent_mode) {
    game_set_speed(500, gamespeed_fps); // Uncapped for Xvfb training
}
```

This must only execute when agent mode is enabled. Normal players must experience the standard 30 FPS game speed.

### Task 7 — Episode Reset
Identify where a new run is started (main menu "play" trigger or equivalent). Add a programmatic reset path:
- A global function `scr_agent_reset_episode()` that restarts the game to the beginning of a new run
- This function must be callable from the socket bridge when the Python side sends `{"command": "reset"}` instead of a normal action message

## Output Artifacts
All new files must be placed in the correct GML project directories:
- New objects → `objects/obj_AgentBridge/`
- New scripts → `scripts/scr_agent_*/`
- The `.yyp` project file must be updated to register all new assets

Produce a summary file `GML_BRIDGE_SUMMARY.md` documenting:
- Every file created or modified, with the reason
- The final state JSON schema (actual field names as implemented)
- The final action JSON schema
- Any deviations from the `INTEGRATION_SPEC.md` recommendations and why
- Known limitations or fragile points Agent 03 should be aware of

## Compilation Verification
You cannot run the game, but you can statically verify your GML by:
1. Checking all referenced variable names against Agent 01's verified list
2. Confirming all `network_*` function calls use correct GML signatures (GMS2 docs: `network_create_server(type, port, max_client)`)
3. Confirming JSON functions used exist in GMS2: `json_stringify()`, `json_parse()`
4. Confirming no script calls functions that don't exist in GML stdlib

List your static verification results in `GML_BRIDGE_SUMMARY.md`.

## Completion Criteria
You are done when:
- `obj_AgentBridge` exists with all four events implemented
- `scr_agent_build_state`, `scr_agent_apply_action`, `scr_agent_compute_reward`, `scr_agent_reset_episode` all exist
- `global.agent_mode` flag gates all bridge behavior
- `game_set_speed(500)` is called when agent mode is active
- `.yyp` file is updated with all new assets registered
- `GML_BRIDGE_SUMMARY.md` is complete

## Do Not
- Modify any existing game logic except at the exact interception points Agent 01 identified
- Remove or alter any existing input handling when `global.agent_mode` is false
- Use deprecated GMS1 functions (`json_encode`, `json_decode` — use `json_stringify`/`json_parse` instead)
- Leave any placeholder or TODO comments in the reward or state scripts — they must be fully implemented
