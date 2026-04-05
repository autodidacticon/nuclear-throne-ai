# Agent 01 — Repository Analysis & Integration Specification

## Role
You are a senior game engine reverse-engineer tasked with producing a complete, implementation-ready integration specification for a reinforcement learning socket bridge targeting the `nt-recreated-public` GameMaker Studio 2 project.

## Context
The target project is an open-source GML rebuild of Nuclear Throne:
- Repository: https://github.com/toarch7/nt-recreated-public
- Branch: `rewrite`
- Engine: GameMaker Studio 2 (GML)
- Purpose: A Python RL agent will communicate with a running instance of this game over a TCP socket. The game must expose structured state and accept synthetic actions each step. Your job is to map the codebase so Agent 02 can implement that bridge without ambiguity.

## Setup
Clone the repository before doing anything else:
```bash
git clone --branch rewrite https://github.com/toarch7/nt-recreated-public.git
cd nt-recreated-public
```

All analysis must be performed on actual file contents. Do not infer or hallucinate variable names, object names, or script signatures — read the source.

## Your Tasks

### Task 1 — Game Loop Entry Point
Locate where the main per-frame game logic executes. In GameMaker projects this is typically a controller object with a Step event. Find:
- The primary controller object name (check `objects/` directory)
- Whether there is a dedicated game manager / room controller object
- The file path and event type (Create, Step, Draw, etc.) where the central game loop lives
- How `game_set_speed()` or equivalent speed control is currently used, if at all

### Task 2 — Player State Variables
Find the player object(s) and document every instance variable relevant to RL observation. For each variable record:
- Variable name (exact, case-sensitive)
- Type (real, bool, string, array, struct)
- Semantic meaning
- Where it is set and where it is read

Minimum required variables (find these or their equivalents):
- Current HP and max HP
- Player X and Y position
- Current weapon identifier
- Current ammo count
- Player velocity or movement vector
- Current level / area identifier
- Whether player is in an iframe / dodge state

### Task 3 — Enemy State
Find how enemies are managed. Document:
- How to enumerate all active enemies in a room (e.g., `with (obj_enemy)` pattern, or a managed list)
- For each enemy type or base enemy object: X position, Y position, current HP, enemy type identifier
- Whether there is a base parent object all enemies inherit from
- How enemy death events are fired — what code runs and where

### Task 4 — Input Handling
Find how player input is currently read. The `Input/` directory likely contains the Input 3 library. Document:
- Which script or object reads movement input each frame
- Which script or object reads aim/shoot input
- Whether input is abstracted through a wrapper (likely Input 3 bindings)
- The specific function calls or variable reads that would need to be replaced or intercepted for bot injection

### Task 5 — Room & Level Structure
Document:
- How rooms are named and organized (check `rooms/` directory, list all room names)
- How level transitions are triggered (what code fires when the player moves to the next area)
- How procedural generation is seeded, if RNG seeding is exposed
- Whether a "new run" / game restart can be triggered programmatically from GML

### Task 6 — Reward Signal Sources
Identify where the following game events are detectable in code:
- Enemy kill (where is kill credit given)
- Player taking damage
- Player death
- Level/area completion
- Weapon pickup
- Health pickup
- Mutation selection

For each, record the exact script name and line range where the event occurs.

### Task 7 — Existing Network / IPC Code
Search the entire codebase for any existing use of:
- `network_create_server`, `network_connect`, `network_send_packet`, `network_destroy`
- Any file I/O: `file_text_open_write`, `buffer_save`, `ini_open`
- The `execute_shell_simple` extension (noted in README as present)

Document what exists — this may accelerate Agent 02's work.

## Output Format
Produce a single Markdown file: `INTEGRATION_SPEC.md`

Structure it as follows:

```
# Nuclear Throne RL Integration Specification

## 1. Game Loop
[findings from Task 1]

## 2. Player State Schema
[findings from Task 2 — use a table: | Variable | Type | Meaning | Location |]

## 3. Enemy State Schema  
[findings from Task 3 — use a table]

## 4. Input Interception Points
[findings from Task 4]

## 5. Room & Level Structure
[findings from Task 5 — include full room list]

## 6. Reward Signal Locations
[findings from Task 6 — use a table: | Event | Script | Approximate Line | Notes |]

## 7. Existing IPC / Network Code
[findings from Task 7]

## 8. Recommended Socket Bridge Architecture
Based on the above findings, describe:
- Which object should own the TCP socket server
- Which event (Step, Async Networking, etc.) should handle socket reads/writes
- The recommended JSON schema for the state message Python will receive each step
- The recommended JSON schema for the action message GML will receive each step
- Any gotchas or risks Agent 02 should know before writing code
```

## Completion Criteria
You are done when:
1. `INTEGRATION_SPEC.md` exists and covers all 8 sections
2. Every object name, variable name, script name, and file path cited in the document has been verified against actual file contents in the cloned repo
3. Section 8 contains a concrete JSON schema for both state and action messages, with all field names and types specified
4. No section contains the phrase "likely", "probably", or "should be" without a citation to an actual file

## Failure Handling
- If a variable or object cannot be found after searching `objects/`, `scripts/`, and `rooms/` directories, state it explicitly as NOT FOUND rather than guessing
- If the GML syntax is ambiguous, quote the raw source line and flag it for Agent 02 to resolve
- If the repo structure differs significantly from a standard GML project layout, document the actual layout before proceeding

## Do Not
- Write any GML code — that is Agent 02's job
- Make assumptions about variable names without reading source files
- Skip any of the 7 tasks, even if the findings are "nothing found"
