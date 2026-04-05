**Nuclear Throne**

Reinforcement Learning Agent

*Project Plan · April 2026*

**1. Project Overview**

This project develops a reinforcement learning agent capable of
autonomously playing Nuclear Throne — a procedurally generated,
real-time roguelike shooter by Vlambeer — at a level of competence
comparable to a skilled human player.

**Terminal Objective**

Produce a trained neural network policy that can complete runs of
Nuclear Throne, navigating procedurally generated levels, managing
weapons and health, and defeating enemies and bosses across the full
game progression.

**Source Material**

- Game engine: nt-recreated-public — a GameMaker Studio 2 / GML
  open-source rebuild of Nuclear Throne

- Branch: rewrite (actively maintained)

- Asset extraction: automated via bundled npm scripts; requires a
  licensed Steam copy of Nuclear Throne

- Compilation targets: Windows, Ubuntu, macOS, Android, iOS

**Why This Is Challenging**

- Real-time decision-making at 30+ FPS with reaction-level timing
  requirements

- Procedural generation: no two runs are identical, preventing
  memorisation strategies

- Sparse, delayed rewards: most actions have no immediate payoff; death
  is the primary signal

- Long episode horizon: full runs span 20–60+ minutes, making credit
  assignment difficult

- Compound action space: simultaneous movement, aiming, shooting, and
  dodging

- Partial observability: the camera is local; enemies and hazards exist
  off-screen

**2. Technical Architecture**

The system is built around a TCP socket bridge connecting the GML game
engine to a Python reinforcement learning pipeline. The game runs as
native macOS processes on the M4 Max host, enabling low-latency
communication and full Apple Silicon performance. Docker containers
with Xvfb remain an option for scaled parallel training but are not
required for initial development, testing, or gameplay recording.

**Component Overview**

|                     |                              |                                                         |
|---------------------|------------------------------|---------------------------------------------------------|
| **Component**       | **Technology**               | **Purpose**                                             |
| Game Engine         | GameMaker Studio 2 / GML     | Game logic, physics, procedural generation              |
| Socket Bridge       | GML TCP (port 7777)          | Exposes state, accepts synthetic actions each frame     |
| Display             | Native macOS (dev/record)    | Native rendering for development and gameplay recording |
| Parallelism         | Multiple macOS processes      | Parallel game instances on M4 Max host                  |
| RL Environment      | Python / Gymnasium           | Wraps socket into standard RL env interface             |
| Training            | Stable Baselines3 / PPO      | Policy optimisation with vectorised environments        |
| Imitation Learning  | imitation library (BC)       | Warm-start from human gameplay demonstrations           |
| Experiment Tracking | Weights & Biases             | Training curves, reward monitoring, checkpointing       |
| Compute             | M4 Max MacBook Pro / 128 GB  | Host for Docker containers and GPU-accelerated training |

**Data Flow**

Each training step follows this sequence:

- GML game engine computes one simulation frame

- scr_agent_build_state serialises player, enemy, and level state to
  JSON

- JSON is sent over TCP socket to the Python Gymnasium environment

- Python decodes the state, normalises it into an observation vector,
  and passes it to PPO

- PPO policy produces an action (move direction, aim angle, shoot,
  dodge)

- Action is sent back over the socket as JSON

- scr_agent_apply_action injects the action as synthetic input into the
  game

- Reward is computed in GML and included in the next state message

**Estimated Throughput**

|                               |                      |                   |                    |
|-------------------------------|----------------------|-------------------|--------------------|
| **Configuration**             | **Simulation Speed** | **Parallel Envs** | **Est. Steps/sec** |
| Screen capture, Mac native    | 1×                   | 1–2               | ~60                |
| Socket bridge, macOS native   | 4–10×                | 2–8               | ~500–2,000         |
| Socket bridge + Xvfb (Docker) | 50–200×              | 16–32             | 50,000–200,000     |

**3. Project Phases**

The project is divided into eight sequential phases. Phases 1–4 are
infrastructure and tooling; Phases 5–8 are data collection and machine
learning. Each phase produces a specific artifact that the next phase
depends on.

|           |          |           |              |                       |
|-----------|----------|-----------|--------------|-----------------------|
| **Phase** | **Name** | **Owner** | **Duration** | **Agent Feasibility** |

|       |                                         |          |       |            |
|-------|-----------------------------------------|----------|-------|------------|
| **1** | **Repository Analysis & Specification** | Agent 01 | Day 1 | **✓ High** |

|       |                                      |          |          |                   |
|-------|--------------------------------------|----------|----------|-------------------|
| **2** | **GML Socket Bridge Implementation** | Agent 02 | Days 2–3 | **✓ Medium-High** |

|       |                                  |          |       |            |
|-------|----------------------------------|----------|-------|------------|
| **3** | **Build & Runtime Infrastructure** | Agent 03 | Day 4 | **✓ High** |

|       |                                  |          |       |                 |
|-------|----------------------------------|----------|-------|-----------------|
| **4** | **Python Gymnasium Environment** | Agent 04 | Day 5 | **✓ Very High** |

|          |                                                        |       |         |                |
|----------|--------------------------------------------------------|-------|---------|----------------|
| **⚠ HI** | **Human Intervention: Build Setup & Asset Extraction** | Human | Day 5–6 | **— Required** |

|          |                                            |       |         |                |
|----------|--------------------------------------------|-------|---------|----------------|
| **⚠ HI** | **Human Intervention: Gameplay Recording** | Human | Day 6–7 | **— Required** |

|       |                                              |          |          |            |
|-------|----------------------------------------------|----------|----------|------------|
| **5** | **Behavioural Cloning (Imitation Learning)** | Agent 05 | Days 7–8 | **✓ High** |

|       |                                   |          |           |              |
|-------|-----------------------------------|----------|-----------|--------------|
| **6** | **PPO Fine-tuning (RL Training)** | Agent 06 | Days 9–20 | **~ Medium** |

|       |                              |          |            |              |
|-------|------------------------------|----------|------------|--------------|
| **7** | **Reward Shaping Iteration** | Agent 06 | Days 13–20 | **~ Medium** |

|       |                               |               |         |           |
|-------|-------------------------------|---------------|---------|-----------|
| **8** | **Evaluation & Human Review** | Human + Agent | Day 20+ | **~ Low** |

**Phase Descriptions**

**Phase 1 — Repository Analysis & Specification**

Agent 01 clones the nt-recreated-public repository and performs a
systematic audit of the GML codebase. It locates the game loop, player
and enemy state variables, input handling, room structure, and all
reward signal events. The output is INTEGRATION_SPEC.md — a verified,
implementation-ready specification that Agent 02 uses as its sole source
of truth. No code is written in this phase.

**Phase 2 — GML Socket Bridge**

Agent 02 reads INTEGRATION_SPEC.md and implements the in-game socket
bridge in GML. This includes: obj_AgentBridge (TCP server, per-frame
state serialisation and action application), scr_agent_build_state,
scr_agent_apply_action, scr_agent_compute_reward, and
scr_agent_reset_episode. A global agent_mode flag gates all bridge
behaviour so normal gameplay is unaffected. game_set_speed(500) is
called when agent mode is active.

**Phase 3 — Build & Runtime Infrastructure**

The primary build target is macOS (Apple Silicon native). Agent 03
produces a macOS launch script supporting N parallel game instances
with unique port assignments, a crash-restart wrapper, and a
verify_build.sh that confirms the socket bridge is reachable before
any ML work begins. Docker + Xvfb infrastructure exists as a fallback
for scaled headless training but is not required for development,
gameplay recording, or initial RL training.

**Phase 4 — Python Gymnasium Environment**

Agent 04 implements NuclearThroneEnv (Gymnasium-compatible), a buffered
TCP socket client, full observation and action space definitions, a mock
socket server for testing, and a complete pytest suite. All tests must
pass against the mock server — no Docker dependency in the test suite.
The environment is verified with Gymnasium's built-in check_env
validator.

**Phase 5 — Behavioural Cloning**

Using the human-recorded demonstration dataset, Agent 05 trains an
initial policy via behavioural cloning using the imitation library. BC
gives the agent a non-random starting policy, which is critical: a
random agent in Nuclear Throne almost never progresses past the first
area, making PPO cold-start impractical.

**Phases 6 & 7 — PPO Fine-tuning & Reward Iteration**

Starting from the BC-initialised weights, PPO fine-tunes the policy
through environment interaction. Reward shaping is iterative: agents
monitor W&B training curves, identify pathological behaviours (reward
hacking, policy collapse, degenerate fixed strategies), propose reward
weight adjustments, and relaunch training. Expected 2–4 iteration
cycles.

**Phase 8 — Evaluation**

Quantitative evaluation via kill count, levels reached, and episode
length. Qualitative review by a human watching recorded rollouts. This
phase is the primary checkpoint for determining whether the agent has
achieved the terminal objective or whether additional reward iteration
is required.

**4. Human Interventions**

The project is designed for maximum autonomous agent execution. Four
points require exactly one human action each. Six to eight total hours
of human time are required across the full project.

**Intervention 1 — GameMaker License (One-time, ~30 minutes)** -- COMPLETE

|              |                                                                                                                                                                       |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **REQUIRED** | Acquire a GameMaker Studio 2 license. Activate via the GameMaker IDE on macOS. Build target is macOS native (Apple Silicon). |

Why it cannot be automated: GameMaker license activation requires
browser-based account authentication and payment verification. There is
no headless or API-accessible activation path.

**Intervention 2 — Asset Extraction (One-time, ~15 minutes)** -- COMPLETE

|              |                                                                                                                                                                                                                                          |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **REQUIRED** | Own Nuclear Throne on Steam. With Steam installed and the game downloaded, run: cd "./~ Build-Scripts ~" npm ci npm run regen --game-path "/path/to/Steam/steamapps/common/Nuclear Throne/nuclearthrone.app/Contents/Resources" |

Why it cannot be automated: the game's proprietary assets are protected
by copyright and cannot be redistributed. The extraction scripts require
a locally authenticated Steam installation. This is a one-time operation.
Note: the regen script does not natively support macOS; the --game-path
flag is required to point directly at the .app bundle's Resources directory.

**Intervention 3 — GameMaker macOS Build (One-time, ~30 minutes)** -- IN PROGRESS

|              |                                                                                                                                                                            |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **REQUIRED** | Build the project from the GameMaker IDE targeting macOS. The compiled .app bundle is the game binary used for all RL training. No Ubuntu/Docker cross-compilation needed. |

Why it cannot be automated: GameMaker IDE requires interactive build
initiation the first time. Subsequent builds can use Igor CLI.
The macOS native build avoids the Ubuntu cross-compilation pipeline
entirely, eliminating the need for a Linux SSH target container.

**Intervention 4 — Gameplay Recording (3–5 hours)**

|              |                                                                                                                                                                                                                                                           |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CRITICAL** | Play Nuclear Throne for 3–5 hours with Agent 04's input logger running. The logger simultaneously captures game state and your keyboard/mouse inputs at each frame. The resulting dataset is the sole training signal for behavioural cloning in Phase 5. |

Why it cannot be automated: reinforcement learning agents in Nuclear
Throne almost never progress past the first area from a random
initialisation. Human demonstration data provides the warm-start that
makes subsequent RL training tractable. A heuristic scripted policy is a
possible substitute but produces significantly lower-quality
demonstrations, extending RL training time considerably.

**Quality guidance for recordings:**

- Play deliberately — the agent will learn your decision-making patterns

- Include deaths — they provide negative examples the BC loss function
  needs

- Vary your character and weapon choices across sessions

- Cover as many level areas as possible — early-game bias in
  demonstrations limits late-game competence

**5. Risk Register**

|                                        |                |                                                                                                                                                                                 |                 |
|----------------------------------------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| **Risk**                               | **Likelihood** | **Impact & Mitigation**                                                                                                                                                         | **Owner**       |
| macOS build failure                    | Low            | GameMaker macOS builds are well-supported on Apple Silicon. Mitigate by checking Console.app logs and GameMaker output.                                                         | Human           |
| Silent socket state desync             | Medium         | GML serialisation bug produces corrupt observations with no error signal. Mitigated by JSON schema validation on every message from day one.                                    | Agent 02/04     |
| Reward hacking by PPO                  | High           | Agent exploits unintended reward signal (e.g., farming one enemy). Mitigated by agent monitoring W&B for suspiciously flat episode-length curves and flagging for human review. | Agent 06        |
| GML variable name hallucination        | Medium         | Agent 02 writes code against wrong variable names. Mitigated by Agent 01's mandatory source-verified spec and Agent 02's pre-implementation checklist.                          | Agent 01/02     |
| GameMaker audio in background          | Low            | macOS native build has full audio support. Audio can be muted in-game or via agent_mode config if undesirable during training.                                                   | Agent 03        |
| Insufficient demonstration quality     | Low-Medium     | Poor BC warm-start extends RL training. Mitigated by quality guidance given to the human recorder and by including deaths in the dataset.                                       | Human (Phase 4) |

**6. Success Criteria**

**Minimum Viable Agent**

- Consistently completes at least 3 levels without dying

- Demonstrates active enemy avoidance and targeting behaviour

- Selects and uses weapons rather than ignoring pickups

- Mean episode length \> 5 minutes across 50 evaluation runs

**Target Agent**

- Reaches the Nuclear Throne (final boss) in at least 10% of runs

- Adapts to procedurally varied layouts without exhibiting fixed routing

- Manages health pickups intelligently relative to current HP

- Mean kill count \> 150 per run across 50 evaluation runs

**Stretch Goal**

- Achieves a loop (completes the game and begins a second cycle)

- Demonstrates mutation selection aligned with weapon/playstyle context

**7. Estimated Timeline**

All estimates assume continuous agent execution and available compute.
Human intervention windows are noted separately.

|            |                                              |                         |                |
|------------|----------------------------------------------|-------------------------|----------------|
| **Day(s)** | **Activity**                                 | **Milestone**           | **Human?**     |
| 1          | Agent 01 — Repository analysis               | INTEGRATION_SPEC.md     | No             |
| 2–3        | Agent 02 — GML socket bridge                 | GML_BRIDGE_SUMMARY.md   | No             |
| 4          | Agent 03 — macOS runtime infrastructure       | verify_build.sh passes  | No             |
| 5          | Agent 04 — Gymnasium environment             | pytest suite passes     | No             |
| 5–6        | ⚠ Human: license, assets, macOS build        | Game binary runs        | YES (~1 hr)    |
| 6–7        | ⚠ Human: gameplay recording (3–5 hrs)        | Demonstration dataset   | YES (~4 hrs)   |
| 7–8        | Agent 05 — Behavioural cloning               | BC policy checkpoint    | No             |
| 9–12       | Agent 06 — PPO initial training run          | First RL checkpoint     | No             |
| 13–20      | Agent 06 — Reward iteration (2–4 cycles)     | Stable improving policy | Spot-checks    |
| 20+        | Human evaluation & qualitative review        | Go / No-go decision     | YES            |

**Total estimated human time: 6–8 hours across two concentrated
sessions.**

**8. Agent Handoff Chain**

Each agent receives the outputs of all prior agents and produces a
specific artifact. No agent begins work until its predecessor's
completion criteria are met.

|           |                                               |                                                     |                                                            |
|-----------|-----------------------------------------------|-----------------------------------------------------|------------------------------------------------------------|
| **Agent** | **Reads**                                     | **Produces**                                        | **Completion Gate**                                        |
| 01        | GML source (live repo)                        | INTEGRATION_SPEC.md                                 | All 8 spec sections complete; no unverified variable names |
| 02        | INTEGRATION_SPEC.md                           | GML bridge objects & scripts; GML_BRIDGE_SUMMARY.md | Static verification passes; .yyp updated                   |
| 03        | GML_BRIDGE_SUMMARY.md; INFRA_README.md        | macOS launch scripts; verify_build.sh               | verify_build.sh exits 0                                    |
| 04        | GML_BRIDGE_SUMMARY.md; INFRA_README.md        | nt_rl/ Python package; test suite                   | pytest passes; check_env passes                            |
| 05        | Demonstration dataset; ENVIRONMENT_HANDOFF.md | BC policy checkpoint (.zip)                         | Val loss converged; policy non-random in eval              |
| 06        | BC checkpoint; W&B project                    | PPO checkpoints; reward iteration log               | Mean levels reached \> 3 across 50 eval runs               |

*Nuclear Throne RL Agent Project · Confidential · April 2026*