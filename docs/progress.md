# Nuclear Throne RL Agent — Progress Tracker

*Started: 2026-04-04*

## Phase 1 — Repository Analysis & Specification (Agent 01) -- COMPLETE

- [x] Clone nt-recreated-public repo (rewrite branch)
- [x] Task 1: Game loop entry point analysis — UberCont primary loop, 30 FPS default
- [x] Task 2: Player state variables documentation — 80+ vars, core RL obs identified
- [x] Task 3: Enemy state schema documentation — hitme→enemy→bossenemy hierarchy, 70+ types
- [x] Task 4: Input handling analysis — KeyCont struct injection point identified
- [x] Task 5: Room & level structure mapping — 2 rooms, procedural gen via room_restart()
- [x] Task 6: Reward signal source identification — 11 reward events mapped with exact locations
- [x] Task 7: Existing network/IPC code audit — full TCP/UDP stack, JSON, packet helpers
- [x] Task 8: Recommended socket bridge architecture — obj_AgentBridge, JSON schemas defined
- [x] Produce INTEGRATION_SPEC.md

## Phase 2 — GML Socket Bridge (Agent 02) -- COMPLETE

- [x] Implement obj_AgentBridge (TCP server) — 4 events: Create, Step, Async Networking, CleanUp
- [x] Implement scr_agent_build_state — player, enemies (nearest 20), GameCont state
- [x] Implement scr_agent_apply_action — KeyCont override with press/release derivation
- [x] Implement scr_agent_compute_reward — 6 reward signals with configurable weights
- [x] Implement scr_agent_reset_episode — destroys GameCont, calls scrRunStart()
- [x] Add agent_mode flag gating — runtime flag via agent_mode.txt, scr_agent_config_init()
- [x] Produce GML_BRIDGE_SUMMARY.md — schemas, static verification, known limitations
- [x] Patch UberCont (init + input injection), Vlambeer (bridge creation), HPPickup (reward hook)
- [x] Update nuclearthronemobile.yyp with 7 new resource entries

## Phase 3 — Build & Runtime Infrastructure (Agent 03) -- NEEDS UPDATE

**Decision: macOS native build replaces Ubuntu/Docker as primary target.**
Rationale: GameMaker macOS builds run natively on Apple Silicon (M4 Max)
with zero emulation overhead. Docker/Xvfb remains as optional fallback
for scaled headless training.

- [x] Docker infrastructure (Ubuntu 22.04) — completed but deprioritized
- [x] docker-compose.yml (4 parallel instances) — completed but deprioritized
- [x] Entrypoint script with Xvfb, PulseAudio dummy, crash-restart — completed but deprioritized
- [x] INFRA_README.md — needs update for macOS-first workflow
- [ ] macOS launch script for N parallel game instances with unique ports
- [ ] macOS crash-restart wrapper
- [ ] verify_build.sh updated for macOS binary
- [ ] Agent mode activation for macOS (.app bundle)

## Phase 4 — Python Gymnasium Environment (Agent 04) -- COMPLETE

- [x] NuclearThroneEnv (Gymnasium-compatible) — full Gymnasium.Env with reset/step/close
- [x] Buffered TCP socket client — newline-delimited JSON, reconnect on crash
- [x] Observation space: Box(112,) — 12 player + 20×5 enemy features
- [x] Action space: MultiDiscrete([9, 24, 2, 2]) — move, aim, shoot, special
- [x] Mock socket server for testing — simulates GML bridge
- [x] pytest suite — 9/9 tests pass including gymnasium check_env
- [x] SubprocVecEnv factory for parallel training
- [x] ENVIRONMENT_HANDOFF.md — obs layout, action encoding, PPO hyperparameters

## Phase 5 — Behavioural Cloning (Agent 05)

- [ ] Human gameplay recording (HI required)
- [ ] BC training pipeline
- [ ] BC policy checkpoint

## Phase 6 & 7 — PPO Fine-tuning & Reward Iteration (Agent 06)

- [ ] PPO training from BC weights
- [ ] W&B experiment tracking
- [ ] Reward shaping iteration (2-4 cycles)
- [ ] Stable improving policy

## Phase 8 — Evaluation

- [ ] Quantitative evaluation (kill count, levels, episode length)
- [ ] Qualitative human review of rollouts
- [ ] Go / No-go decision
