# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reinforcement learning agent for Nuclear Throne, built on `nt-recreated-public` (a GameMaker Studio 2 open-source rebuild). A GML TCP socket bridge sends game state as JSON; a Python Gymnasium environment receives it and drives PPO training via Stable Baselines3.

Phases 1-4 are complete (spec, GML bridge, infra, Gymnasium env). Phase 5+ (behavioral cloning, PPO training) are pending.

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run tests (9 tests, uses mock socket server — no game needed)
pytest nt_rl/tests/test_env.py -v

# Run a single test
pytest nt_rl/tests/test_env.py::test_reset -v

# Docker build (game simulation container, optional — macOS native is primary)
cd nt-recreated-public && docker build -t nt-agent .

# Docker parallel instances (ports 7777-7780)
cd nt-recreated-public && docker compose up -d
```

## Architecture

```
GML Game (30-500 FPS)          TCP/JSON (port 7777+)          Python RL
┌──────────────────┐           ┌─────────────┐           ┌──────────────────┐
│ obj_AgentBridge   │──state──▸│ newline-     │──state──▸│ NuclearThroneEnv │
│ scr_agent_*       │◂─action──│ delimited    │◂─action──│ (Gymnasium)      │
│ UberCont patched  │          │ JSON socket  │          │ PPO / SB3        │
└──────────────────┘           └─────────────┘           └──────────────────┘
```

**GML side** (`nt-recreated-public/scripts/scr_agent_*`):
- `scr_agent_build_state` — collects player (12 features) + nearest 20 enemies (5 features each) + game state into JSON
- `scr_agent_apply_action` — overrides `KeyCont` struct to inject actions before `Player::Step_0`
- `scr_agent_compute_reward` — 6 reward signals (kill, level, health pickup, damage, death, survival)
- `scr_agent_config` — configurable weights and constants, gated by `agent_mode.txt` file presence
- `scr_agent_reset_episode` — destroys `GameCont`, calls `scrRunStart()`

**Python side** (`nt_rl/`):
- `env.py` — `NuclearThroneEnv`: Gymnasium wrapper, buffered TCP client, newline-delimited JSON protocol
- `obs_utils.py` — `encode_observation()`: normalizes JSON state to 112-float vector
- `config.py` — `EnvConfig` dataclass: all tunable parameters (ports, spaces, normalization constants)
- `vec_env.py` — `SubprocVecEnv` factory for parallel training across N game instances
- `tests/mock_server.py` — simulated GML bridge for testing without the game

**Spaces:**
- Observation: `Box(112,)` — 12 player features + 20 enemies x 5 features, all normalized [0,1]
- Action: `MultiDiscrete([9, 24, 2, 2])` — move direction (8 cardinal + none), aim angle (24 bins x 15deg), shoot, special

## Key Conventions

- Reward weights are authoritative in GML (`scr_agent_config.gml`); Python `EnvConfig` mirrors them for reference only
- Agent mode activates at runtime via `agent_mode.txt` file presence — no recompilation needed
- Each parallel game instance uses a unique port (base 7777, incrementing)
- Primary build target is macOS native on Apple Silicon; Docker/Xvfb is optional fallback
- Phase handoff documents (`INTEGRATION_SPEC.md`, `GML_BRIDGE_SUMMARY.md`, `INFRA_README.md`, `ENVIRONMENT_HANDOFF.md`) contain detailed specs per phase
- Agent prompts live in `prompts/agent_0N_*.md` — one per project phase

## Devcontainer

The `.devcontainer/` provides a full development environment with Python ML toolchain, Docker-in-Docker, and firewall. The firewall (`init-firewall.sh`) restricts outbound traffic for the `vscode` user only — root/dockerd traffic is unrestricted. The container workspace is mounted at the same path as the host (`/Users/richard/git/nuclear-throne-ai`) so Claude Code conversations are portable between host and container.
