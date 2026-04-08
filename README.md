# Nuclear Throne AI

Reinforcement learning agent for [Nuclear Throne](https://store.steampowered.com/app/242680/Nuclear_Throne/), trained via behavioral cloning from human demonstrations then fine-tuned with PPO. Built on [nt-recreated-public](https://github.com/YellowAfterlife/nt-recreated-public), an open-source GameMaker Studio 2 rebuild.

## Architecture

```
Game (500 FPS)              UDP (port 7777+)              Python RL
┌──────────────────┐        ┌─────────────┐        ┌──────────────────┐
│ AgentBridge       │─state─▸│  JSON UDP   │─state─▸│ NuclearThroneEnv │
│ scr_agent_*       │◂─action│  datagrams  │◂─action│ (Gymnasium)      │
│ UberCont patched  │        └─────────────┘        │ PPO / SB3        │
└──────────────────┘                                └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+ with dependencies: `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`)
- GameMaker Studio 2 (for building the rebuild)
- Nuclear Throne on Steam (for recording demonstrations via NTT)

### 1. Record Human Demonstrations (NTT)

Install the [NTT recording mod](docs/ntt-recording-setup.md), play Nuclear Throne, and collect `.jsonl` demonstration files in `nt-data/`.

### 2. Convert Demonstrations

```bash
python -m nt_rl.bc.ntt_converter --input nt-data --output demonstrations --validate -v
```

### 3. Train Behavioral Cloning Policy

```bash
python scripts/bc_balanced.py
```

This produces a class-balanced BC policy at `checkpoints/bc_balanced/best_state_dict.pt`.

### 4. Build and Launch the Game

```bash
# Single instance (from GameMaker IDE)
./scripts/gm_build.sh run

# Parallel instances (4x, for faster training)
./scripts/gm_build.sh run    # build first, then kill
./scripts/launch_parallel.sh 4
```

The game requires `agent_mode.txt` in its working directory to activate the agent bridge.

### 5. Train PPO

```bash
# Single instance
.venv/bin/python3 scripts/ppo_single.py --timesteps 500000

# 4 parallel instances
.venv/bin/python3 scripts/ppo_single.py --n-envs 4 --timesteps 1000000
```

### 6. Watch the Agent Play

See [Playing with the Agent](docs/playing-with-agent.md).

## Project Structure

```
nt_rl/                      Python RL package
├── env.py                  Gymnasium environment (UDP bridge)
├── obs_utils.py            Observation encoding (112-float vector)
├── config.py               EnvConfig dataclass
├── vec_env.py              Parallel environment factory
├── bc/                     Behavioral cloning
│   ├── train.py            BC training (imitation library)
│   ├── dataset.py          Demonstration dataset loader
│   ├── ntt_converter.py    NTT .jsonl → .npz converter
│   ├── recorder.py         Live gameplay recorder
│   ├── evaluate.py         Policy evaluation
│   └── config.py           BCConfig
├── ppo/                    PPO fine-tuning
│   ├── train.py            PPO training loop + iteration driver
│   ├── config.py           PPOConfig
│   ├── reward_config.py    Reward weights (iteration target)
│   ├── diagnose.py         Pathology detection
│   └── evaluate.py         Standardized evaluation
├── ntt_bridge_adapter.py   File-to-TCP adapter for official NT
└── tests/                  Test suite

nt-recreated-public/        GameMaker rebuild (submodule)
├── scripts/scr_agent_*     GML agent bridge scripts
├── objects/AgentBridge/    UDP bridge object
└── objects/UberCont/       Patched for agent action injection

ntt_mods/                   NTT mods for official Nuclear Throne
├── nt_recorder.mod.gml    Human gameplay recorder
└── nt_agent_bridge.mod.gml Agent bridge (file-based IPC)

scripts/                    Runner scripts
├── gm_build.sh            GameMaker build/run/kill
├── launch_parallel.sh     Launch N parallel game instances
├── ppo_single.py          PPO training entry point
├── bc_balanced.py         Class-balanced BC training
└── play.py                Inference/demo script

nt-data/                   Raw NTT recording logs (.jsonl)
demonstrations/            Converted training data (.npz)
checkpoints/               Model checkpoints
```

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | AI assistant instructions and conventions |
| [INTEGRATION_SPEC.md](INTEGRATION_SPEC.md) | GML ↔ Python bridge protocol specification |
| [GML_BRIDGE_SUMMARY.md](GML_BRIDGE_SUMMARY.md) | GML bridge implementation details |
| [ENVIRONMENT_HANDOFF.md](ENVIRONMENT_HANDOFF.md) | Gymnasium environment design and obs/action spaces |
| [INFRA_README.md](INFRA_README.md) | Infrastructure and Docker setup |
| [BC_TRAINING_SUMMARY.md](BC_TRAINING_SUMMARY.md) | Latest BC training results |
| [docs/playing-with-agent.md](docs/playing-with-agent.md) | How to run the trained agent |
| [docs/ntt-recording-setup.md](docs/ntt-recording-setup.md) | NTT recording mod setup |

### Agent Phase Prompts

| Phase | Prompt |
|-------|--------|
| 1. Repo Analysis | [prompts/agent_01_repo_analysis.md](prompts/agent_01_repo_analysis.md) |
| 2. GML Bridge | [prompts/agent_02_gml_bridge.md](prompts/agent_02_gml_bridge.md) |
| 3. Infrastructure | [prompts/agent_03_infrastructure.md](prompts/agent_03_infrastructure.md) |
| 4. Gymnasium Env | [prompts/agent_04_gymnasium_env.md](prompts/agent_04_gymnasium_env.md) |
| 5. Behavioral Cloning | [prompts/agent_05_behavioural_cloning.md](prompts/agent_05_behavioural_cloning.md) |
| 6. PPO Training | [prompts/agent_06_ppo_training.md](prompts/agent_06_ppo_training.md) |

## Observation Space

112-float vector: 12 player features + 20 enemies x 5 features each. See [obs_utils.py](nt_rl/obs_utils.py) for layout.

## Action Space

`MultiDiscrete([9, 24, 2, 2])`: movement direction (8 + none), aim angle (24 bins x 15deg), shoot, special.

## Key Design Decisions

- **UDP bridge** instead of TCP — eliminates backpressure stalls at high FPS
- **Cumulative rewards** on GML side with Python-side deltas — no reward loss from skipped frames
- **`want_restart` resets** — uses the game's own restart path instead of fragile programmatic resets
- **Class-balanced BC loss** — per-dimension weighted cross-entropy prevents action collapse
- **`program_directory` config detection** — enables parallel instances with per-instance port files
