# Nuclear Throne RL Agent — Infrastructure Guide

*Phase 3 Output — Agent 03 — 2026-04-04*

---

## Prerequisites

### Required Software
- **Docker Desktop** (v4.25+) with Rosetta/QEMU support enabled (for ARM64 hosts)
- **GameMaker Studio 2** license file (`gamemaker-license.pem`)
- **Nuclear Throne** on Steam (for asset extraction)

### Required Files on Host
```
nt-recreated-public/
├── gamemaker-license.pem     # GameMaker license (human-provided, one-time)
├── build-output/             # Pre-compiled game binary + assets
│   ├── NuclearThrone          # The compiled Ubuntu executable
│   └── assets/                # Compiled game assets
├── datafiles/                # Game data files (from npm run regen)
└── ...                       # Rest of the GML project
```

### Asset Extraction (One-Time)
With Nuclear Throne installed on Steam:
```bash
cd "~ Build-Scripts ~"
npm ci
npm run regen
```

---

## Building

### Build the Docker Image
```bash
cd nt-recreated-public
docker build -t nt-agent .
```

**Note:** The Dockerfile uses `--platform=linux/amd64` because GameMaker's Linux runner is x86-64 only. On Apple Silicon (M4 Max), Docker Desktop uses Rosetta 2 for emulation. Performance penalty is ~20% which is acceptable since simulation speed is not the bottleneck.

### Build with GameMaker License Secret
If using the Igor builder stage:
```bash
docker build --secret id=gm_license,src=./gamemaker-license.pem -t nt-agent .
```

---

## Running

### Single Instance
```bash
docker run -d \
  -p 7777:7777 \
  -v ./build-output:/game/bin:ro \
  --name nt-agent-1 \
  nt-agent
```

### 4 Parallel Instances (Default)
```bash
docker compose up -d
```

Port mapping:
| Container | Host Port | Display |
|-----------|-----------|---------|
| nt-agent-1 | 7777 | :99 |
| nt-agent-2 | 7778 | :100 |
| nt-agent-3 | 7779 | :101 |
| nt-agent-4 | 7780 | :102 |

### N Parallel Instances (Custom)
```bash
# Generate override file for 16 instances
./scripts/scale.sh 16

# Launch all 16
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

Port mapping for N instances: `localhost:(7777 + i - 1)` maps to container port 7777 for instance `i`.

### Verify the Build
```bash
./scripts/verify_build.sh
```
Exits 0 if the socket bridge is reachable on port 7777. Exits 1 with container logs on failure.

---

## Debugging

### View Container Logs
```bash
docker logs nt-agent-1 -f
```

### View the Xvfb Display
Attach a VNC server to the virtual framebuffer for visual debugging:
```bash
# Install x11vnc inside a running container
docker exec nt-agent-1 apt-get update && apt-get install -y x11vnc
docker exec -d nt-agent-1 x11vnc -display :99 -nopw -listen 0.0.0.0 -forever

# Connect from host with any VNC client on port 5900
```

Alternatively, capture a screenshot:
```bash
docker exec nt-agent-1 xwd -root -display :99 | convert xwd:- screenshot.png
```

Or stream via ffmpeg:
```bash
docker exec nt-agent-1 ffmpeg -f x11grab -i :99 -vframes 1 -y /tmp/frame.png
docker cp nt-agent-1:/tmp/frame.png ./debug-frame.png
```

### Shell into a Running Container
```bash
docker exec -it nt-agent-1 /bin/bash
```

### Check if Socket is Listening
```bash
docker exec nt-agent-1 nc -z localhost 7777 && echo "OK" || echo "NOT LISTENING"
```

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DISPLAY_NUM` | `99` | Xvfb display number (unique per container) |
| `GAME_BINARY` | `./NuclearThrone` | Path to game executable |
| `RESTART_DELAY` | `2` | Seconds between crash restarts |
| `MAX_RESTARTS` | `1000` | Maximum restart attempts before container exits |
| `SDL_AUDIODRIVER` | `pulseaudio` | Audio driver (fallback: `dummy`) |

### Agent Mode
Agent mode is activated by the presence of `agent_mode.txt` in the game's working directory. The Dockerfile and entrypoint create this file automatically. No code changes needed.

---

## Rebuilding After GML Changes

If you modify the GML bridge code (Phase 2 files):

1. **Recompile** the game using GameMaker IDE or Igor CLI
2. **Copy** the new binary to `build-output/`
3. **Restart** containers:
   ```bash
   docker compose restart
   ```

If you only change Python code (Phase 4+), no Docker rebuild is needed — the Python agent connects over TCP from the host.

---

## Architecture Notes

### Why x86-64 Emulation?
GameMaker's Linux runner binary is compiled for x86-64 only. There is no ARM64 native build. On Apple Silicon:
- Docker Desktop uses Rosetta 2 for transparent emulation
- Performance overhead is ~20% on M4 Max
- This is acceptable because simulation speed (50-200x real-time) is compute-bound on GML game logic, not the container overhead

### Why Xvfb?
GameMaker requires a display server for rendering, even headless. Xvfb (X Virtual Framebuffer) provides a virtual display with:
- No GPU required (software rendering)
- Configurable resolution (320x240 matches game native)
- No physical display needed

### Why PulseAudio Dummy Sink?
GameMaker initializes the audio subsystem at startup and may crash without an audio device. The PulseAudio null sink provides a virtual audio output that discards all samples. Fallback: `SDL_AUDIODRIVER=dummy` bypasses PulseAudio entirely.

### Crash Restart Loop
The entrypoint script automatically restarts the game if it crashes. This is essential during RL training because:
- Edge cases in procedural generation may trigger crashes
- The Python agent handles socket reconnection gracefully
- Training should not stop due to transient game failures

---

## Known Issues

1. **Igor CLI availability**: Igor is distributed as part of GameMaker Studio 2 and may require GUI-based license activation the first time. The Dockerfile builder stage assumes Igor is pre-installed or mounted. First-time setup requires human intervention (see plan Phase HI).

2. **GameMaker audio crash**: If PulseAudio fails to start and `SDL_AUDIODRIVER=dummy` doesn't work, try wrapping the game binary with `padsp`: modify entrypoint to use `padsp ./NuclearThrone`.

3. **Xvfb GLX extensions**: Some GameMaker versions require GLX. The Xvfb command includes `+extension GLX` but software GLX may not support all features. If the game fails to start with GLX errors, try `LIBGL_ALWAYS_SOFTWARE=1`.

4. **Container memory**: Each game instance uses ~200-400 MB RAM. For 32 parallel instances, ensure the Docker host has at least 16 GB allocated to Docker.

5. **File descriptors**: Many parallel containers with TCP sockets may hit fd limits. Increase with `ulimit -n 65535` on the host if needed.
