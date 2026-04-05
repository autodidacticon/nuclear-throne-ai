# Agent 03 — Docker, Xvfb & GameMaker Ubuntu Build Infrastructure

## Role
You are a DevOps / infrastructure engineer responsible for containerizing the modified Nuclear Throne GML project so it runs headlessly under Xvfb on ARM Linux (Docker on Apple Silicon), with the agent socket bridge active, at uncapped simulation speed.

## Context
Agents 01 and 02 have produced a modified GML codebase with a TCP socket bridge on port 7777 and `global.agent_mode = true` enabling headless RL training. Your job is to produce Docker infrastructure that:
1. Builds the GameMaker Ubuntu executable from source
2. Runs it under Xvfb with no physical display required
3. Exposes port 7777 for Python RL agent communication
4. Scales to N parallel instances via docker-compose

Read `GML_BRIDGE_SUMMARY.md` before starting. Note any fragile points Agent 02 flagged.

## Critical Constraints
- Target architecture: **ARM64** (Apple Silicon M4 Max host)
- Target OS inside container: **Ubuntu 22.04 LTS**
- GameMaker build tool: **Igor** (GameMaker's CLI compiler)
- The GameMaker project requires a **licensed GameMaker installation** — the Dockerfile must accommodate a pre-provided license file via a mounted secret or build arg, NOT hardcoded credentials
- Assets are **not in the repo** — they must be injected via a Docker volume or build-time mount from the host (the human has already run `npm run regen` and the asset directories are populated locally)

## Prerequisites Assumed Present on Host
The following are assumed to exist on the host machine before any Docker command runs:
- `nt-recreated-public/` — the fully cloned and asset-populated repo
- `gamemaker-license.pem` or equivalent license file (human-provided)
- Docker Desktop for Mac with Rosetta/ARM support enabled

## Task 1 — Dependency Research
Before writing a Dockerfile, look up the current GameMaker Linux runner dependencies. Search for:
- "GameMaker Studio 2 Ubuntu runner dependencies 2024"
- "GameMaker Igor CLI headless Linux"
- "YoYo Games Ubuntu runtime requirements"

Document the required system packages in your output. Common requirements include specific versions of: `libopenal`, `libssl`, `libcurl`, `libuuid`, audio daemon substitutes (PulseAudio or dummy), and the GameMaker runner binary format.

If web search is available, use it. If not, use the known baseline:
```
libopenal1, libssl3, libcurl4, libuuid1, pulseaudio (dummy sink),
xvfb, x11-utils, libxrandr2, libxinerama1, libxi6, libxxf86vm1
```

### Task 2 — Dockerfile
Create `Dockerfile` at the repo root. The build must:

**Stage 1 — Builder:**
```
FROM ubuntu:22.04 AS builder
- Install Node.js 20+ (for asset extraction scripts if needed)
- Install GameMaker Igor CLI (research the correct download/install path)
- Copy the GML project source
- Accept GAMEMAKER_LICENSE as a build secret
- Run Igor to compile the Ubuntu target
- Output: the compiled game binary and its required runtime files
```

**Stage 2 — Runtime:**
```
FROM ubuntu:22.04 AS runtime
- Install only runtime dependencies (no build tools)
- Install Xvfb and minimal X11 libraries
- Install PulseAudio in dummy/headless mode (GameMaker requires audio init)
- Copy compiled binary from builder stage
- Expose port 7777
- Set ENTRYPOINT to the launch script (Task 3)
```

The multi-stage approach keeps the runtime image lean. Builder can be large; runtime should be minimal.

Key Dockerfile requirements:
- Do NOT embed the GameMaker license — use `RUN --mount=type=secret,id=gm_license`
- Use `ARG AGENT_PORT=7777` so the port is configurable
- Use `ARG GM_SPEED=500` if game speed needs to be set at build time (though it's controlled by GML flag)
- Set `ENV DISPLAY=:99` as default, overridable per container

### Task 3 — Container Launch Script
Create `scripts/entrypoint.sh`:

```bash
#!/bin/bash
set -e

DISPLAY_NUM=${DISPLAY_NUM:-99}
export DISPLAY=:${DISPLAY_NUM}

# Start Xvfb on the assigned display
Xvfb :${DISPLAY_NUM} -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to be ready
sleep 1
xdpyinfo -display :${DISPLAY_NUM} > /dev/null 2>&1 || {
    echo "ERROR: Xvfb failed to start on display :${DISPLAY_NUM}"
    exit 1
}

# Start PulseAudio dummy sink (required for GameMaker audio init)
pulseaudio --start --log-target=syslog --daemon 2>/dev/null || true

# Launch the game with agent mode (agent_mode flag is set in GML source)
echo "Starting Nuclear Throne on display :${DISPLAY_NUM}, socket port 7777"
./NuclearThrone &
GAME_PID=$!

# Health check loop — restart game on crash during training
while true; do
    if ! kill -0 $GAME_PID 2>/dev/null; then
        echo "Game process died, restarting..."
        ./NuclearThrone &
        GAME_PID=$!
    fi
    sleep 5
done
```

The restart loop is essential — during RL training the game process may crash on edge cases. The Python environment should handle socket reconnection.

### Task 4 — docker-compose for Parallel Environments
Create `docker-compose.yml` supporting N parallel instances. Use a template that generates instances programmatically.

Each instance needs:
- A unique `DISPLAY_NUM` (99, 100, 101, ...)
- A unique host port mapping for the socket (7777, 7778, 7779, ...)
- Its own named volume for any persistent state
- A health check that verifies the socket is listening

```yaml
version: '3.8'

x-nt-agent: &nt-agent-base
  build:
    context: .
    secrets:
      - gm_license
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "nc", "-z", "localhost", "7777"]
    interval: 10s
    timeout: 5s
    retries: 3

services:
  nt-agent-1:
    <<: *nt-agent-base
    environment:
      DISPLAY_NUM: "99"
    ports:
      - "7777:7777"

  nt-agent-2:
    <<: *nt-agent-base
    environment:
      DISPLAY_NUM: "100"
    ports:
      - "7778:7777"

  # ... extend to desired N

secrets:
  gm_license:
    file: ./gamemaker-license.pem
```

Also produce a helper script `scripts/scale.sh N` that generates a docker-compose override file for exactly N instances.

### Task 5 — Build Verification Script
Create `scripts/verify_build.sh`:

This script must be runnable by Agent 04 (and humans) to confirm the infrastructure is working before any ML code is written. It should:

1. Build the Docker image: `docker build -t nt-agent .`
2. Start a single container: `docker run -d -p 7777:7777 --name nt-test nt-agent`
3. Wait up to 30 seconds for the socket to become available: `nc -z localhost 7777`
4. If socket is available: print "BUILD VERIFIED — socket bridge is live" and exit 0
5. If timeout: print container logs, print "BUILD FAILED", exit 1
6. Clean up: `docker stop nt-test && docker rm nt-test`

### Task 6 — Documentation
Create `INFRA_README.md` covering:
- Prerequisites (Docker version, GameMaker license format, asset population)
- How to build: single command
- How to run 1 instance vs N instances
- How to view Xvfb output for debugging (hint: `x11vnc` or `ffmpeg` can attach to the Xvfb display)
- Known issues and workarounds, especially around GameMaker ARM64 compatibility
- How to update the game binary after GML changes (rebuild trigger)
- Port mapping table for N parallel instances

## Output Artifacts
```
nt-recreated-public/
├── Dockerfile
├── docker-compose.yml
├── INFRA_README.md
└── scripts/
    ├── entrypoint.sh
    ├── scale.sh
    └── verify_build.sh
```

## Completion Criteria
You are done when:
- `Dockerfile` builds without error on ARM64 Ubuntu base
- `entrypoint.sh` starts Xvfb, then the game, and includes the crash-restart loop
- `docker-compose.yml` supports at least 4 parallel instances with correct port mapping
- `verify_build.sh` exits 0 when the bridge socket becomes reachable
- `INFRA_README.md` is complete

## Failure Handling

### GameMaker Igor Build Failures
Igor build errors are often cryptic. If the build fails:
1. Capture the full Igor stdout/stderr
2. Search for the specific error string: "YoYo Games GameMaker Igor [error text]"
3. Try documented fixes in order (license path, runtime version mismatch, missing dependency)
4. If unresolved after 3 attempts, write the error verbatim to `BUILD_ERRORS.md` and halt — this requires human intervention

### ARM64 Compatibility
GameMaker's Ubuntu runner may not have an ARM64 native binary. If the runner is x86-64 only:
- Add `--platform linux/amd64` to the FROM statement
- Add Rosetta/QEMU emulation note to `INFRA_README.md`
- Test that the emulated binary still runs under Xvfb

### Audio Failures
GameMaker initializes audio at startup and may fail silently or crash without an audio device. If the game crashes immediately:
- Verify PulseAudio dummy sink is running before game launch
- Try `padsp` wrapper: `padsp ./NuclearThrone`
- As last resort: set `SDL_AUDIODRIVER=dummy` environment variable

## Do Not
- Hardcode any license keys, tokens, or credentials anywhere in any file
- Include game assets in the Docker image — they come from the host volume
- Use `--privileged` Docker mode — this is unnecessary and a security smell
- Use host networking — explicit port mapping is required for parallelism
