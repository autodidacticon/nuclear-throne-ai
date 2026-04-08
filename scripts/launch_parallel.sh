#!/bin/bash
# Launch N parallel Nuclear Throne game instances for PPO training.
# Usage: ./scripts/launch_parallel.sh [N_INSTANCES]
#
# Each instance runs on a separate UDP port (7777, 7778, ...).
# Requires the game to be built first: ./scripts/gm_build.sh run (then kill it).

set -euo pipefail

N=${1:-4}
BASE_PORT=7777
GAME_DIR="$(cd "$(dirname "$0")/.." && pwd)/output/nuclearthronemobile"
RUNNER="/Users/Shared/GameMakerStudio2/Cache/runtimes/runtime-2024.14.4.268/mac/YoYo Runner.app/Contents/MacOS/Mac_Runner"

if [ ! -f "$GAME_DIR/game.ios" ]; then
    echo "ERROR: Game not built. Run: ./scripts/gm_build.sh run  (then kill it)"
    exit 1
fi

# Kill any existing instances
pkill -f Mac_Runner 2>/dev/null && echo "Killed existing instances" && sleep 1 || true

echo "=== Launching $N parallel game instances ==="

for i in $(seq 0 $((N-1))); do
    PORT=$((BASE_PORT + i))
    INSTANCE_DIR="/tmp/nt_instance_${PORT}"

    # Full copy of game directory (APFS clone is instant)
    rm -rf "$INSTANCE_DIR"
    cp -Rc "$GAME_DIR" "$INSTANCE_DIR" 2>/dev/null || cp -R "$GAME_DIR" "$INSTANCE_DIR"

    # Create agent mode and port files
    touch "$INSTANCE_DIR/agent_mode.txt"
    echo "$PORT" > "$INSTANCE_DIR/agent_port.txt"

    # Launch the runner — must include -debugoutput or the runner hangs
    cd "$INSTANCE_DIR"
    "$RUNNER" -game "$INSTANCE_DIR/game.ios" -debugoutput "$INSTANCE_DIR/debug.log" &>/dev/null &
    cd - >/dev/null

    echo "  Instance $i: port $PORT, pid $!, dir $INSTANCE_DIR"
done

echo ""
echo "Waiting for instances to start..."
sleep 5

# Verify all ports
OK=0
for i in $(seq 0 $((N-1))); do
    PORT=$((BASE_PORT + i))
    if lsof -i :$PORT -sTCP:LISTEN 2>/dev/null | grep -q Mac_Runner || \
       lsof -i UDP:$PORT 2>/dev/null | grep -q Mac_Runner; then
        echo "  Port $PORT: OK"
        OK=$((OK + 1))
    else
        echo "  Port $PORT: NOT READY (may need more time)"
    fi
done

echo ""
echo "$OK/$N instances ready."
echo ""
echo "To train:"
echo "  .venv/bin/python3 scripts/ppo_single.py --n-envs $N --timesteps 1000000"
echo ""
echo "To kill all:"
echo "  pkill -f Mac_Runner"
