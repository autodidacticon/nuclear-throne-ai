#!/bin/bash
set -euo pipefail

IGOR="/Users/Shared/GameMakerStudio2/Cache/runtimes/runtime-2024.14.4.268/bin/igor/osx/arm64/Igor"
RUNTIME="/Users/Shared/GameMakerStudio2/Cache/runtimes/runtime-2024.14.4.268"
USER_DIR="$HOME/Library/Application Support/GameMakerStudio2/richard.moorhead_5037433"
PROJECT="/Users/richard/git/nuclear-throne-ai/nt-recreated-public/nuclearthronemobile.yyp"
CACHE="/tmp/nt_build_cache"
TEMP="/tmp/nt_build_temp"

COMMON_ARGS=(
  --project="$PROJECT"
  --rp="$RUNTIME"
  --uf="$USER_DIR"
  --cache="$CACHE"
)

case "${1:-help}" in
  clean)
    echo "==> Cleaning build cache..."
    "$IGOR" "${COMMON_ARGS[@]}" mac Clean
    ;;
  clean-code)
    echo "==> Cleaning code cache..."
    "$IGOR" "${COMMON_ARGS[@]}" mac CleanCode
    ;;
  clean-graphics)
    echo "==> Cleaning graphics cache..."
    "$IGOR" "${COMMON_ARGS[@]}" mac CleanGraphics
    ;;
  clean-audio)
    echo "==> Cleaning audio cache..."
    "$IGOR" "${COMMON_ARGS[@]}" mac CleanAudio
    ;;
  run)
    echo "==> Building and running (VM)..."
    "$IGOR" "${COMMON_ARGS[@]}" --temp="$TEMP" -j=8 --runtime=VM mac Run
    ;;
  rebuild)
    echo "==> Clean + Build + Run..."
    "$IGOR" "${COMMON_ARGS[@]}" mac Clean
    "$IGOR" "${COMMON_ARGS[@]}" --temp="$TEMP" -j=8 --runtime=VM mac Run
    ;;
  stop)
    echo "==> Stopping game (Igor)..."
    "$IGOR" "${COMMON_ARGS[@]}" mac Stop
    ;;
  kill)
    echo "==> Killing Mac_Runner (game) and related processes..."
    pkill -f Mac_Runner && echo "Killed game." || echo "No running game instance found."
    pkill -f "tail.*nuclearthronemobile.*debug.log" 2>/dev/null || true
    ;;
  package)
    echo "==> Packaging as zip..."
    "$IGOR" "${COMMON_ARGS[@]}" --temp="$TEMP" -j=8 --runtime=VM \
      --of=NuclearThrone --tf=NuclearThrone.zip mac PackageZip
    ;;
  *)
    echo "Usage: $0 {clean|clean-code|clean-graphics|clean-audio|run|stop|kill|rebuild|package}"
    echo ""
    echo "  clean          Full cache clean"
    echo "  clean-code     Clean compiled scripts only"
    echo "  clean-graphics Clean shaders, sprites, textures"
    echo "  clean-audio    Clean audio cache"
    echo "  run            Build (VM) and launch"
    echo "  stop           Stop the game (Igor-launched only)"
    echo "  kill           Kill all game processes (any launch method)"
    echo "  rebuild        Clean then build and launch"
    echo "  package        Build and package as zip"
    ;;
esac
