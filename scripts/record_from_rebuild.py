"""Record human gameplay demonstrations from the rebuild's UDP bridge.

Place `agent_record.txt` (instead of `agent_mode.txt`) in the game's working
directory. This activates the bridge at normal game speed with human keyboard
control, and includes the human's actions in the state JSON.

This script listens for UDP state packets, writes them as JSONL chunks
(same format as the NTT recorder), ready for the BC converter pipeline.

Usage:
    # 1. Place agent_record.txt in the game directory
    touch output/nuclearthronemobile/agent_record.txt

    # 2. Build and run the game
    ./scripts/gm_build.sh run

    # 3. Start this recorder
    python scripts/record_from_rebuild.py --port 7777 --output nt-data

    # 4. Play Nuclear Throne normally. Press Ctrl-C to stop recording.

    # 5. Convert to training data
    python -m nt_rl.bc.ntt_converter --input nt-data --output demonstrations --validate -v
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Game host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=7777, help="Game UDP port (default: 7777)")
    p.add_argument("--output", default="nt-data", help="Output directory for .jsonl chunks")
    p.add_argument("--flush-interval", type=int, default=60, help="Frames per chunk file (default: 60)")
    p.add_argument("--session-id", default=None, help="Session ID (default: auto-generated)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    session_id = args.session_id or f"{int(time.time())}_{os.getpid()}"
    addr = (args.host, args.port)

    # Create UDP socket and send a registration packet
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    sock.settimeout(5.0)

    print(f"Rebuild Gameplay Recorder")
    print(f"  Listening on: {addr}")
    print(f"  Output dir:   {args.output}")
    print(f"  Session ID:   {session_id}")
    print(f"  Flush every:  {args.flush_interval} frames")
    print()
    print(f"Make sure the game is running with `agent_record.txt` in its working directory.")
    print(f"Press Ctrl-C to stop recording.")
    print()

    # Register with the game by sending a no-op action
    noop = json.dumps({
        "type": "action", "move_dir": 0, "moving": False,
        "aim_dir": 0, "fire": False, "spec": False,
    }).encode()
    sock.sendto(noop, addr)

    # Wait for first state packet
    print("Waiting for game to send state...")
    try:
        data, _ = sock.recvfrom(65536)
        state = json.loads(data.decode())
        has_human_action = "human_action" in state
        print(f"  Connected! human_action present: {has_human_action}")
        if not has_human_action:
            print()
            print("  WARNING: State does not include human_action.")
            print("  Make sure the game was started with `agent_record.txt`, not `agent_mode.txt`.")
            print("  Without human_action, BC training won't have action labels.")
            print()
    except socket.timeout:
        print("ERROR: No response from game. Is it running with agent_record.txt?")
        sys.exit(1)

    # Recording state
    episode = 0
    frame = 0
    part_num = 0
    buffer_lines: list[str] = []
    total_frames = 0
    prev_done = True  # Start in "dead" state to detect first episode
    running = True

    def _flush():
        nonlocal buffer_lines, part_num
        if not buffer_lines:
            return
        filename = f"ntt_demo_{session_id}_{episode:04d}_part{part_num:04d}.jsonl"
        filepath = os.path.join(args.output, filename)
        with open(filepath, "w") as f:
            f.write("\n".join(buffer_lines) + "\n")
        part_num += 1
        buffer_lines = []

    def _signal_handler(signum, _frame):
        nonlocal running
        print("\nStopping recorder...")
        running = False

    signal.signal(signal.SIGINT, _signal_handler)

    print()
    print("Recording...")

    while running:
        # Keep sending no-op to stay registered (game needs periodic packets)
        sock.sendto(noop, addr)

        try:
            sock.settimeout(1.0)
            data, _ = sock.recvfrom(65536)
        except socket.timeout:
            continue

        try:
            state = json.loads(data.decode())
        except json.JSONDecodeError:
            continue

        done = state.get("done", False)

        # Detect episode boundaries
        if prev_done and not done:
            # New episode starting
            _flush()
            episode += 1
            frame = 0
            part_num = 0
            print(f"  Episode {episode} started")

        prev_done = done

        # Skip frames where the player isn't alive (menu, death screen)
        if done:
            continue

        # Skip paused states (mutation screen)
        if state.get("mutation_screen", False):
            continue

        # Add frame number
        state["frame"] = frame

        # Write the state as a JSON line
        buffer_lines.append(json.dumps(state))
        frame += 1
        total_frames += 1

        # Periodic flush
        if len(buffer_lines) >= args.flush_interval:
            _flush()

        # Status update
        if total_frames % 300 == 0:
            player = state.get("player", {})
            game = state.get("game", {})
            hp = player.get("hp", 0)
            max_hp = player.get("max_hp", 0)
            area = game.get("area", 0)
            kills = game.get("kills", 0)
            n_enemies = len(state.get("enemies", []))
            n_projs = len(state.get("projectiles", []))
            print(f"    frame={total_frames} ep={episode} "
                  f"hp={hp}/{max_hp} area={area} kills={kills} "
                  f"enemies={n_enemies} projs={n_projs}")

    # Final flush
    _flush()

    sock.close()

    print()
    print(f"Recording complete.")
    print(f"  Episodes:     {episode}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Output dir:   {args.output}")
    print()
    print(f"Next step: convert to training data:")
    print(f"  rm -f demonstrations/*.npz")
    print(f"  python -m nt_rl.bc.ntt_converter --input {args.output} --output demonstrations --validate -v")


if __name__ == "__main__":
    main()
