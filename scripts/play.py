#!/usr/bin/env python3
"""Inference script: load a trained policy and play Nuclear Throne.

Usage:
    python3 scripts/play.py --checkpoint checkpoints/bc_run_c/final --stats
    python3 scripts/play.py --port 7777 --episodes 10 --deterministic --render-actions
"""

import argparse
import signal
import socket
import sys
import time

import gymnasium
import numpy as np

# Ensure project root is on the path when running as a script
sys.path.insert(0, ".")

from nt_rl.config import EnvConfig
from nt_rl.env import NuclearThroneEnv


# Human-readable labels for action dimensions
_MOVE_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE", "STOP"]
_AIM_BIN_DEG = 15  # degrees per aim bin


def load_policy(checkpoint_path: str, env_config: EnvConfig):
    """Load an SB3 ActorCriticPolicy from a checkpoint file.

    The BC training pipeline saves policies via policy.save(), which produces
    a zip file (possibly without the .zip extension). Load it back with
    ActorCriticPolicy.load().
    """
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy

    obs_space = gymnasium.spaces.Box(
        low=-1.0, high=1.0,
        shape=(env_config.obs_dim,),
        dtype=np.float32,
    )
    act_space = gymnasium.spaces.MultiDiscrete(
        [env_config.n_move_dirs, env_config.n_aim_angles, 2, 2]
    )

    # PyTorch 2.6+ defaults to weights_only=True in torch.load(), but SB3
    # policy checkpoints contain gymnasium space objects that require full
    # unpickling.  Temporarily override torch.load to allow this.
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        policy = ActorCriticPolicy.load(checkpoint_path)
    finally:
        torch.load = _orig_load

    policy.eval()

    # Verify spaces match
    if policy.observation_space.shape != obs_space.shape:
        print(
            f"WARNING: policy obs space {policy.observation_space.shape} "
            f"does not match expected {obs_space.shape}",
            file=sys.stderr,
        )

    return policy


def format_action(action) -> str:
    """Format a MultiDiscrete action as a human-readable string."""
    move_idx = int(action[0])
    aim_idx = int(action[1])
    shoot = bool(action[2])
    spec = bool(action[3])

    move = _MOVE_LABELS[move_idx]
    aim_deg = aim_idx * _AIM_BIN_DEG
    parts = [f"move={move}", f"aim={aim_deg:3d}deg"]
    if shoot:
        parts.append("FIRE")
    if spec:
        parts.append("SPEC")
    return " ".join(parts)


def check_connection(host: str, port: int, timeout: float = 2.0) -> bool:
    """Quick check if the game bridge is reachable on the given port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def run(args):
    env_config = EnvConfig()

    # --- Load policy ---
    print(f"Loading policy from: {args.checkpoint}")
    try:
        policy = load_policy(args.checkpoint, env_config)
    except Exception as e:
        print(f"ERROR: Failed to load policy: {e}", file=sys.stderr)
        print(
            f"\nMake sure the checkpoint exists at: {args.checkpoint}",
            file=sys.stderr,
        )
        print(
            "The policy should be saved with SB3's policy.save() (zip format).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Policy loaded. Network: {policy.net_arch}")

    # --- Pre-flight connection check ---
    print(f"\nChecking connection to game on localhost:{args.port}...")
    if not check_connection("localhost", args.port, timeout=2.0):
        print(
            f"ERROR: Cannot connect to localhost:{args.port}\n"
            "\n"
            "Make sure the game is running before starting this script:\n"
            "  - Rebuild: run the game in GameMaker with agent_mode.txt present\n"
            "  - Official NT: start ntt_bridge_adapter.py first, then launch the game\n"
            "\n"
            "See docs/playing-with-agent.md for full setup instructions.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Connection OK.")

    # --- Create environment ---
    env = NuclearThroneEnv(port=args.port, config=env_config)

    # --- Episode tracking ---
    episode_stats = []
    current_episode = 0
    total_steps = 0

    # --- Graceful Ctrl+C ---
    interrupted = False

    def _on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _on_interrupt)

    # --- Main loop ---
    print(f"\nStarting inference ({'deterministic' if args.deterministic else 'stochastic'} mode)")
    if args.episodes > 0:
        print(f"Will play {args.episodes} episode(s).")
    else:
        print("Playing indefinitely (Ctrl+C to stop).")
    print()

    try:
        while not interrupted:
            # Check episode limit
            if args.episodes > 0 and current_episode >= args.episodes:
                break

            current_episode += 1
            obs, info = env.reset()
            ep_reward = 0.0
            ep_length = 0
            ep_kills = 0
            ep_level = 0
            done = False

            if args.stats:
                print(f"--- Episode {current_episode} ---")

            while not done and not interrupted:
                action, _ = policy.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_reward += reward
                ep_length += 1
                total_steps += 1
                done = terminated or truncated

                # Track game stats from info
                game_info = info.get("game", {})
                if "kills" in game_info:
                    ep_kills = game_info["kills"]
                if "level" in game_info:
                    ep_level = game_info["level"]

                if args.render_actions:
                    print(
                        f"  step={ep_length:5d}  reward={reward:+6.2f}  "
                        f"{format_action(action)}"
                    )

                # Check for connection errors
                if "error" in info:
                    print(
                        f"Connection error during episode {current_episode}: "
                        f"{info['error']}",
                        file=sys.stderr,
                    )
                    break

            # Record episode stats
            ep_stats = {
                "episode": current_episode,
                "reward": ep_reward,
                "length": ep_length,
                "kills": ep_kills,
                "level": ep_level,
            }
            episode_stats.append(ep_stats)

            if args.stats:
                print(
                    f"  Result: reward={ep_reward:+.2f}  "
                    f"length={ep_length}  "
                    f"kills={ep_kills}  "
                    f"level={ep_level}"
                )
                print()

    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
    finally:
        env.close()

    # --- Summary ---
    print_summary(episode_stats, total_steps)


def print_summary(episode_stats: list[dict], total_steps: int):
    """Print cumulative statistics across all episodes."""
    n = len(episode_stats)
    if n == 0:
        print("\nNo episodes completed.")
        return

    rewards = [e["reward"] for e in episode_stats]
    lengths = [e["length"] for e in episode_stats]
    kills = [e["kills"] for e in episode_stats]
    levels = [e["level"] for e in episode_stats]

    # Find best episode by reward
    best_idx = int(np.argmax(rewards))
    best = episode_stats[best_idx]

    print("=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"  Episodes completed: {n}")
    print(f"  Total steps:        {total_steps}")
    print()
    print(f"  Mean reward:  {np.mean(rewards):+.2f}  (std: {np.std(rewards):.2f})")
    print(f"  Mean length:  {np.mean(lengths):.0f}")
    print(f"  Mean kills:   {np.mean(kills):.1f}")
    print(f"  Mean level:   {np.mean(levels):.1f}")
    print()
    print(
        f"  Best run: episode {best['episode']}  "
        f"reward={best['reward']:+.2f}  "
        f"kills={best['kills']}  "
        f"level={best['level']}"
    )
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained policy and play Nuclear Throne",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/play.py --stats\n"
            "  python3 scripts/play.py --checkpoint checkpoints/bc_run_c/final --episodes 5\n"
            "  python3 scripts/play.py --stochastic --render-actions\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bc_run_c/final",
        help="Path to SB3 policy checkpoint (default: checkpoints/bc_run_c/final)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7777,
        help="Game bridge TCP port (default: 7777)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to play (default: 0 = infinite)",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic action selection (default)",
    )
    mode_group.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action selection",
    )

    parser.add_argument(
        "--render-actions",
        action="store_true",
        help="Print each action to stdout (for debugging)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print per-episode statistics (reward, length, kills, levels)",
    )

    args = parser.parse_args()

    # Handle the mutually exclusive deterministic/stochastic flags
    if args.stochastic:
        args.deterministic = False

    run(args)


if __name__ == "__main__":
    main()
