"""Evaluate a trained PPO checkpoint against a live Nuclear Throne instance.

Runs N episodes in (deterministic or stochastic) mode and reports aggregate
statistics plus diagnostics specifically useful for the "is the agent actually
moving?" question:

  - Kills / levels / episode length / reward
  - Action distribution over the whole eval
  - Mean net displacement per episode (are directional moves cancelling out?)
  - Mean total path length per episode (how much did the agent actually move?)
  - Displacement efficiency (net / total)

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint checkpoints/ppo/v4_cycle2/ppo_8790000_steps.zip \
        --episodes 10 \
        --host 127.0.0.1 --port 7777
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from nt_rl.config import EnvConfig
from nt_rl.env import NuclearThroneEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to PPO .zip checkpoint")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)",
    )
    p.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic sampling instead",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Safety cap per episode (prevents infinite runs if death detection fails)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: alongside checkpoint as eval_report.json)",
    )
    return p.parse_args()


def run_one_episode(
    model,
    env: NuclearThroneEnv,
    config: EnvConfig,
    deterministic: bool,
    max_steps: int,
    ep_idx: int,
) -> dict:
    """Run a single episode and return a dict of per-episode metrics."""
    obs, info = env.reset()

    # Initial player position (normalized, indices 0-1 in obs)
    prev_x = float(obs[0]) * config.room_width
    prev_y = float(obs[1]) * config.room_height
    start_x, start_y = prev_x, prev_y

    ep_reward = 0.0
    ep_length = 0
    total_path_length = 0.0

    # Action distribution counters
    move_counts = np.zeros(9, dtype=np.int64)  # 0-7 dirs + none
    shoot_count = 0
    special_count = 0

    # Game metrics (updated each step from info)
    max_area = 0.0
    max_level = 0.0
    max_kills = 0
    last_info: dict = {}

    done = False
    while not done and ep_length < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track action distribution
        move_counts[int(action[0])] += 1
        if int(action[2]) == 1:
            shoot_count += 1
        if int(action[3]) == 1:
            special_count += 1

        # Track displacement
        cur_x = float(obs[0]) * config.room_width
        cur_y = float(obs[1]) * config.room_height
        dx = cur_x - prev_x
        dy = cur_y - prev_y
        step_dist = (dx * dx + dy * dy) ** 0.5
        # Ignore teleport-scale jumps (area transitions move the player)
        if step_dist < 200.0:
            total_path_length += step_dist
        prev_x, prev_y = cur_x, cur_y

        # Track game progress
        game = info.get("game", {}) or {}
        area = float(game.get("area", 0) or 0)
        level = float(game.get("level", 0) or 0)
        kills = int(game.get("kills", 0) or 0)
        if area > max_area:
            max_area = area
        if level > max_level:
            max_level = level
        if kills > max_kills:
            max_kills = kills

        ep_reward += float(reward)
        ep_length += 1
        last_info = info

    # Net displacement (straight-line distance start → end)
    net_dx = prev_x - start_x
    net_dy = prev_y - start_y
    net_disp = (net_dx * net_dx + net_dy * net_dy) ** 0.5

    # Displacement efficiency: 1.0 = perfectly straight travel; 0.0 = moved in circles
    efficiency = net_disp / total_path_length if total_path_length > 1.0 else 0.0

    # Movement split
    total_actions = move_counts.sum()
    none_pct = move_counts[8] / max(total_actions, 1)
    shoot_pct = shoot_count / max(total_actions, 1)
    special_pct = special_count / max(total_actions, 1)

    return {
        "episode": ep_idx + 1,
        "length": ep_length,
        "reward": round(ep_reward, 2),
        "kills": max_kills,
        "max_area": max_area,
        "max_level": max_level,
        "net_displacement_px": round(net_disp, 1),
        "total_path_length_px": round(total_path_length, 1),
        "displacement_efficiency": round(efficiency, 3),
        "none_pct": round(none_pct * 100, 1),
        "shoot_pct": round(shoot_pct * 100, 1),
        "special_pct": round(special_pct * 100, 1),
        "move_counts": move_counts.tolist(),
        "terminated": bool(last_info.get("error") is None and done),
    }


def summarize(records: list[dict]) -> dict:
    """Compute aggregate statistics from per-episode records."""
    n = len(records)
    if n == 0:
        return {}

    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in records]))

    def _std(key: str) -> float:
        return float(np.std([r[key] for r in records]))

    # Aggregate move counts
    total_moves = np.zeros(9, dtype=np.int64)
    for r in records:
        total_moves += np.array(r["move_counts"], dtype=np.int64)
    total = total_moves.sum()
    agg_move_pct = (total_moves / max(total, 1)) * 100

    return {
        "n_episodes": n,
        "length": {"mean": _mean("length"), "std": _std("length")},
        "reward": {"mean": _mean("reward"), "std": _std("reward")},
        "kills": {"mean": _mean("kills"), "std": _std("kills")},
        "max_area": {"mean": _mean("max_area"), "std": _std("max_area")},
        "max_level": {"mean": _mean("max_level"), "std": _std("max_level")},
        "net_displacement_px": {
            "mean": _mean("net_displacement_px"),
            "std": _std("net_displacement_px"),
        },
        "total_path_length_px": {
            "mean": _mean("total_path_length_px"),
            "std": _std("total_path_length_px"),
        },
        "displacement_efficiency": {
            "mean": _mean("displacement_efficiency"),
            "std": _std("displacement_efficiency"),
        },
        "none_pct": _mean("none_pct"),
        "shoot_pct": _mean("shoot_pct"),
        "special_pct": _mean("special_pct"),
        "aggregate_move_distribution": {
            "E":    round(agg_move_pct[0], 1),
            "NE":   round(agg_move_pct[1], 1),
            "N":    round(agg_move_pct[2], 1),
            "NW":   round(agg_move_pct[3], 1),
            "W":    round(agg_move_pct[4], 1),
            "SW":   round(agg_move_pct[5], 1),
            "S":    round(agg_move_pct[6], 1),
            "SE":   round(agg_move_pct[7], 1),
            "none": round(agg_move_pct[8], 1),
        },
    }


def print_report(summary: dict, records: list[dict], checkpoint: str, det: bool) -> None:
    print("=" * 70)
    print("PPO EVALUATION REPORT")
    print("=" * 70)
    print(f"Checkpoint:    {checkpoint}")
    print(f"Episodes:      {summary.get('n_episodes', 0)}")
    print(f"Mode:          {'deterministic' if det else 'stochastic'}")
    print()

    if not summary:
        print("No episodes completed.")
        return

    def _fmt(d: dict) -> str:
        return f"{d['mean']:.1f} (+/- {d['std']:.1f})"

    print("Gameplay metrics:")
    print(f"  Episode length:      {_fmt(summary['length'])} steps")
    print(f"  Reward:              {_fmt(summary['reward'])}")
    print(f"  Kills:               {_fmt(summary['kills'])}")
    print(f"  Max area reached:    {_fmt(summary['max_area'])}")
    print(f"  Max level reached:   {_fmt(summary['max_level'])}")
    print()

    print("Movement diagnostics:")
    print(f"  Net displacement:    {_fmt(summary['net_displacement_px'])} px")
    print(f"  Total path length:   {_fmt(summary['total_path_length_px'])} px")
    print(f"  Displacement eff.:   {_fmt(summary['displacement_efficiency'])}")
    print("    (1.0 = straight travel, 0.0 = pacing in circles)")
    print()

    print("Action distribution (aggregated over all episodes):")
    md = summary["aggregate_move_distribution"]
    for dir_name in ("E", "NE", "N", "NW", "W", "SW", "S", "SE", "none"):
        print(f"  {dir_name:>4}: {md[dir_name]:>5.1f}%")
    print(f"  shoot:   {summary['shoot_pct']:.1f}%")
    print(f"  special: {summary['special_pct']:.1f}%")
    print()

    print("Per-episode breakdown:")
    hdr = f"{'Ep':>3} {'Len':>5} {'Rwd':>8} {'Kill':>5} {'Area':>5} {'Lvl':>5} "
    hdr += f"{'NetDisp':>9} {'PathLen':>9} {'Eff':>6} {'None%':>7}"
    print(hdr)
    for r in records:
        print(
            f"{r['episode']:>3} {r['length']:>5} {r['reward']:>8.1f} "
            f"{r['kills']:>5} {r['max_area']:>5.0f} {r['max_level']:>5.0f} "
            f"{r['net_displacement_px']:>9.0f} {r['total_path_length_px']:>9.0f} "
            f"{r['displacement_efficiency']:>6.3f} {r['none_pct']:>6.1f}%"
        )
    print()

    # Diagnosis
    mean_eff = summary["displacement_efficiency"]["mean"]
    mean_path = summary["total_path_length_px"]["mean"]
    mean_net = summary["net_displacement_px"]["mean"]

    print("Diagnosis:")
    if mean_path < 100:
        print("  [!] Agent barely moves at all (mean path < 100 px).")
        print("      The policy is literally immobile. Check obs / action wiring.")
    elif mean_eff < 0.1:
        print("  [!] Agent moves a lot but goes nowhere (efficiency < 0.1).")
        print("      Directional moves cancel out — no strategic direction signal.")
    elif mean_eff < 0.3:
        print("  [.] Agent wanders. Some progress but inefficient pathing.")
    else:
        print("  [OK] Agent travels purposefully (efficiency >= 0.3).")

    if summary["kills"]["mean"] < 1:
        print("  [!] Agent killed fewer than 1 enemy per episode on average.")
    if summary["max_level"]["mean"] < 1:
        print("  [!] Agent never advanced past the starting level.")


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Verify game is reachable
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(3)
        s.sendto(
            b'{"type":"action","move_dir":0,"moving":false,"aim_dir":0,'
            b'"fire":false,"spec":false}',
            (args.host, args.port),
        )
        s.recvfrom(65536)
        s.close()
    except Exception as e:
        print(f"ERROR: Cannot reach game at {args.host}:{args.port}: {e}", file=sys.stderr)
        print("Start the game with agent_mode.txt before running eval.")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = PPO.load(args.checkpoint, device="cpu")

    config = EnvConfig(host=args.host, base_port=args.port)
    env = NuclearThroneEnv(port=args.port, config=config)

    print(f"Running {args.episodes} episode(s) "
          f"({'deterministic' if args.deterministic else 'stochastic'})...")
    print()

    records: list[dict] = []
    t0 = time.time()
    try:
        for i in range(args.episodes):
            rec = run_one_episode(
                model, env, config, args.deterministic, args.max_steps, i
            )
            records.append(rec)
            print(f"  Ep {i+1}/{args.episodes}: len={rec['length']} "
                  f"reward={rec['reward']:.1f} kills={rec['kills']} "
                  f"area={int(rec['max_area'])} "
                  f"net_disp={int(rec['net_displacement_px'])}px "
                  f"path={int(rec['total_path_length_px'])}px "
                  f"eff={rec['displacement_efficiency']:.2f}")
    except KeyboardInterrupt:
        print("\nInterrupted — summarizing what we have...")
    finally:
        env.close()

    elapsed = time.time() - t0
    print()
    print(f"Completed {len(records)} episode(s) in {elapsed:.1f}s")
    print()

    summary = summarize(records)
    print_report(summary, records, args.checkpoint, args.deterministic)

    # Save JSON
    output = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "eval_report.json"
    )
    with open(output, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "deterministic": args.deterministic,
                "episodes": records,
                "summary": summary,
                "elapsed_seconds": round(elapsed, 1),
            },
            f,
            indent=2,
        )
    print()
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
