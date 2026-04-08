"""Standardised evaluation protocol for PPO-trained Nuclear Throne agents.

Produces a definitive performance report with per-episode data and checks
against minimum viable and target criteria from the project spec.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Criteria thresholds (mirrors diagnose.py constants)
# ---------------------------------------------------------------------------

MIN_VIABLE_LEVELS = 3.0
MIN_VIABLE_LENGTH = 300
TARGET_THRONE_RATE = 0.10
TARGET_KILLS = 150


# ---------------------------------------------------------------------------
# Per-episode record
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    """Data recorded for a single evaluation episode."""

    episode: int
    reward: float
    length: int
    levels_reached: float = 0.0
    kills: int = 0
    death_cause: str = "unknown"
    reached_throne: bool = False


# ---------------------------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------------------------


@dataclass
class EvalSummary:
    """Aggregated evaluation results across all episodes."""

    checkpoint_path: str
    n_episodes: int
    timestamp: str = ""

    # Aggregated metrics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_length: float = 0.0
    std_length: float = 0.0
    mean_levels_reached: float = 0.0
    std_levels_reached: float = 0.0
    mean_kills: float = 0.0
    std_kills: float = 0.0
    throne_rate: float = 0.0
    throne_count: int = 0

    # Criteria checks
    min_viable_levels_ok: bool = False
    min_viable_length_ok: bool = False
    target_throne_ok: bool = False
    target_kills_ok: bool = False

    # Verdict
    verdict: str = ""
    recommendation: str = ""

    # Raw per-episode data
    episodes: list[dict] = field(default_factory=list)

    def to_metrics_dict(self) -> dict:
        """Return a flat dict of metrics for use with DiagnosticReport."""
        return {
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_length": self.mean_length,
            "std_length": self.std_length,
            "mean_levels_reached": self.mean_levels_reached,
            "std_levels_reached": self.std_levels_reached,
            "mean_kills": self.mean_kills,
            "std_kills": self.std_kills,
            "throne_rate": self.throne_rate,
            "throne_count": self.throne_count,
        }


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def run_final_evaluation(
    checkpoint_path: str,
    env,
    n_episodes: int = 50,
    deterministic: bool = True,
    save_dir: str | None = None,
) -> EvalSummary:
    """Run the standardised evaluation protocol.

    Args:
        checkpoint_path: Path to the SB3 model checkpoint (.zip).
        env: A Gymnasium-compatible environment instance (NuclearThroneEnv or
            similar). The caller is responsible for creating and closing it.
        n_episodes: Number of evaluation episodes to run (default 50).
        deterministic: Whether to use deterministic policy (default True).
        save_dir: Directory to save ``final_eval.json``. If None, saves
            alongside the checkpoint.

    Returns:
        An EvalSummary containing all per-episode data and aggregate metrics.
    """
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint_path)
    return _evaluate_model(
        model, env, n_episodes, deterministic, checkpoint_path, save_dir
    )


def evaluate_model(
    model,
    env,
    n_episodes: int = 50,
    deterministic: bool = True,
    checkpoint_path: str = "<in-memory>",
    save_dir: str | None = None,
) -> EvalSummary:
    """Evaluate an already-loaded model (avoids re-loading from disk).

    Same interface as ``run_final_evaluation`` but takes a model object.
    """
    return _evaluate_model(model, env, n_episodes, deterministic, checkpoint_path, save_dir)


def _evaluate_model(
    model,
    env,
    n_episodes: int,
    deterministic: bool,
    checkpoint_path: str,
    save_dir: str | None,
) -> EvalSummary:
    """Internal implementation shared by both public entry points."""
    episode_records: list[EpisodeRecord] = []

    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        # Extract game-specific info
        game_info = info.get("game", {})
        kills = int(game_info.get("kills", 0))
        levels = float(game_info.get("level", game_info.get("levels_reached", 0)))
        death_cause = str(game_info.get("death_cause", "unknown"))
        reached_throne = bool(game_info.get("reached_throne", False))

        # Heuristic: if level >= 14 (7 areas x 2 stages), treat as throne
        if levels >= 14:
            reached_throne = True

        record = EpisodeRecord(
            episode=ep_idx + 1,
            reward=ep_reward,
            length=ep_length,
            levels_reached=levels,
            kills=kills,
            death_cause=death_cause,
            reached_throne=reached_throne,
        )
        episode_records.append(record)

    summary = _build_summary(checkpoint_path, episode_records)

    # Save JSON
    if save_dir is None:
        save_dir = str(Path(checkpoint_path).parent) if checkpoint_path != "<in-memory>" else "."
    _save_eval_json(summary, save_dir)

    return summary


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _build_summary(checkpoint_path: str, records: list[EpisodeRecord]) -> EvalSummary:
    """Compute aggregate metrics and criteria checks from episode records."""
    n = len(records)
    rewards = np.array([r.reward for r in records])
    lengths = np.array([r.length for r in records])
    levels = np.array([r.levels_reached for r in records])
    kills_arr = np.array([r.kills for r in records])
    throne_count = sum(1 for r in records if r.reached_throne)

    summary = EvalSummary(
        checkpoint_path=checkpoint_path,
        n_episodes=n,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_length=float(np.mean(lengths)),
        std_length=float(np.std(lengths)),
        mean_levels_reached=float(np.mean(levels)),
        std_levels_reached=float(np.std(levels)),
        mean_kills=float(np.mean(kills_arr)),
        std_kills=float(np.std(kills_arr)),
        throne_rate=throne_count / n if n > 0 else 0.0,
        throne_count=throne_count,
        episodes=[asdict(r) for r in records],
    )

    # Criteria checks
    summary.min_viable_levels_ok = summary.mean_levels_reached >= MIN_VIABLE_LEVELS
    summary.min_viable_length_ok = summary.mean_length >= MIN_VIABLE_LENGTH
    summary.target_throne_ok = summary.throne_rate >= TARGET_THRONE_RATE
    summary.target_kills_ok = summary.mean_kills >= TARGET_KILLS

    # Verdict
    min_viable_met = summary.min_viable_levels_ok and summary.min_viable_length_ok
    target_met = summary.target_throne_ok and summary.target_kills_ok

    if target_met and min_viable_met:
        summary.verdict = "TARGET MET"
        summary.recommendation = "Training complete. Target criteria satisfied."
    elif min_viable_met:
        summary.verdict = "MINIMUM VIABLE MET -- TARGET NOT MET"
        summary.recommendation = "Continue training or accept current performance."
    else:
        summary.verdict = "MINIMUM VIABLE NOT MET"
        summary.recommendation = "Further training or reward adjustment required."

    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(summary: EvalSummary) -> str:
    """Produce the formatted evaluation summary report as a string."""

    def _check(ok: bool) -> str:
        return "[PASS]" if ok else "[FAIL]"

    lines = [
        "=== FINAL EVALUATION REPORT ===",
        f"Checkpoint: {summary.checkpoint_path}",
        f"Episodes: {summary.n_episodes}",
        f"Timestamp: {summary.timestamp}",
        "",
        "Performance:",
        f"  Mean reward:           {summary.mean_reward:.1f}  "
        f"(+/-{summary.std_reward:.1f})",
        f"  Mean episode length:   {summary.mean_length:.0f} steps  "
        f"(+/-{summary.std_length:.0f})",
        f"  Mean levels reached:   {summary.mean_levels_reached:.1f}  "
        f"(+/-{summary.std_levels_reached:.1f})",
        f"  Mean kills:            {summary.mean_kills:.1f}  "
        f"(+/-{summary.std_kills:.1f})",
        f"  Nuclear Throne rate:   {summary.throne_rate:.0%}  "
        f"({summary.throne_count}/{summary.n_episodes} episodes)",
        "",
        "Minimum Viable Criteria:",
        f"  {_check(summary.min_viable_levels_ok)} Mean levels reached > {MIN_VIABLE_LEVELS}  "
        f"[{summary.mean_levels_reached:.1f}]",
        f"  {_check(summary.min_viable_length_ok)} Mean episode length > {MIN_VIABLE_LENGTH}  "
        f"[{summary.mean_length:.0f}]",
        "",
        "Target Criteria:",
        f"  {_check(summary.target_throne_ok)} Nuclear Throne rate >= {TARGET_THRONE_RATE:.0%}  "
        f"[{summary.throne_rate:.0%}]",
        f"  {_check(summary.target_kills_ok)} Mean kills > {TARGET_KILLS}  "
        f"[{summary.mean_kills:.1f}]",
        "",
        f"Verdict: {summary.verdict}",
        f"Recommendation: {summary.recommendation}",
    ]

    return "\n".join(lines)


def print_report(summary: EvalSummary) -> None:
    """Print the formatted evaluation report to stdout."""
    print(format_report(summary))


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def _save_eval_json(summary: EvalSummary, save_dir: str) -> str:
    """Save the full evaluation data to final_eval.json.

    Returns the path to the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "final_eval.json")

    data = {
        "checkpoint_path": summary.checkpoint_path,
        "n_episodes": summary.n_episodes,
        "timestamp": summary.timestamp,
        "metrics": {
            "mean_reward": summary.mean_reward,
            "std_reward": summary.std_reward,
            "mean_length": summary.mean_length,
            "std_length": summary.std_length,
            "mean_levels_reached": summary.mean_levels_reached,
            "std_levels_reached": summary.std_levels_reached,
            "mean_kills": summary.mean_kills,
            "std_kills": summary.std_kills,
            "throne_rate": summary.throne_rate,
            "throne_count": summary.throne_count,
        },
        "criteria": {
            "min_viable_levels_ok": summary.min_viable_levels_ok,
            "min_viable_length_ok": summary.min_viable_length_ok,
            "target_throne_ok": summary.target_throne_ok,
            "target_kills_ok": summary.target_kills_ok,
        },
        "verdict": summary.verdict,
        "recommendation": summary.recommendation,
        "episodes": summary.episodes,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_eval_json(path: str) -> EvalSummary:
    """Load an EvalSummary from a previously saved final_eval.json."""
    with open(path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    criteria = data.get("criteria", {})

    summary = EvalSummary(
        checkpoint_path=data.get("checkpoint_path", ""),
        n_episodes=data.get("n_episodes", 0),
        timestamp=data.get("timestamp", ""),
        mean_reward=metrics.get("mean_reward", 0.0),
        std_reward=metrics.get("std_reward", 0.0),
        mean_length=metrics.get("mean_length", 0.0),
        std_length=metrics.get("std_length", 0.0),
        mean_levels_reached=metrics.get("mean_levels_reached", 0.0),
        std_levels_reached=metrics.get("std_levels_reached", 0.0),
        mean_kills=metrics.get("mean_kills", 0.0),
        std_kills=metrics.get("std_kills", 0.0),
        throne_rate=metrics.get("throne_rate", 0.0),
        throne_count=metrics.get("throne_count", 0),
        episodes=data.get("episodes", []),
        min_viable_levels_ok=criteria.get("min_viable_levels_ok", False),
        min_viable_length_ok=criteria.get("min_viable_length_ok", False),
        target_throne_ok=criteria.get("target_throne_ok", False),
        target_kills_ok=criteria.get("target_kills_ok", False),
        verdict=data.get("verdict", ""),
        recommendation=data.get("recommendation", ""),
    )

    return summary
