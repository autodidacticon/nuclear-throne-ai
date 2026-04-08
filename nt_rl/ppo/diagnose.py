"""Pathology detection and diagnostic reporting for PPO training cycles.

Analyses evaluation metrics and training logs to detect five failure modes
(idle farming, corner hiding, death loop, reward plateau, action collapse)
and recommends reward configuration adjustments for the next cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds (from spec Task 4)
# ---------------------------------------------------------------------------

# Pathology 1 — Idle Farming
IDLE_FARMING_KILL_RATE = 2.0  # kills per step
IDLE_FARMING_EPISODE_FRACTION = 0.10  # >10% of episodes

# Pathology 2 — Corner Hiding
CORNER_HIDING_MIN_LENGTH = 1000
CORNER_HIDING_MAX_KILLS = 5
CORNER_HIDING_MAX_LEVELS = 1.5

# Pathology 3 — Death Loop
DEATH_LOOP_MAX_LENGTH = 50
DEATH_LOOP_MIN_STEPS_TRAINED = 500_000

# Pathology 4 — Reward Plateau
PLATEAU_WINDOW_STEPS = 1_000_000
PLATEAU_SLOPE_THRESHOLD = 0.0001  # per 10k steps

# Pathology 5 — Action Collapse
ACTION_COLLAPSE_MIN_ENTROPY = 0.2
ACTION_COLLAPSE_N_SAMPLES = 500

# Success criteria
MIN_VIABLE_LEVELS = 3.0
MIN_VIABLE_LENGTH = 300
TARGET_THRONE_RATE = 0.10
TARGET_KILLS = 150


# ---------------------------------------------------------------------------
# DiagnosticReport
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticReport:
    """Structured diagnosis of a PPO training cycle."""

    cycle: int
    steps_trained: int
    eval_metrics: dict                          # Final eval metrics at end of cycle
    pathologies_detected: list[str] = field(default_factory=list)
    convergence_verdict: str = "TRAINING"       # CONVERGED | PLATEAU | DEGRADED | TRAINING
    recommended_action: str = "CONTINUE"        # CONTINUE | ADJUST_REWARDS | RESTART_FROM_BC | DONE
    reward_adjustments: dict = field(default_factory=dict)
    notes: str = ""

    # Per-episode data used by detectors (optional, not always available)
    episode_kill_rates: list[float] | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_eval_results(
        cls,
        eval_metrics: dict,
        training_log: list[dict] | None = None,
        cycle: int = 1,
        steps_trained: int = 0,
        per_episode_data: list[dict] | None = None,
        action_entropies: list[float] | None = None,
    ) -> DiagnosticReport:
        """Build a DiagnosticReport from evaluation data and training logs.

        Args:
            eval_metrics: Aggregated evaluation metrics — must contain at
                minimum ``mean_reward``, ``mean_length``. Optionally
                ``mean_kills``, ``mean_levels_reached``, ``std_reward``,
                ``std_length``, ``throne_rate``.
            training_log: List of dicts with at least ``step`` and
                ``mean_reward`` keys, ordered by step. Used for plateau
                detection.
            cycle: The reward-shaping cycle number.
            steps_trained: Total environment steps completed in this cycle.
            per_episode_data: Optional list of per-episode dicts with keys
                ``kills`` and ``length`` — enables idle-farming detection.
            action_entropies: Optional list of per-action-dimension entropy
                values (one float per dimension). Enables action-collapse
                detection.

        Returns:
            A fully populated DiagnosticReport.
        """
        report = cls(
            cycle=cycle,
            steps_trained=steps_trained,
            eval_metrics=dict(eval_metrics),
        )

        pathologies: list[str] = []
        notes_parts: list[str] = []

        # --- Pathology 1: Idle Farming ---
        idle_result = _detect_idle_farming(eval_metrics, per_episode_data)
        if idle_result["detected"]:
            pathologies.append("idle_farming")
            notes_parts.append(idle_result["note"])

        # --- Pathology 2: Corner Hiding ---
        corner_result = _detect_corner_hiding(eval_metrics)
        if corner_result["detected"]:
            pathologies.append("corner_hiding")
            notes_parts.append(corner_result["note"])

        # --- Pathology 3: Death Loop ---
        death_result = _detect_death_loop(eval_metrics, steps_trained)
        if death_result["detected"]:
            pathologies.append("death_loop")
            notes_parts.append(death_result["note"])

        # --- Pathology 4: Reward Plateau ---
        plateau_result = _detect_reward_plateau(training_log)
        if plateau_result["detected"]:
            pathologies.append("reward_plateau")
            notes_parts.append(plateau_result["note"])

        # --- Pathology 5: Action Collapse ---
        collapse_result = _detect_action_collapse(action_entropies)
        if collapse_result["detected"]:
            pathologies.append("action_collapse")
            notes_parts.append(collapse_result["note"])

        report.pathologies_detected = pathologies
        report.notes = " | ".join(notes_parts) if notes_parts else "No pathologies detected."

        # --- Convergence verdict ---
        report.convergence_verdict = _compute_verdict(eval_metrics, pathologies)

        # --- Recommended action ---
        report.recommended_action = _compute_recommended_action(
            eval_metrics, pathologies, report.convergence_verdict
        )

        # --- Reward adjustments ---
        if report.recommended_action in ("ADJUST_REWARDS", "CONTINUE") and pathologies:
            report.reward_adjustments = suggest_reward_adjustments(report)

        return report


# ---------------------------------------------------------------------------
# Individual pathology detectors
# ---------------------------------------------------------------------------


def _detect_idle_farming(
    eval_metrics: dict,
    per_episode_data: list[dict] | None,
) -> dict[str, Any]:
    """Pathology 1: High kill rate per step with short episodes."""
    if per_episode_data:
        flagged = 0
        for ep in per_episode_data:
            length = ep.get("length", 1)
            kills = ep.get("kills", 0)
            if length > 0 and kills / length > IDLE_FARMING_KILL_RATE:
                flagged += 1
        fraction = flagged / len(per_episode_data)
        if fraction > IDLE_FARMING_EPISODE_FRACTION:
            return {
                "detected": True,
                "note": (
                    f"Idle farming: {fraction:.0%} of episodes have kill rate "
                    f"> {IDLE_FARMING_KILL_RATE}/step"
                ),
            }
    else:
        # Fallback: use aggregated metrics
        mean_kills = eval_metrics.get("mean_kills", 0)
        mean_length = eval_metrics.get("mean_length", 1)
        if mean_length > 0 and mean_kills / mean_length > IDLE_FARMING_KILL_RATE:
            return {
                "detected": True,
                "note": (
                    f"Idle farming (aggregate): kill rate "
                    f"{mean_kills / mean_length:.2f}/step"
                ),
            }

    return {"detected": False, "note": ""}


def _detect_corner_hiding(eval_metrics: dict) -> dict[str, Any]:
    """Pathology 2: Long episodes, near-zero kills, no progression."""
    mean_length = eval_metrics.get("mean_length", 0)
    mean_kills = eval_metrics.get("mean_kills", 0)
    mean_levels = eval_metrics.get("mean_levels_reached", 0)

    if (
        mean_length > CORNER_HIDING_MIN_LENGTH
        and mean_kills < CORNER_HIDING_MAX_KILLS
        and mean_levels < CORNER_HIDING_MAX_LEVELS
    ):
        return {
            "detected": True,
            "note": (
                f"Corner hiding: length={mean_length:.0f}, "
                f"kills={mean_kills:.1f}, levels={mean_levels:.1f}"
            ),
        }
    return {"detected": False, "note": ""}


def _detect_death_loop(eval_metrics: dict, steps_trained: int) -> dict[str, Any]:
    """Pathology 3: Agent dies within first 50 steps after significant training."""
    mean_length = eval_metrics.get("mean_length", 0)

    if steps_trained >= DEATH_LOOP_MIN_STEPS_TRAINED and mean_length < DEATH_LOOP_MAX_LENGTH:
        return {
            "detected": True,
            "note": (
                f"Death loop: mean_length={mean_length:.0f} after "
                f"{steps_trained:,} steps trained"
            ),
        }
    return {"detected": False, "note": ""}


def _detect_reward_plateau(training_log: list[dict] | None) -> dict[str, Any]:
    """Pathology 4: Reward stops improving over last 1M steps.

    Computes linear regression slope of mean_reward over the last
    PLATEAU_WINDOW_STEPS. If slope < PLATEAU_SLOPE_THRESHOLD per 10k steps,
    flags as plateau.
    """
    if not training_log or len(training_log) < 3:
        return {"detected": False, "note": ""}

    # Extract (step, mean_reward) pairs
    points = []
    for entry in training_log:
        step = entry.get("step", entry.get("timestep", 0))
        reward = entry.get("mean_reward", entry.get("eval/mean_reward"))
        if step is not None and reward is not None:
            points.append((step, reward))

    if len(points) < 3:
        return {"detected": False, "note": ""}

    points.sort(key=lambda p: p[0])
    max_step = points[-1][0]
    window_start = max_step - PLATEAU_WINDOW_STEPS

    # Filter to last window
    recent = [(s, r) for s, r in points if s >= window_start]
    if len(recent) < 3:
        return {"detected": False, "note": ""}

    steps_arr = np.array([p[0] for p in recent], dtype=np.float64)
    rewards_arr = np.array([p[1] for p in recent], dtype=np.float64)

    # Normalize steps to units of 10k
    steps_10k = steps_arr / 10_000.0

    # Linear regression: slope
    if steps_10k[-1] - steps_10k[0] < 1e-9:
        return {"detected": False, "note": ""}

    slope, _ = np.polyfit(steps_10k, rewards_arr, 1)

    if slope < PLATEAU_SLOPE_THRESHOLD:
        return {
            "detected": True,
            "note": (
                f"Reward plateau: slope={slope:.6f}/10k steps over last "
                f"{PLATEAU_WINDOW_STEPS:,} steps (threshold={PLATEAU_SLOPE_THRESHOLD})"
            ),
        }
    return {"detected": False, "note": ""}


def _detect_action_collapse(action_entropies: list[float] | None) -> dict[str, Any]:
    """Pathology 5: Policy outputs near-identical actions for all observations.

    Expects a list of per-action-dimension entropy values.
    """
    if not action_entropies:
        return {"detected": False, "note": ""}

    dim_names = ["move_dir", "aim_bin", "shoot", "special"]
    collapsed_dims = []

    for i, entropy in enumerate(action_entropies):
        dim_name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
        if entropy < ACTION_COLLAPSE_MIN_ENTROPY:
            collapsed_dims.append(f"{dim_name}(H={entropy:.3f})")

    if collapsed_dims:
        return {
            "detected": True,
            "note": f"Action collapse in: {', '.join(collapsed_dims)}",
        }
    return {"detected": False, "note": ""}


# ---------------------------------------------------------------------------
# Verdict & recommendation logic
# ---------------------------------------------------------------------------


def _compute_verdict(eval_metrics: dict, pathologies: list[str]) -> str:
    """Determine convergence verdict from metrics and detected pathologies."""
    mean_levels = eval_metrics.get("mean_levels_reached", 0)
    mean_length = eval_metrics.get("mean_length", 0)

    if mean_levels >= MIN_VIABLE_LEVELS and mean_length >= MIN_VIABLE_LENGTH:
        return "CONVERGED"

    if "death_loop" in pathologies:
        return "DEGRADED"

    if "reward_plateau" in pathologies:
        return "PLATEAU"

    return "TRAINING"


def _compute_recommended_action(
    eval_metrics: dict,
    pathologies: list[str],
    verdict: str,
) -> str:
    """Decide what to do next based on verdict and pathologies."""
    if verdict == "CONVERGED":
        return "DONE"

    if "death_loop" in pathologies:
        return "RESTART_FROM_BC"

    if pathologies:
        return "ADJUST_REWARDS"

    if verdict == "PLATEAU":
        return "ADJUST_REWARDS"

    return "CONTINUE"


# ---------------------------------------------------------------------------
# Reward adjustment suggestions
# ---------------------------------------------------------------------------


def suggest_reward_adjustments(report: DiagnosticReport) -> dict:
    """Recommend RewardConfig field changes based on detected pathologies.

    Returns a dict mapping RewardConfig field names to suggested new values.
    Applies the *first* applicable pathology fix only — the spec says not to
    compound multiple fixes in a single cycle.
    """
    adjustments: dict[str, Any] = {}

    for pathology in report.pathologies_detected:
        if pathology == "idle_farming":
            adjustments["reward_idle_penalty"] = -0.3
            adjustments["reward_kill"] = max(
                1.0,
                report.eval_metrics.get("_current_reward_kill", 5.0) * 0.5,
            )
            adjustments["reward_level_complete"] = 15.0
            adjustments["_reason"] = (
                "idle_farming: increased idle penalty, halved kill reward, "
                "boosted level completion reward"
            )
            break

        if pathology == "corner_hiding":
            adjustments["reward_survival_per_step"] = -0.005
            adjustments["reward_idle_penalty"] = -0.3
            adjustments["reward_idle_penalty_threshold"] = 60
            adjustments["_reason"] = (
                "corner_hiding: made survival per-step negative, increased "
                "idle penalty, lowered idle threshold"
            )
            break

        if pathology == "death_loop":
            # This case is handled by RESTART_FROM_BC; include LR guidance.
            adjustments["_lr_multiplier"] = 0.2
            adjustments["_reason"] = (
                "death_loop: BC warm-start destroyed. Restart from BC "
                "checkpoint with 5x lower learning rate."
            )
            break

        if pathology == "reward_plateau":
            adjustments["_ent_coef"] = 0.02
            adjustments["_gamma"] = 0.97
            adjustments["_reason"] = (
                "reward_plateau: increase entropy coeff to 0.02, reduce "
                "gamma to 0.97 to encourage shorter-horizon exploration"
            )
            break

        if pathology == "action_collapse":
            adjustments["_ent_coef"] = 0.05
            adjustments["_lr_multiplier"] = 0.5
            adjustments["_reason"] = (
                "action_collapse: significantly increase entropy coeff, "
                "halve learning rate"
            )
            break

    return adjustments


# ---------------------------------------------------------------------------
# Utility: compute action entropies from a policy and observations
# ---------------------------------------------------------------------------


def compute_action_entropies(policy, obs_samples: np.ndarray) -> list[float]:
    """Sample observations through a policy and return per-dimension entropy.

    Args:
        policy: SB3-compatible policy with ``predict(obs, deterministic=True)``.
        obs_samples: Array of shape ``(N, obs_dim)``.

    Returns:
        List of entropy values, one per action dimension
        (move_dir, aim_bin, shoot, special).
    """
    n = len(obs_samples)
    dim_sizes = [9, 24, 2, 2]
    n_dims = len(dim_sizes)
    actions = np.zeros((n, n_dims), dtype=np.int32)

    for i in range(n):
        action, _ = policy.predict(obs_samples[i], deterministic=True)
        actions[i] = action

    entropies: list[float] = []
    for dim, size in enumerate(dim_sizes):
        counts = np.bincount(actions[:, dim], minlength=size).astype(np.float64)
        probs = counts / counts.sum()
        # Shannon entropy (base e); clip to avoid log(0)
        probs_clipped = np.clip(probs, 1e-10, 1.0)
        h = -float(np.sum(probs * np.log(probs_clipped)))
        entropies.append(h)

    return entropies


# ---------------------------------------------------------------------------
# Utility: check minimum viable criteria
# ---------------------------------------------------------------------------


def meets_minimum_viable_criteria(eval_metrics: dict) -> bool:
    """Return True if evaluation metrics satisfy minimum viable criteria."""
    mean_levels = eval_metrics.get("mean_levels_reached", 0)
    mean_length = eval_metrics.get("mean_length", 0)
    return mean_levels >= MIN_VIABLE_LEVELS and mean_length >= MIN_VIABLE_LENGTH


def meets_target_criteria(eval_metrics: dict) -> bool:
    """Return True if evaluation metrics satisfy target criteria."""
    throne_rate = eval_metrics.get("throne_rate", 0)
    mean_kills = eval_metrics.get("mean_kills", 0)
    return throne_rate >= TARGET_THRONE_RATE and mean_kills >= TARGET_KILLS
