"""Policy evaluation utilities for behavioural cloning."""

import warnings

import numpy as np


def evaluate_policy(policy, env, n_episodes: int = 20) -> dict:
    """Run n_episodes with the policy and return evaluation metrics.

    Args:
        policy: SB3-compatible policy with predict(obs, deterministic=True).
        env: Gymnasium environment (NuclearThroneEnv or similar).
        n_episodes: Number of episodes to evaluate.

    Returns:
        Dict with mean_reward, std_reward, mean_length, episodes_completed,
        and optionally mean_kills and mean_levels_reached from info.
    """
    rewards, lengths, kills, levels = [], [], [], []

    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        lengths.append(ep_length)

        game_info = info.get("game", {})
        if "kills" in game_info:
            kills.append(game_info["kills"])
        if "level" in game_info:
            levels.append(game_info["level"])

    result = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "episodes_completed": n_episodes,
    }
    if kills:
        result["mean_kills"] = float(np.mean(kills))
    if levels:
        result["mean_levels_reached"] = float(np.mean(levels))

    if result["mean_length"] < 50:
        warnings.warn(
            f"Mean episode length is {result['mean_length']:.0f} steps — "
            "the policy may be dying immediately"
        )

    return result


def action_distribution_report(policy, obs_samples: np.ndarray) -> dict:
    """Sample observations through the policy and report action distribution.

    Args:
        policy: SB3-compatible policy with predict().
        obs_samples: Array of shape (N, obs_dim) to run through the policy.

    Returns:
        Dict mapping action dimension names to their value distributions.
    """
    n = len(obs_samples)
    actions = np.zeros((n, 4), dtype=np.int32)
    for i in range(n):
        action, _ = policy.predict(obs_samples[i], deterministic=True)
        actions[i] = action

    dim_names = ["move_dir", "aim_bin", "shoot", "special"]
    dim_sizes = [9, 24, 2, 2]
    report = {}

    for dim, (name, size) in enumerate(zip(dim_names, dim_sizes)):
        counts = np.bincount(actions[:, dim], minlength=size)
        pcts = counts / counts.sum() * 100
        report[name] = {
            "counts": counts.tolist(),
            "pcts": pcts.tolist(),
            "dominant_value": int(pcts.argmax()),
            "dominant_pct": float(pcts.max()),
        }
        if pcts.max() > 80:
            warnings.warn(
                f"Action collapse: {name} outputs value {pcts.argmax()} "
                f"for {pcts.max():.0f}% of samples"
            )

    return report
