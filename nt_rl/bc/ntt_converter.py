"""Convert NTT recording mod logs (.jsonl) to BC training data (.npz).

The NTT mod records human gameplay from the official Nuclear Throne game as
newline-delimited JSON files (one JSON object per frame). This converter
transforms those logs into the same .npz format that DemonstrationRecorder
produces from the rebuild's socket bridge, so the training pipeline can
consume both data sources interchangeably.

Usage:
    python -m nt_rl.bc.ntt_converter --input /path/to/ntt/logs --output demonstrations
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from nt_rl.config import EnvConfig
from nt_rl.obs_utils import encode_observation
from nt_rl.bc.recorder import discretize_action

logger = logging.getLogger(__name__)


def _map_variable_names(state: dict) -> dict:
    """Map NTT game variable names to the rebuild's variable names in-place.

    NTT uses official GameMaker variable names:
        my_health -> hp
        maxhealth -> max_hp
        type_id   -> hitid  (object_index used as hitid proxy)
    """
    player = state.get("player", {})
    if "my_health" in player:
        player["hp"] = player.pop("my_health")
    if "maxhealth" in player:
        player["max_hp"] = player.pop("maxhealth")

    for enemy in state.get("enemies", []):
        if "my_health" in enemy:
            enemy["hp"] = enemy.pop("my_health")
        if "maxhealth" in enemy:
            enemy["max_hp"] = enemy.pop("maxhealth")
        if "type_id" in enemy:
            enemy["hitid"] = enemy.pop("type_id")

    return state


def _compute_reward(signals: dict, player: dict, is_terminal: bool,
                    config: EnvConfig) -> float:
    """Compute reward from raw NTT reward signals using EnvConfig weights."""
    reward = config.reward_survival_per_step

    reward += signals.get("kills_this_frame", 0) * config.reward_kill
    reward += signals.get("damage_this_frame", 0) * config.reward_damage_taken

    if signals.get("healed_this_frame", False):
        max_hp = max(player.get("max_hp", 1), 1)
        hp = player.get("hp", max_hp)
        hp_ratio = hp / max_hp
        if hp_ratio < 0.5:
            reward += config.reward_health_pickup

    if signals.get("level_changed", False):
        reward += config.reward_level_complete

    if is_terminal:
        reward += config.reward_death

    return reward


def _detect_episode_boundaries(frames: list[dict]) -> list[list[dict]]:
    """Split a list of parsed frames into episodes.

    Episode boundaries are detected by:
    - frame counter resetting to a lower value
    - player health jumping from 0 back to maxhealth
    """
    if not frames:
        return []

    episodes: list[list[dict]] = []
    current_episode: list[dict] = [frames[0]]

    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]

        # Check frame counter reset
        prev_frame = prev.get("frame", 0)
        curr_frame = curr.get("frame", 0)
        frame_reset = curr_frame < prev_frame

        # Check health jump: previous hp was 0 and current hp equals maxhealth
        prev_hp = prev.get("player", {}).get("my_health", -1)
        curr_hp = curr.get("player", {}).get("my_health", -1)
        curr_maxhp = curr.get("player", {}).get("maxhealth", -1)
        health_jump = (prev_hp <= 0) and (curr_hp > 0) and (curr_hp == curr_maxhp)

        if frame_reset or health_jump:
            if current_episode:
                episodes.append(current_episode)
            current_episode = [curr]
        else:
            current_episode.append(curr)

    if current_episode:
        episodes.append(current_episode)

    return episodes


class NTTLogConverter:
    """Converts NTT recording mod .jsonl logs to .npz demonstration files."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()

    def convert_file(self, jsonl_path: str, output_dir: str = "demonstrations") -> list[str]:
        """Convert one .jsonl log file to one or more .npz episode files.

        Returns list of created file paths.
        """
        jsonl_path = str(jsonl_path)
        os.makedirs(output_dir, exist_ok=True)

        # Parse all frames from the file
        frames = self._parse_jsonl(jsonl_path)
        if not frames:
            logger.warning("No valid frames in %s — skipping", jsonl_path)
            return []

        # Detect episode boundaries within the file
        episodes = _detect_episode_boundaries(frames)
        logger.info("Found %d episode(s) in %s", len(episodes), jsonl_path)

        # Convert each episode
        created_files = []
        base_name = Path(jsonl_path).stem
        timestamp = int(time.time())

        for ep_idx, episode_frames in enumerate(episodes):
            result = self._convert_episode(episode_frames)
            if result is None:
                logger.warning("Episode %d in %s produced no valid data — skipping",
                               ep_idx, jsonl_path)
                continue

            obs, actions, rewards, dones = result

            filename = f"{base_name}_ep{ep_idx:04d}_{timestamp}.npz"
            filepath = os.path.join(output_dir, filename)

            np.savez_compressed(
                filepath,
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )
            created_files.append(filepath)
            logger.info("  Episode %d: %d frames, reward=%.1f -> %s",
                        ep_idx, len(obs), rewards.sum(), filename)

        return created_files

    def convert_directory(self, input_dir: str, output_dir: str = "demonstrations") -> int:
        """Convert all .jsonl files in a directory.

        Returns total number of episodes converted.
        """
        input_path = Path(input_dir)
        jsonl_files = sorted(input_path.glob("*.jsonl"))

        if not jsonl_files:
            logger.warning("No .jsonl files found in %s", input_dir)
            return 0

        total_episodes = 0
        for jsonl_file in jsonl_files:
            logger.info("Converting %s ...", jsonl_file.name)
            created = self.convert_file(str(jsonl_file), output_dir)
            total_episodes += len(created)

        return total_episodes

    def _parse_jsonl(self, jsonl_path: str) -> list[dict]:
        """Parse a .jsonl file, skipping malformed lines."""
        frames = []
        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                    frames.append(frame)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON at %s:%d: %s",
                                   jsonl_path, line_num, e)
        return frames

    def _convert_episode(self, frames: list[dict]) -> (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None):
        """Convert a list of raw NTT frames into numpy arrays.

        Returns (obs, actions, rewards, dones) or None if episode is empty.
        """
        if not frames:
            return None

        obs_list = []
        action_list = []
        reward_list = []

        for i, raw_frame in enumerate(frames):
            # Deep copy the relevant parts to avoid mutating the original
            frame = {
                "player": dict(raw_frame.get("player", {})),
                "enemies": [dict(e) for e in raw_frame.get("enemies", [])],
                "game": dict(raw_frame.get("game", {})),
            }

            # Map NTT variable names to rebuild names
            _map_variable_names(frame)

            # Encode observation
            obs = encode_observation(frame, self.config)
            obs_list.append(obs)

            # Discretize action
            human_action = raw_frame.get("human_action", {})
            action = discretize_action(human_action, self.config.n_aim_angles)
            action_list.append(action)

            # Compute reward
            signals = raw_frame.get("reward_signals", {})
            is_last = (i == len(frames) - 1)
            player_hp = frame["player"].get("hp", 1)
            is_terminal = is_last and player_hp <= 0

            reward = _compute_reward(signals, frame["player"], is_terminal, self.config)
            reward_list.append(reward)

        n = len(obs_list)
        if n == 0:
            return None

        obs = np.array(obs_list, dtype=np.float32)
        actions = np.array(action_list, dtype=np.int32)
        rewards = np.array(reward_list, dtype=np.float32)

        # Set dones: True only on the last frame if player died
        dones = np.zeros(n, dtype=bool)
        last_frame = frames[-1]
        last_player = last_frame.get("player", {})
        last_hp = last_player.get("my_health", last_player.get("hp", 1))
        if last_hp <= 0:
            dones[-1] = True

        return obs, actions, rewards, dones

    def validate(self, output_dir: str):
        """Validate converted files match the expected format."""
        config = self.config
        npz_files = sorted(Path(output_dir).glob("*.npz"))

        if not npz_files:
            print("No .npz files found to validate.")
            return

        total_transitions = 0
        total_episodes = len(npz_files)
        lengths = []
        action_counts = [np.zeros(config.n_move_dirs, dtype=int),
                         np.zeros(config.n_aim_angles, dtype=int),
                         np.zeros(2, dtype=int),
                         np.zeros(2, dtype=int)]

        for f in npz_files:
            data = np.load(f)
            obs = data["obs"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]

            n = len(obs)
            assert len(actions) == n and len(rewards) == n and len(dones) == n, \
                f"Array length mismatch in {f}"

            # Observation dimension check
            assert obs.shape[1] == config.obs_dim, \
                f"Obs dim {obs.shape[1]} != expected {config.obs_dim} in {f}"

            # Action range checks
            action_limits = [config.n_move_dirs, config.n_aim_angles, 2, 2]
            for dim, limit in enumerate(action_limits):
                assert np.all(actions[:, dim] >= 0) and np.all(actions[:, dim] < limit), \
                    f"Action dim {dim} out of range [0, {limit}) in {f}"

            total_transitions += n
            lengths.append(n)

            for dim in range(4):
                action_counts[dim] += np.bincount(actions[:, dim],
                                                  minlength=action_counts[dim].shape[0])

        print(f"Validation passed for {total_episodes} episode(s):")
        print(f"  Total transitions: {total_transitions:,}")
        print(f"  Episodes:          {total_episodes}")
        print(f"  Mean length:       {np.mean(lengths):.0f}")

        dim_names = ["move_dir", "aim_bin", "shoot", "special"]
        print("  Action distribution:")
        for dim, name in enumerate(dim_names):
            total = action_counts[dim].sum()
            if total > 0:
                pcts = action_counts[dim] / total * 100
                dominant = pcts.max()
                if len(pcts) <= 4:
                    dist_str = ", ".join(f"{p:.0f}%" for p in pcts)
                    print(f"    {name}: [{dist_str}]")
                else:
                    print(f"    {name}: max={dominant:.0f}% (value {pcts.argmax()})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NTT recording mod logs to BC training data (.npz)"
    )
    parser.add_argument("--input", required=True,
                        help="Directory containing .jsonl files, or a single .jsonl file")
    parser.add_argument("--output", default="demonstrations",
                        help="Output directory for .npz files (default: demonstrations)")
    parser.add_argument("--config-file", default=None,
                        help="Optional EnvConfig override (JSON file)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate output after conversion")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load config
    config = EnvConfig()
    if args.config_file:
        with open(args.config_file, "r") as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning("Unknown config key: %s", key)

    converter = NTTLogConverter(config=config)

    input_path = Path(args.input)
    if input_path.is_file():
        created = converter.convert_file(str(input_path), args.output)
        total = len(created)
    elif input_path.is_dir():
        total = converter.convert_directory(str(input_path), args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return

    print(f"\nConversion complete: {total} episode(s)")

    if args.validate and total > 0:
        print()
        converter.validate(args.output)


if __name__ == "__main__":
    main()
