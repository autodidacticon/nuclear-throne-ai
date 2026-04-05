"""Demonstration dataset loading and preprocessing for behavioural cloning."""

import glob
import os
import warnings

import numpy as np

from nt_rl.config import EnvConfig


class DemonstrationDataset:
    """Loads and validates demonstration data from per-episode .npz files.

    Expected format: demonstrations/ directory containing .npz files,
    each with arrays: obs (T,112), actions (T,4), rewards (T,), dones (T,).
    """

    def __init__(self, data_dir: str = "demonstrations", config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.data_dir = data_dir

        self.obs: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.rewards: np.ndarray | None = None
        self.dones: np.ndarray | None = None

        # Per-episode boundaries for splitting by episode
        self._episode_boundaries: list[tuple[int, int]] = []

        self._load()

    def _load(self):
        npz_files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))

        if not npz_files:
            raise FileNotFoundError(
                f"No demonstration dataset found at expected paths. "
                f"Human intervention required. "
                f"Searched: {self.data_dir}/*.npz"
            )

        all_obs, all_actions, all_rewards, all_dones = [], [], [], []
        offset = 0

        for f in npz_files:
            data = np.load(f)
            obs = data["obs"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]

            n = len(obs)
            assert len(actions) == n and len(rewards) == n and len(dones) == n, \
                f"Array length mismatch in {f}"

            all_obs.append(obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_dones.append(dones)
            self._episode_boundaries.append((offset, offset + n))
            offset += n

        self.obs = np.concatenate(all_obs).astype(np.float32)
        self.actions = np.concatenate(all_actions).astype(np.int32)
        self.rewards = np.concatenate(all_rewards).astype(np.float32)
        self.dones = np.concatenate(all_dones).astype(bool)

        self._validate()

    def _validate(self):
        expected_dim = self.config.obs_dim
        if self.obs.shape[1] != expected_dim:
            raise ValueError(
                f"Observation dimension mismatch: got {self.obs.shape[1]}, "
                f"expected {expected_dim}"
            )

        action_limits = np.array([
            self.config.n_move_dirs,
            self.config.n_aim_angles,
            2, 2,
        ])
        for dim in range(4):
            if np.any(self.actions[:, dim] < 0) or np.any(self.actions[:, dim] >= action_limits[dim]):
                raise ValueError(
                    f"Action dimension {dim} out of range [0, {action_limits[dim]}): "
                    f"min={self.actions[:, dim].min()}, max={self.actions[:, dim].max()}"
                )

        # Flag truncated episodes
        truncated = 0
        for start, end in self._episode_boundaries:
            if not self.dones[end - 1]:
                truncated += 1
        if truncated:
            warnings.warn(f"{truncated} episode(s) appear truncated (done never True)")

    @property
    def n_transitions(self) -> int:
        return len(self.obs)

    @property
    def n_episodes(self) -> int:
        return len(self._episode_boundaries)

    def print_statistics(self):
        """Print dataset summary statistics."""
        print(f"Dataset: {self.data_dir}")
        print(f"  Transitions: {self.n_transitions:,}")
        print(f"  Episodes:    {self.n_episodes}")

        lengths = [end - start for start, end in self._episode_boundaries]
        ep_rewards = [
            float(self.rewards[start:end].sum())
            for start, end in self._episode_boundaries
        ]

        print(f"  Mean episode length:  {np.mean(lengths):.0f}")
        print(f"  Mean episode reward:  {np.mean(ep_rewards):.1f}")

        # Action distribution
        print("  Action distribution:")
        dim_names = ["move_dir", "aim_bin", "shoot", "special"]
        dim_sizes = [self.config.n_move_dirs, self.config.n_aim_angles, 2, 2]

        for dim, (name, size) in enumerate(zip(dim_names, dim_sizes)):
            counts = np.bincount(self.actions[:, dim], minlength=size)
            pcts = counts / counts.sum() * 100
            dominant = pcts.max()
            if dominant > 95:
                warnings.warn(f"Class imbalance: {name} has {dominant:.0f}% in one value")
            if size <= 4:
                dist_str = ", ".join(f"{p:.0f}%" for p in pcts)
                print(f"    {name}: [{dist_str}]")
            else:
                print(f"    {name}: max={dominant:.0f}% (value {pcts.argmax()})")

        # Level coverage
        level_idx = 11  # level_norm is at index 11
        levels = np.unique(np.round(self.obs[:, level_idx] * self.config.max_level))
        print(f"  Distinct levels: {len(levels)} {sorted(levels.astype(int).tolist())}")
        if len(levels) < 3:
            warnings.warn("Dataset covers fewer than 3 levels — policy may not generalize")

    def split(self, train_ratio: float = 0.9) -> tuple["DemonstrationDataset", "DemonstrationDataset"]:
        """Split by episode into train and validation sets."""
        n_train = max(1, int(self.n_episodes * train_ratio))

        # Shuffle episode indices deterministically
        rng = np.random.RandomState(42)
        ep_indices = rng.permutation(self.n_episodes)
        train_eps = sorted(ep_indices[:n_train])
        val_eps = sorted(ep_indices[n_train:])

        train_ds = self._subset(train_eps)
        val_ds = self._subset(val_eps)
        return train_ds, val_ds

    def _subset(self, episode_indices: list[int]) -> "DemonstrationDataset":
        """Create a new dataset from a subset of episodes."""
        ds = DemonstrationDataset.__new__(DemonstrationDataset)
        ds.config = self.config
        ds.data_dir = self.data_dir
        ds._episode_boundaries = []

        indices = []
        offset = 0
        for ep_idx in episode_indices:
            start, end = self._episode_boundaries[ep_idx]
            indices.extend(range(start, end))
            ds._episode_boundaries.append((offset, offset + (end - start)))
            offset += end - start

        indices = np.array(indices)
        ds.obs = self.obs[indices]
        ds.actions = self.actions[indices]
        ds.rewards = self.rewards[indices]
        ds.dones = self.dones[indices]
        return ds

    def to_imitation_trajectories(self):
        """Convert to imitation library Trajectory objects."""
        from imitation.data.types import Trajectory

        trajectories = []
        for start, end in self._episode_boundaries:
            terminal = bool(self.dones[end - 1])
            # imitation expects obs to have one extra element (final obs after last action)
            # We duplicate the last observation as the terminal observation
            obs_seq = np.concatenate([
                self.obs[start:end],
                self.obs[end - 1:end],
            ])
            traj = Trajectory(
                obs=obs_seq,
                acts=self.actions[start:end],
                infos=None,
                terminal=terminal,
            )
            trajectories.append(traj)

        return trajectories
