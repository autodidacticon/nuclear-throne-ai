"""Tests for the behavioural cloning pipeline."""

import os
import tempfile
import time

import numpy as np
import pytest

from nt_rl.bc.recorder import discretize_action, DemonstrationRecorder
from nt_rl.bc.dataset import DemonstrationDataset
from nt_rl.config import EnvConfig
from nt_rl.tests.mock_server import MockRecordingServer


class TestDiscretizeAction:
    def test_east_movement(self):
        a = discretize_action({"moving": True, "move_dir": 0, "aim_dir": 90, "fire": True, "spec": False})
        assert list(a) == [0, 6, 1, 0]

    def test_no_movement(self):
        a = discretize_action({"moving": False, "move_dir": 45, "aim_dir": 0, "fire": False, "spec": True})
        assert list(a) == [8, 0, 0, 1]

    def test_wrap_360(self):
        a = discretize_action({"moving": True, "move_dir": 359, "aim_dir": 359, "fire": False, "spec": False})
        assert a[0] == 0  # wraps to east
        assert a[1] == 0  # wraps to bin 0

    def test_sw_direction(self):
        a = discretize_action({"moving": True, "move_dir": 225, "aim_dir": 180, "fire": True, "spec": True})
        assert a[0] == 5  # SW
        assert a[1] == 12  # 180/15 = 12

    def test_all_cardinal_directions(self):
        expected = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4, 225: 5, 270: 6, 315: 7}
        for angle, idx in expected.items():
            a = discretize_action({"moving": True, "move_dir": angle, "aim_dir": 0, "fire": False, "spec": False})
            assert a[0] == idx, f"angle {angle} expected {idx} got {a[0]}"


class TestRecorderAndDataset:
    @pytest.fixture
    def output_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_record_and_load(self, output_dir):
        """Integration test: mock server -> recorder -> dataset loader."""
        port = 18888
        server = MockRecordingServer(port=port, max_steps=30, n_episodes=2, fps=1000)
        server.start()

        try:
            recorder = DemonstrationRecorder(port=port, output_dir=output_dir)
            recorder.run()
        finally:
            server.stop()

        # Verify .npz files were created
        npz_files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
        assert len(npz_files) >= 1, "No episodes recorded"

        # Load via DemonstrationDataset
        ds = DemonstrationDataset(data_dir=output_dir)
        assert ds.n_transitions > 0
        assert ds.n_episodes >= 1
        assert ds.obs.shape[1] == EnvConfig().obs_dim
        assert ds.actions.shape[1] == 4

        # Validate action ranges
        assert ds.actions[:, 0].max() < 9
        assert ds.actions[:, 1].max() < 24
        assert ds.actions[:, 2].max() < 2
        assert ds.actions[:, 3].max() < 2

        # Test split
        train, val = ds.split(train_ratio=0.5)
        assert train.n_transitions > 0 or val.n_transitions > 0

        # Print stats (smoke test)
        ds.print_statistics()

    def test_dataset_not_found(self):
        with pytest.raises(FileNotFoundError, match="No demonstration dataset"):
            DemonstrationDataset(data_dir="/nonexistent/path")


# ---------------------------------------------------------------------------
# NTT Log Converter Tests
# ---------------------------------------------------------------------------

import json

from nt_rl.bc.ntt_converter import (
    NTTLogConverter,
    _map_variable_names,
    _compute_reward,
    _detect_episode_boundaries,
)
from nt_rl.obs_utils import encode_observation


def _make_ntt_frame(
    frame: int = 1,
    x: float = 5040.0,
    y: float = 4980.0,
    my_health: int = 6,
    maxhealth: int = 8,
    hspeed: float = 2.5,
    vspeed: float = -1.0,
    gunangle: float = 135.0,
    wep: int = 3,
    ammo: list | None = None,
    reload: int = 0,
    can_shoot: bool = True,
    roll: bool = False,
    race: int = 1,
    area: int = 1,
    subarea: int = 1,
    level: int = 2,
    loops: int = 0,
    kills: int = 7,
    move_dir: float = 90.0,
    moving: bool = True,
    aim_dir: float = 135.0,
    fire: bool = True,
    spec: bool = False,
    swap: bool = False,
    pick: bool = False,
    enemies: list | None = None,
    kills_this_frame: int = 0,
    damage_this_frame: int = 0,
    healed_this_frame: bool = False,
    level_changed: bool = False,
) -> dict:
    """Build a synthetic NTT frame dict with NTT variable names."""
    if ammo is None:
        ammo = [0, 15, 8, 0, 0, 0]
    if enemies is None:
        enemies = [
            {"x": 5200, "y": 4900, "my_health": 5, "maxhealth": 10, "type_id": 42}
        ]
    return {
        "frame": frame,
        "player": {
            "x": x, "y": y,
            "my_health": my_health, "maxhealth": maxhealth,
            "hspeed": hspeed, "vspeed": vspeed,
            "gunangle": gunangle,
            "wep": wep, "bwep": 12,
            "ammo": ammo,
            "reload": reload, "can_shoot": can_shoot,
            "roll": roll, "race": race, "nexthurt": 0,
        },
        "enemies": enemies,
        "game": {
            "area": area, "subarea": subarea, "level": level,
            "loops": loops, "kills": kills, "hard": 0,
        },
        "human_action": {
            "move_dir": move_dir, "moving": moving,
            "aim_dir": aim_dir, "fire": fire,
            "spec": spec, "swap": swap, "pick": pick,
        },
        "reward_signals": {
            "kills_this_frame": kills_this_frame,
            "damage_this_frame": damage_this_frame,
            "healed_this_frame": healed_this_frame,
            "level_changed": level_changed,
        },
    }


def _write_jsonl(frames: list[dict], path: str):
    """Write a list of frame dicts as a .jsonl file."""
    with open(path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")


class TestNTTConverter:
    @pytest.fixture
    def output_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def input_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    # ------------------------------------------------------------------
    # 1. Single episode conversion
    # ------------------------------------------------------------------
    def test_ntt_converter_single_episode(self, input_dir, output_dir):
        """Convert a synthetic .jsonl file, verify .npz output shapes and values."""
        config = EnvConfig()
        n_frames = 10

        frames = [_make_ntt_frame(frame=i, my_health=max(1, 6 - i))
                  for i in range(n_frames)]
        # Make last frame a death
        frames[-1]["player"]["my_health"] = 0

        jsonl_path = os.path.join(input_dir, "ntt_demo_test_0001.jsonl")
        _write_jsonl(frames, jsonl_path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(jsonl_path, output_dir)

        assert len(created) == 1, f"Expected 1 episode file, got {len(created)}"

        # Load and verify shapes
        data = np.load(created[0])
        assert data["obs"].shape == (n_frames, config.obs_dim)
        assert data["actions"].shape == (n_frames, 4)
        assert data["rewards"].shape == (n_frames,)
        assert data["dones"].shape == (n_frames,)

        # Verify dtypes
        assert data["obs"].dtype == np.float32
        assert data["actions"].dtype == np.int32
        assert data["rewards"].dtype == np.float32
        assert data["dones"].dtype == bool

        # Verify terminal flag on last frame (player died)
        assert data["dones"][-1] == True
        assert not np.any(data["dones"][:-1])

        # Action ranges
        assert np.all(data["actions"][:, 0] < config.n_move_dirs)
        assert np.all(data["actions"][:, 1] < config.n_aim_angles)
        assert np.all(data["actions"][:, 2] < 2)
        assert np.all(data["actions"][:, 3] < 2)

        # Loadable by DemonstrationDataset
        ds = DemonstrationDataset(data_dir=output_dir, config=config)
        assert ds.n_transitions == n_frames
        assert ds.n_episodes == 1

    # ------------------------------------------------------------------
    # 2. Variable name mapping
    # ------------------------------------------------------------------
    def test_ntt_converter_variable_mapping(self):
        """Verify NTT names produce identical observations to rebuild names."""
        config = EnvConfig()

        # NTT-format state
        ntt_state = {
            "player": {
                "x": 5040.0, "y": 4980.0,
                "my_health": 6, "maxhealth": 8,
                "hspeed": 2.5, "vspeed": -1.0,
                "gunangle": 135.0,
                "wep": 3, "bwep": 12,
                "ammo": [0, 15, 8, 0, 0, 0],
                "reload": 0, "can_shoot": True,
                "roll": False, "race": 1,
            },
            "enemies": [
                {"x": 5200, "y": 4900, "my_health": 5, "maxhealth": 10, "type_id": 42},
                {"x": 5300, "y": 5100, "my_health": 3, "maxhealth": 6, "type_id": 15},
            ],
            "game": {"area": 1, "subarea": 1, "level": 2, "loops": 0, "kills": 7, "hard": 0},
        }

        # Rebuild-format state (already uses hp/max_hp/hitid)
        rebuild_state = {
            "player": {
                "x": 5040.0, "y": 4980.0,
                "hp": 6, "max_hp": 8,
                "hspeed": 2.5, "vspeed": -1.0,
                "gunangle": 135.0,
                "wep": 3, "bwep": 12,
                "ammo": [0, 15, 8, 0, 0, 0],
                "reload": 0, "can_shoot": True,
                "roll": False, "race": 1,
            },
            "enemies": [
                {"x": 5200, "y": 4900, "hp": 5, "max_hp": 10, "hitid": 42},
                {"x": 5300, "y": 5100, "hp": 3, "max_hp": 6, "hitid": 15},
            ],
            "game": {"area": 1, "subarea": 1, "level": 2, "loops": 0, "kills": 7, "hard": 0},
        }

        # Map NTT names
        mapped = _map_variable_names(ntt_state)

        obs_mapped = encode_observation(mapped, config)
        obs_rebuild = encode_observation(rebuild_state, config)

        np.testing.assert_array_equal(obs_mapped, obs_rebuild)

    # ------------------------------------------------------------------
    # 3. Multi-episode detection
    # ------------------------------------------------------------------
    def test_ntt_converter_multi_episode(self, input_dir, output_dir):
        """A .jsonl with frame counter reset should produce multiple .npz files."""
        config = EnvConfig()

        # Episode 1: frames 0-4, player dies at end
        ep1 = [_make_ntt_frame(frame=i, my_health=max(0, 4 - i))
               for i in range(5)]
        ep1[-1]["player"]["my_health"] = 0

        # Episode 2: frames 0-3 (frame counter resets), player alive
        ep2 = [_make_ntt_frame(frame=i, my_health=8, maxhealth=8)
               for i in range(4)]

        all_frames = ep1 + ep2

        jsonl_path = os.path.join(input_dir, "ntt_demo_multi_0001.jsonl")
        _write_jsonl(all_frames, jsonl_path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(jsonl_path, output_dir)

        assert len(created) == 2, f"Expected 2 episodes, got {len(created)}"

        # Episode 1: 5 frames, terminated
        data1 = np.load(created[0])
        assert data1["obs"].shape[0] == 5
        assert data1["dones"][-1] == True

        # Episode 2: 4 frames, not terminated (player alive)
        data2 = np.load(created[1])
        assert data2["obs"].shape[0] == 4
        assert data2["dones"][-1] == False

        # Both loadable by DemonstrationDataset
        ds = DemonstrationDataset(data_dir=output_dir, config=config)
        assert ds.n_episodes == 2
        assert ds.n_transitions == 9

    # ------------------------------------------------------------------
    # 4. Reward computation
    # ------------------------------------------------------------------
    def test_ntt_converter_reward_computation(self):
        """Verify reward signals are correctly weighted."""
        config = EnvConfig()

        # Base survival reward
        signals_base = {
            "kills_this_frame": 0,
            "damage_this_frame": 0,
            "healed_this_frame": False,
            "level_changed": False,
        }
        player_base = {"hp": 6, "max_hp": 8}
        r = _compute_reward(signals_base, player_base, is_terminal=False, config=config)
        assert r == pytest.approx(config.reward_survival_per_step)

        # Kill reward
        signals_kill = dict(signals_base, kills_this_frame=2)
        r = _compute_reward(signals_kill, player_base, is_terminal=False, config=config)
        expected = config.reward_survival_per_step + 2 * config.reward_kill
        assert r == pytest.approx(expected)

        # Damage penalty
        signals_dmg = dict(signals_base, damage_this_frame=3)
        r = _compute_reward(signals_dmg, player_base, is_terminal=False, config=config)
        expected = config.reward_survival_per_step + 3 * config.reward_damage_taken
        assert r == pytest.approx(expected)

        # Heal reward: only applies when HP < 50%
        player_low_hp = {"hp": 2, "max_hp": 8}
        signals_heal = dict(signals_base, healed_this_frame=True)
        r = _compute_reward(signals_heal, player_low_hp, is_terminal=False, config=config)
        expected = config.reward_survival_per_step + config.reward_health_pickup
        assert r == pytest.approx(expected)

        # Heal reward: should NOT apply when HP >= 50%
        player_high_hp = {"hp": 5, "max_hp": 8}
        r = _compute_reward(signals_heal, player_high_hp, is_terminal=False, config=config)
        expected = config.reward_survival_per_step  # No heal bonus
        assert r == pytest.approx(expected)

        # Level complete
        signals_level = dict(signals_base, level_changed=True)
        r = _compute_reward(signals_level, player_base, is_terminal=False, config=config)
        expected = config.reward_survival_per_step + config.reward_level_complete
        assert r == pytest.approx(expected)

        # Death penalty
        r = _compute_reward(signals_base, player_base, is_terminal=True, config=config)
        expected = config.reward_survival_per_step + config.reward_death
        assert r == pytest.approx(expected)

        # Combined: kill + damage + death
        signals_combo = {
            "kills_this_frame": 1,
            "damage_this_frame": 2,
            "healed_this_frame": False,
            "level_changed": False,
        }
        r = _compute_reward(signals_combo, player_base, is_terminal=True, config=config)
        expected = (config.reward_survival_per_step
                    + 1 * config.reward_kill
                    + 2 * config.reward_damage_taken
                    + config.reward_death)
        assert r == pytest.approx(expected)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------
    def test_ntt_converter_empty_file(self, input_dir, output_dir):
        """An empty .jsonl produces no output files."""
        jsonl_path = os.path.join(input_dir, "empty.jsonl")
        with open(jsonl_path, "w") as f:
            pass

        converter = NTTLogConverter()
        created = converter.convert_file(jsonl_path, output_dir)
        assert len(created) == 0

    def test_ntt_converter_health_jump_boundary(self, input_dir, output_dir):
        """Episode boundary from health jump (hp 0 -> maxhealth)."""
        config = EnvConfig()

        # Episode 1: player dies (frame counter does NOT reset)
        ep1 = [_make_ntt_frame(frame=i, my_health=max(0, 3 - i))
               for i in range(4)]
        ep1[-1]["player"]["my_health"] = 0

        # Episode 2: same ascending frame counter, but health jumps back
        ep2 = [_make_ntt_frame(frame=4 + i, my_health=8, maxhealth=8)
               for i in range(3)]

        all_frames = ep1 + ep2
        jsonl_path = os.path.join(input_dir, "health_jump.jsonl")
        _write_jsonl(all_frames, jsonl_path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(jsonl_path, output_dir)

        assert len(created) == 2
        data1 = np.load(created[0])
        data2 = np.load(created[1])
        assert data1["obs"].shape[0] == 4
        assert data2["obs"].shape[0] == 3

    def test_ntt_converter_directory(self, input_dir, output_dir):
        """convert_directory processes all .jsonl files in a directory."""
        config = EnvConfig()

        for file_idx in range(3):
            frames = [_make_ntt_frame(frame=i, my_health=4)
                      for i in range(5)]
            path = os.path.join(input_dir, f"ntt_demo_test_{file_idx:04d}.jsonl")
            _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        total = converter.convert_directory(input_dir, output_dir)

        assert total == 3

        ds = DemonstrationDataset(data_dir=output_dir, config=config)
        assert ds.n_episodes == 3
        assert ds.n_transitions == 15
