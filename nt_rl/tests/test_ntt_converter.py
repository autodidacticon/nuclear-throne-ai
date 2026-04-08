"""Tests for the NTT recording converter (ntt_converter.py).

Covers: JSONL parsing (including sanitization of non-standard NTT JSON),
episode boundary detection, variable name mapping, single/multi-episode
conversion, directory conversion, output validation, and edge cases.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from nt_rl.bc.ntt_converter import (
    NTTLogConverter,
    _detect_episode_boundaries,
    _compute_reward,
    _discretize_action_from_velocity,
    _map_variable_names,
    _sanitize_ntt_json,
)
from nt_rl.config import EnvConfig
from nt_rl.obs_utils import encode_observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(
    frame: int = 0,
    x: float = 10044.26,
    y: float = 10016.0,
    my_health: int = 10,
    maxhealth: int = 10,
    hspeed: float = 0.0,
    vspeed: float = 0.0,
    gunangle: float = 112.98,
    wep: int = 1,
    ammo: list | None = None,
    reload: float = -0.50,
    can_shoot: bool = True,
    roll: bool = False,
    race: str = "crystal",
    nexthurt: int = 123,
    enemies: list | None = None,
    area: int = 1,
    subarea: int = 1,
    level: int = 1,
    loops: int = 0,
    kills: int = 3,
    hard: int = 1,
    move_dir: float = 0.0,
    moving: bool = False,
    aim_dir: float = 112.98,
    fire: bool = False,
    spec: bool = False,
    swap: bool = False,
    pick: bool = False,
    kills_this_frame: int = 0,
    damage_this_frame: int = 0,
    healed_this_frame: bool = False,
    level_changed: bool = False,
) -> dict:
    """Build a synthetic NTT-format frame dict (uses NTT variable names)."""
    if ammo is None:
        ammo = [999, 92, 0, 0, 0, 0]
    if enemies is None:
        enemies = [
            {"x": 9971.78, "y": 9910.53, "my_health": 4, "maxhealth": 4,
             "type_id": "ref object 14"},
        ]
    return {
        "frame": frame,
        "player": {
            "x": x, "y": y,
            "my_health": my_health, "maxhealth": maxhealth,
            "hspeed": hspeed, "vspeed": vspeed,
            "gunangle": gunangle,
            "wep": wep, "bwep": 0,
            "ammo": ammo,
            "reload": reload, "can_shoot": can_shoot,
            "roll": roll, "race": race, "nexthurt": nexthurt,
        },
        "enemies": enemies,
        "game": {
            "area": area, "subarea": subarea, "level": level,
            "loops": loops, "kills": kills, "hard": hard,
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
    """Write a list of frame dicts as a .jsonl file (standard JSON)."""
    with open(path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")


def _write_raw_ntt_jsonl(lines: list[str], path: str):
    """Write raw string lines to a .jsonl file (may be non-standard JSON)."""
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def input_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# 1. JSONL sanitizer: quoting unquoted string values
# ---------------------------------------------------------------------------

class TestSanitizeNttJson:
    def test_quotes_race_field(self):
        line = '{"race":crystal}'
        result = _sanitize_ntt_json(line)
        assert result == '{"race":"crystal"}'
        assert json.loads(result) == {"race": "crystal"}

    def test_quotes_ref_object_type_id(self):
        line = '{"type_id":ref object 14}'
        result = _sanitize_ntt_json(line)
        assert result == '{"type_id":"ref object 14"}'
        parsed = json.loads(result)
        assert parsed["type_id"] == "ref object 14"

    def test_quotes_multiple_ref_objects(self):
        line = '{"enemies":[{"type_id":ref object 14},{"type_id":ref object 298}]}'
        result = _sanitize_ntt_json(line)
        parsed = json.loads(result)
        assert parsed["enemies"][0]["type_id"] == "ref object 14"
        assert parsed["enemies"][1]["type_id"] == "ref object 298"

    def test_preserves_true_false_null(self):
        line = '{"can_shoot":true,"roll":false,"data":null}'
        result = _sanitize_ntt_json(line)
        assert result == line  # no change
        parsed = json.loads(result)
        assert parsed["can_shoot"] is True
        assert parsed["roll"] is False
        assert parsed["data"] is None

    def test_preserves_numeric_values(self):
        line = '{"x":10044.26,"frame":0,"my_health":10}'
        result = _sanitize_ntt_json(line)
        # Numeric values start with a digit, so the regex should not match them
        assert result == line
        parsed = json.loads(result)
        assert parsed["x"] == 10044.26

    def test_preserves_already_quoted_strings(self):
        line = '{"race":"crystal","type_id":"ref object 14"}'
        result = _sanitize_ntt_json(line)
        # Already-quoted strings have their value starting with ", not a letter
        parsed = json.loads(result)
        assert parsed["race"] == "crystal"
        assert parsed["type_id"] == "ref object 14"

    def test_quotes_robot_race(self):
        line = '{"race":robot}'
        result = _sanitize_ntt_json(line)
        assert json.loads(result) == {"race": "robot"}

    def test_full_real_line(self):
        """Sanitize a line matching real NTT output format."""
        line = (
            '{"frame":0,"player":{"x":100,"y":200,"my_health":10,"maxhealth":10,'
            '"hspeed":0,"vspeed":0,"gunangle":90,"wep":1,"bwep":0,'
            '"ammo":[999,92,0,0,0,0],"reload":-0.50,"can_shoot":true,'
            '"roll":false,"race":crystal,"nexthurt":123},'
            '"enemies":[{"x":50,"y":60,"my_health":4,"maxhealth":4,'
            '"type_id":ref object 14}],'
            '"game":{"area":1,"subarea":1,"level":1,"loops":0,"kills":3,"hard":1},'
            '"human_action":{"move_dir":0,"moving":false,"aim_dir":90,'
            '"fire":false,"spec":false,"swap":false,"pick":false},'
            '"reward_signals":{"kills_this_frame":0,"damage_this_frame":0,'
            '"healed_this_frame":false,"level_changed":false}}'
        )
        result = _sanitize_ntt_json(line)
        parsed = json.loads(result)
        assert parsed["player"]["race"] == "crystal"
        assert parsed["enemies"][0]["type_id"] == "ref object 14"
        assert parsed["player"]["can_shoot"] is True
        assert parsed["player"]["roll"] is False


# ---------------------------------------------------------------------------
# 2. JSONL parsing (valid and malformed lines)
# ---------------------------------------------------------------------------

class TestParseJsonl:
    def test_parses_valid_standard_json(self, input_dir):
        """Standard JSON frames parse without issues."""
        frames = [_make_frame(frame=i) for i in range(5)]
        path = os.path.join(input_dir, "valid.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter()
        parsed = converter._parse_jsonl(path)
        assert len(parsed) == 5
        assert parsed[0]["frame"] == 0
        assert parsed[4]["frame"] == 4

    def test_parses_ntt_nonstandard_json(self, input_dir):
        """Non-standard NTT JSON with unquoted strings is sanitized and parsed."""
        raw_line = (
            '{"frame":0,"player":{"x":100,"y":200,"my_health":10,"maxhealth":10,'
            '"hspeed":0,"vspeed":0,"gunangle":90,"wep":1,"bwep":0,'
            '"ammo":[999,92,0,0,0,0],"reload":0,"can_shoot":true,'
            '"roll":false,"race":crystal,"nexthurt":0},'
            '"enemies":[{"x":50,"y":60,"my_health":4,"maxhealth":4,'
            '"type_id":ref object 14}],'
            '"game":{"area":1,"subarea":1,"level":1,"loops":0,"kills":0,"hard":1},'
            '"human_action":{"move_dir":0,"moving":false,"aim_dir":90,'
            '"fire":false,"spec":false,"swap":false,"pick":false},'
            '"reward_signals":{"kills_this_frame":0,"damage_this_frame":0,'
            '"healed_this_frame":false,"level_changed":false}}'
        )
        path = os.path.join(input_dir, "ntt_raw.jsonl")
        _write_raw_ntt_jsonl([raw_line], path)

        converter = NTTLogConverter()
        parsed = converter._parse_jsonl(path)
        assert len(parsed) == 1
        assert parsed[0]["player"]["race"] == "crystal"
        assert parsed[0]["enemies"][0]["type_id"] == "ref object 14"

    def test_skips_malformed_lines(self, input_dir):
        """Completely malformed lines are skipped, valid lines are kept."""
        frame0 = json.dumps(_make_frame(frame=0))
        frame1 = json.dumps(_make_frame(frame=1))
        path = os.path.join(input_dir, "mixed.jsonl")
        _write_raw_ntt_jsonl([
            frame0,
            "NOT JSON AT ALL {{{{",
            "",  # blank line
            frame1,
            "{truncated json",
        ], path)

        converter = NTTLogConverter()
        parsed = converter._parse_jsonl(path)
        assert len(parsed) == 2
        assert parsed[0]["frame"] == 0
        assert parsed[1]["frame"] == 1

    def test_empty_file_returns_empty_list(self, input_dir):
        path = os.path.join(input_dir, "empty.jsonl")
        with open(path, "w"):
            pass

        converter = NTTLogConverter()
        parsed = converter._parse_jsonl(path)
        assert parsed == []

    def test_blank_lines_only(self, input_dir):
        path = os.path.join(input_dir, "blanks.jsonl")
        _write_raw_ntt_jsonl(["", "   ", "\t", ""], path)

        converter = NTTLogConverter()
        parsed = converter._parse_jsonl(path)
        assert parsed == []


# ---------------------------------------------------------------------------
# 3. Episode boundary detection
# ---------------------------------------------------------------------------

class TestDetectEpisodeBoundaries:
    def test_single_episode_no_boundaries(self):
        frames = [_make_frame(frame=i) for i in range(10)]
        episodes = _detect_episode_boundaries(frames)
        assert len(episodes) == 1
        assert len(episodes[0]) == 10

    def test_frame_counter_reset_splits_episodes(self):
        # Episode 1: frames 0-4
        ep1 = [_make_frame(frame=i, my_health=5) for i in range(5)]
        # Episode 2: frame counter resets to 0
        ep2 = [_make_frame(frame=i, my_health=10) for i in range(3)]
        episodes = _detect_episode_boundaries(ep1 + ep2)
        assert len(episodes) == 2
        assert len(episodes[0]) == 5
        assert len(episodes[1]) == 3

    def test_health_jump_splits_episodes(self):
        # Episode 1: player dies (hp drops to 0)
        ep1 = [_make_frame(frame=i, my_health=max(0, 3 - i), maxhealth=10)
               for i in range(4)]
        ep1[-1]["player"]["my_health"] = 0

        # Episode 2: frame counter continues but health jumps back to max
        ep2 = [_make_frame(frame=4 + i, my_health=10, maxhealth=10)
               for i in range(3)]

        episodes = _detect_episode_boundaries(ep1 + ep2)
        assert len(episodes) == 2
        assert len(episodes[0]) == 4
        assert len(episodes[1]) == 3

    def test_empty_frames_list(self):
        episodes = _detect_episode_boundaries([])
        assert episodes == []

    def test_single_frame(self):
        episodes = _detect_episode_boundaries([_make_frame(frame=0)])
        assert len(episodes) == 1
        assert len(episodes[0]) == 1

    def test_three_episodes_mixed_boundaries(self):
        # Episode 1: frames 0-2 (death at frame 2)
        ep1 = [_make_frame(frame=0, my_health=5, maxhealth=8),
               _make_frame(frame=1, my_health=2, maxhealth=8),
               _make_frame(frame=2, my_health=0, maxhealth=8)]
        # Episode 2: health jump (frame continues)
        ep2 = [_make_frame(frame=3, my_health=8, maxhealth=8),
               _make_frame(frame=4, my_health=7, maxhealth=8)]
        # Episode 3: frame counter resets
        ep3 = [_make_frame(frame=0, my_health=8, maxhealth=8)]

        episodes = _detect_episode_boundaries(ep1 + ep2 + ep3)
        assert len(episodes) == 3
        assert len(episodes[0]) == 3
        assert len(episodes[1]) == 2
        assert len(episodes[2]) == 1


# ---------------------------------------------------------------------------
# 4. Variable name mapping
# ---------------------------------------------------------------------------

class TestMapVariableNames:
    def test_player_health_renamed(self):
        state = {"player": {"my_health": 6, "maxhealth": 8}, "enemies": []}
        result = _map_variable_names(state)
        assert "hp" in result["player"]
        assert "max_hp" in result["player"]
        assert "my_health" not in result["player"]
        assert "maxhealth" not in result["player"]
        assert result["player"]["hp"] == 6
        assert result["player"]["max_hp"] == 8

    def test_enemy_health_renamed(self):
        state = {
            "player": {},
            "enemies": [{"my_health": 4, "maxhealth": 4, "type_id": 14}],
        }
        result = _map_variable_names(state)
        enemy = result["enemies"][0]
        assert enemy["hp"] == 4
        assert enemy["max_hp"] == 4
        assert "my_health" not in enemy
        assert "maxhealth" not in enemy

    def test_type_id_numeric_to_hitid(self):
        state = {"player": {}, "enemies": [{"type_id": 42}]}
        _map_variable_names(state)
        assert state["enemies"][0]["hitid"] == 42
        assert "type_id" not in state["enemies"][0]

    def test_type_id_ref_object_string_to_hitid(self):
        state = {"player": {}, "enemies": [{"type_id": "ref object 14"}]}
        _map_variable_names(state)
        assert state["enemies"][0]["hitid"] == 14

    def test_type_id_ref_object_large_number(self):
        state = {"player": {}, "enemies": [{"type_id": "ref object 426"}]}
        _map_variable_names(state)
        assert state["enemies"][0]["hitid"] == 426

    def test_type_id_invalid_string_defaults_to_zero(self):
        state = {"player": {}, "enemies": [{"type_id": "garbage string"}]}
        _map_variable_names(state)
        assert state["enemies"][0]["hitid"] == 0

    def test_type_id_empty_string_defaults_to_zero(self):
        state = {"player": {}, "enemies": [{"type_id": ""}]}
        _map_variable_names(state)
        assert state["enemies"][0]["hitid"] == 0

    def test_mapped_obs_matches_rebuild_format(self):
        """After mapping, encode_observation should produce same result as rebuild format."""
        config = EnvConfig()

        ntt_state = {
            "player": {
                "x": 5040.0, "y": 4980.0,
                "my_health": 6, "maxhealth": 8,
                "hspeed": 2.5, "vspeed": -1.0,
                "gunangle": 135.0, "wep": 3, "bwep": 12,
                "ammo": [0, 15, 8, 0, 0, 0],
                "reload": 0, "can_shoot": True,
                "roll": False, "race": 1,
            },
            "enemies": [
                {"x": 5200, "y": 4900, "my_health": 5, "maxhealth": 10,
                 "type_id": "ref object 42"},
            ],
            "game": {"area": 1, "subarea": 1, "level": 2, "loops": 0,
                     "kills": 7, "hard": 0},
        }

        rebuild_state = {
            "player": {
                "x": 5040.0, "y": 4980.0,
                "hp": 6, "max_hp": 8,
                "hspeed": 2.5, "vspeed": -1.0,
                "gunangle": 135.0, "wep": 3, "bwep": 12,
                "ammo": [0, 15, 8, 0, 0, 0],
                "reload": 0, "can_shoot": True,
                "roll": False, "race": 1,
            },
            "enemies": [
                {"x": 5200, "y": 4900, "hp": 5, "max_hp": 10, "hitid": 42},
            ],
            "game": {"area": 1, "subarea": 1, "level": 2, "loops": 0,
                     "kills": 7, "hard": 0},
        }

        mapped = _map_variable_names(ntt_state)
        obs_mapped = encode_observation(mapped, config)
        obs_rebuild = encode_observation(rebuild_state, config)
        np.testing.assert_array_equal(obs_mapped, obs_rebuild)

    def test_multiple_enemies_all_mapped(self):
        state = {
            "player": {"my_health": 10, "maxhealth": 10},
            "enemies": [
                {"my_health": 4, "maxhealth": 4, "type_id": "ref object 14"},
                {"my_health": 2, "maxhealth": 2, "type_id": "ref object 298"},
                {"my_health": 15, "maxhealth": 15, "type_id": 426},
            ],
        }
        _map_variable_names(state)
        assert state["enemies"][0]["hitid"] == 14
        assert state["enemies"][1]["hitid"] == 298
        assert state["enemies"][2]["hitid"] == 426
        for e in state["enemies"]:
            assert "hp" in e
            assert "max_hp" in e
            assert "my_health" not in e
            assert "maxhealth" not in e
            assert "type_id" not in e


# ---------------------------------------------------------------------------
# 5. Velocity-based action discretization
# ---------------------------------------------------------------------------

class TestDiscretizeActionFromVelocity:
    """The NTT mod's move_dir/moving fields are broken (only 0 or 180).
    Movement is derived from hspeed/vspeed instead."""

    def test_stationary_when_speed_below_threshold(self):
        player = {"hspeed": 0.1, "vspeed": 0.1}
        action = _discretize_action_from_velocity(player, {"aim_dir": 0})
        assert action[0] == 8  # no movement

    def test_east(self):
        player = {"hspeed": 3.0, "vspeed": 0.0}
        action = _discretize_action_from_velocity(player, {"aim_dir": 0})
        assert action[0] == 0  # E

    def test_north(self):
        # Negative vspeed = up in GameMaker
        player = {"hspeed": 0.0, "vspeed": -3.0}
        action = _discretize_action_from_velocity(player, {"aim_dir": 0})
        assert action[0] == 2  # N

    def test_southwest(self):
        player = {"hspeed": -2.0, "vspeed": 2.0}
        action = _discretize_action_from_velocity(player, {"aim_dir": 0})
        assert action[0] == 5  # SW

    def test_all_eight_directions(self):
        """Each 45-degree sector maps to the expected bin."""
        import math
        # (hspeed, vspeed) -> expected bin
        cases = [
            (3.0, 0.0, 0),    # E
            (2.0, -2.0, 1),   # NE
            (0.0, -3.0, 2),   # N
            (-2.0, -2.0, 3),  # NW
            (-3.0, 0.0, 4),   # W
            (-2.0, 2.0, 5),   # SW
            (0.0, 3.0, 6),    # S
            (2.0, 2.0, 7),    # SE
        ]
        for hs, vs, expected_bin in cases:
            player = {"hspeed": hs, "vspeed": vs}
            action = _discretize_action_from_velocity(player, {"aim_dir": 0})
            assert action[0] == expected_bin, \
                f"hspeed={hs}, vspeed={vs}: expected bin {expected_bin}, got {action[0]}"

    def test_aim_shoot_special_from_human_action(self):
        """Aim/shoot/special still come from human_action, not velocity."""
        player = {"hspeed": 3.0, "vspeed": 0.0}
        ha = {"aim_dir": 90.0, "fire": True, "spec": True}
        action = _discretize_action_from_velocity(player, ha, n_aim_angles=24)
        assert action[1] == 6   # 90 / 15 = 6
        assert action[2] == 1   # shoot
        assert action[3] == 1   # special

    def test_missing_velocity_defaults_to_stationary(self):
        player = {}
        action = _discretize_action_from_velocity(player, {"aim_dir": 0})
        assert action[0] == 8  # no movement

    def test_converter_uses_velocity_not_move_dir(self):
        """End-to-end: converter ignores broken move_dir, uses hspeed/vspeed."""
        import tempfile
        config = EnvConfig()
        # move_dir says East (0), but velocity says North
        frames = [_make_frame(frame=0, hspeed=0.0, vspeed=-3.0,
                              moving=True, move_dir=0.0)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            _write_jsonl(frames, path)
            out = os.path.join(d, "out")
            converter = NTTLogConverter(config=config)
            created = converter.convert_file(path, out)
            data = np.load(created[0])
            assert data["actions"][0, 0] == 2  # N from velocity, not 0 (E) from move_dir


# ---------------------------------------------------------------------------
# 6. Reward computation
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_survival_only(self):
        config = EnvConfig()
        signals = {"kills_this_frame": 0, "damage_this_frame": 0,
                    "healed_this_frame": False, "level_changed": False}
        player = {"hp": 8, "max_hp": 8}
        r = _compute_reward(signals, player, is_terminal=False, config=config)
        assert r == pytest.approx(config.reward_survival_per_step)

    def test_kill_reward(self):
        config = EnvConfig()
        signals = {"kills_this_frame": 3, "damage_this_frame": 0,
                    "healed_this_frame": False, "level_changed": False}
        player = {"hp": 8, "max_hp": 8}
        r = _compute_reward(signals, player, is_terminal=False, config=config)
        expected = config.reward_survival_per_step + 3 * config.reward_kill
        assert r == pytest.approx(expected)

    def test_damage_penalty(self):
        config = EnvConfig()
        signals = {"kills_this_frame": 0, "damage_this_frame": 2,
                    "healed_this_frame": False, "level_changed": False}
        player = {"hp": 6, "max_hp": 8}
        r = _compute_reward(signals, player, is_terminal=False, config=config)
        expected = config.reward_survival_per_step + 2 * config.reward_damage_taken
        assert r == pytest.approx(expected)

    def test_heal_below_half_hp(self):
        config = EnvConfig()
        signals = {"kills_this_frame": 0, "damage_this_frame": 0,
                    "healed_this_frame": True, "level_changed": False}
        player = {"hp": 3, "max_hp": 8}  # 3/8 = 0.375 < 0.5
        r = _compute_reward(signals, player, is_terminal=False, config=config)
        expected = config.reward_survival_per_step + config.reward_health_pickup
        assert r == pytest.approx(expected)

    def test_heal_above_half_hp_no_bonus(self):
        config = EnvConfig()
        signals = {"kills_this_frame": 0, "damage_this_frame": 0,
                    "healed_this_frame": True, "level_changed": False}
        player = {"hp": 5, "max_hp": 8}  # 5/8 = 0.625 >= 0.5
        r = _compute_reward(signals, player, is_terminal=False, config=config)
        expected = config.reward_survival_per_step  # no heal bonus
        assert r == pytest.approx(expected)

    def test_level_complete(self):
        config = EnvConfig()
        signals = {"kills_this_frame": 0, "damage_this_frame": 0,
                    "healed_this_frame": False, "level_changed": True}
        player = {"hp": 8, "max_hp": 8}
        r = _compute_reward(signals, player, is_terminal=False, config=config)
        expected = config.reward_survival_per_step + config.reward_level_complete
        assert r == pytest.approx(expected)

    def test_death_penalty(self):
        config = EnvConfig()
        signals = {"kills_this_frame": 0, "damage_this_frame": 0,
                    "healed_this_frame": False, "level_changed": False}
        player = {"hp": 0, "max_hp": 8}
        r = _compute_reward(signals, player, is_terminal=True, config=config)
        expected = config.reward_survival_per_step + config.reward_death
        assert r == pytest.approx(expected)

    def test_combined_signals(self):
        config = EnvConfig()
        signals = {"kills_this_frame": 1, "damage_this_frame": 1,
                    "healed_this_frame": False, "level_changed": True}
        player = {"hp": 6, "max_hp": 8}
        r = _compute_reward(signals, player, is_terminal=True, config=config)
        expected = (config.reward_survival_per_step
                    + 1 * config.reward_kill
                    + 1 * config.reward_damage_taken
                    + config.reward_level_complete
                    + config.reward_death)
        assert r == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6. Single episode conversion
# ---------------------------------------------------------------------------

class TestSingleEpisodeConversion:
    def test_obs_shape_and_dtype(self, input_dir, output_dir):
        config = EnvConfig()
        n = 5
        frames = [_make_frame(frame=i) for i in range(n)]
        path = os.path.join(input_dir, "single.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        assert len(created) == 1

        data = np.load(created[0])
        assert data["obs"].shape == (n, config.obs_dim)
        assert data["obs"].dtype == np.float32

    def test_action_shape_and_ranges(self, input_dir, output_dir):
        config = EnvConfig()
        # Use hspeed/vspeed to set movement (converter derives direction from velocity)
        frames = [_make_frame(frame=i, hspeed=0.0, vspeed=-3.0,
                              aim_dir=45.0, fire=True, spec=False)
                  for i in range(8)]
        path = os.path.join(input_dir, "actions.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        data = np.load(created[0])

        assert data["actions"].shape == (8, 4)
        assert data["actions"].dtype == np.int32
        assert np.all(data["actions"][:, 0] >= 0)
        assert np.all(data["actions"][:, 0] < config.n_move_dirs)
        assert np.all(data["actions"][:, 1] >= 0)
        assert np.all(data["actions"][:, 1] < config.n_aim_angles)
        assert np.all(data["actions"][:, 2] < 2)
        assert np.all(data["actions"][:, 3] < 2)

    def test_rewards_array(self, input_dir, output_dir):
        config = EnvConfig()
        frames = [_make_frame(frame=i, kills_this_frame=(1 if i == 2 else 0))
                  for i in range(5)]
        path = os.path.join(input_dir, "rewards.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        data = np.load(created[0])

        assert data["rewards"].shape == (5,)
        assert data["rewards"].dtype == np.float32
        # Frame 2 should have a higher reward (kill bonus)
        assert data["rewards"][2] > data["rewards"][0]

    def test_dones_death_episode(self, input_dir, output_dir):
        """Player dying on last frame sets dones[-1] = True."""
        config = EnvConfig()
        frames = [_make_frame(frame=i, my_health=max(0, 3 - i), maxhealth=8)
                  for i in range(5)]
        frames[-1]["player"]["my_health"] = 0

        path = os.path.join(input_dir, "death.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        data = np.load(created[0])

        assert data["dones"][-1] == True
        assert not np.any(data["dones"][:-1])

    def test_dones_alive_episode(self, input_dir, output_dir):
        """Player alive at end means no dones are True (truncated episode)."""
        config = EnvConfig()
        frames = [_make_frame(frame=i, my_health=8, maxhealth=8)
                  for i in range(5)]
        path = os.path.join(input_dir, "alive.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        data = np.load(created[0])

        assert not np.any(data["dones"])

    def test_obs_values_normalized(self, input_dir, output_dir):
        """Observations should be within [0, 1] (or [-1, 1] for speeds)."""
        config = EnvConfig()
        frames = [_make_frame(frame=0, x=5040.0, y=5040.0, hspeed=3.0,
                              vspeed=-2.0, my_health=6, maxhealth=8)]
        path = os.path.join(input_dir, "norm.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        data = np.load(created[0])
        obs = data["obs"][0]

        # Most features should be in [0, 1], speeds in [-1, 1]
        assert obs[0] >= 0.0 and obs[0] <= 1.0  # x normalized
        assert obs[1] >= 0.0 and obs[1] <= 1.0  # y normalized
        assert obs[2] >= 0.0 and obs[2] <= 1.0  # hp ratio
        assert obs[3] >= -1.0 and obs[3] <= 1.0  # hspeed normalized
        assert obs[4] >= -1.0 and obs[4] <= 1.0  # vspeed normalized


# ---------------------------------------------------------------------------
# 7. Multi-episode conversion from a single file
# ---------------------------------------------------------------------------

class TestMultiEpisodeConversion:
    def test_frame_counter_reset(self, input_dir, output_dir):
        """Frame counter resetting to 0 should split into two episodes."""
        config = EnvConfig()
        ep1 = [_make_frame(frame=i, my_health=5) for i in range(4)]
        ep2 = [_make_frame(frame=i, my_health=10) for i in range(3)]

        path = os.path.join(input_dir, "multi_reset.jsonl")
        _write_jsonl(ep1 + ep2, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        assert len(created) == 2

        data1 = np.load(created[0])
        data2 = np.load(created[1])
        assert data1["obs"].shape[0] == 4
        assert data2["obs"].shape[0] == 3

    def test_health_jump_boundary(self, input_dir, output_dir):
        """Death followed by health jump (no frame reset) splits episodes."""
        config = EnvConfig()
        ep1 = [_make_frame(frame=i, my_health=max(0, 2 - i), maxhealth=8)
               for i in range(3)]
        ep1[-1]["player"]["my_health"] = 0

        ep2 = [_make_frame(frame=3 + i, my_health=8, maxhealth=8)
               for i in range(4)]

        path = os.path.join(input_dir, "multi_health.jsonl")
        _write_jsonl(ep1 + ep2, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        assert len(created) == 2

        data1 = np.load(created[0])
        assert data1["dones"][-1] == True  # died
        data2 = np.load(created[1])
        assert data2["dones"][-1] == False  # alive at end

    def test_three_episodes_in_one_file(self, input_dir, output_dir):
        config = EnvConfig()
        ep1 = [_make_frame(frame=i, my_health=5) for i in range(3)]
        ep2 = [_make_frame(frame=i, my_health=8) for i in range(4)]
        ep3 = [_make_frame(frame=i, my_health=10) for i in range(2)]

        path = os.path.join(input_dir, "triple.jsonl")
        _write_jsonl(ep1 + ep2 + ep3, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        assert len(created) == 3


# ---------------------------------------------------------------------------
# 8. Directory conversion
# ---------------------------------------------------------------------------

class TestDirectoryConversion:
    def test_converts_all_jsonl_files(self, input_dir, output_dir):
        config = EnvConfig()
        for idx in range(3):
            frames = [_make_frame(frame=i, my_health=5)
                      for i in range(4 + idx)]
            path = os.path.join(input_dir, f"ntt_demo_{idx:04d}.jsonl")
            _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        total = converter.convert_directory(input_dir, output_dir)
        assert total == 3

        npz_files = sorted(f for f in os.listdir(output_dir)
                           if f.endswith(".npz"))
        assert len(npz_files) == 3

    def test_empty_directory(self, input_dir, output_dir):
        converter = NTTLogConverter()
        total = converter.convert_directory(input_dir, output_dir)
        assert total == 0

    def test_ignores_non_jsonl_files(self, input_dir, output_dir):
        """Only .jsonl files should be processed."""
        config = EnvConfig()
        # Write one valid .jsonl
        frames = [_make_frame(frame=i) for i in range(3)]
        _write_jsonl(frames, os.path.join(input_dir, "valid.jsonl"))

        # Write a .txt file that should be ignored
        with open(os.path.join(input_dir, "notes.txt"), "w") as f:
            f.write("not a jsonl file")

        converter = NTTLogConverter(config=config)
        total = converter.convert_directory(input_dir, output_dir)
        assert total == 1


# ---------------------------------------------------------------------------
# 9. Validation of output .npz files
# ---------------------------------------------------------------------------

class TestValidation:
    def test_validate_passes_on_valid_output(self, input_dir, output_dir, capsys):
        config = EnvConfig()
        frames = [_make_frame(frame=i, my_health=max(0, 5 - i), maxhealth=8,
                              moving=True, move_dir=90, aim_dir=45, fire=True)
                  for i in range(6)]
        frames[-1]["player"]["my_health"] = 0

        path = os.path.join(input_dir, "validate.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        converter.convert_file(path, output_dir)
        converter.validate(output_dir)

        captured = capsys.readouterr()
        assert "Validation passed" in captured.out
        assert "6" in captured.out  # total transitions

    def test_validate_empty_dir(self, output_dir, capsys):
        converter = NTTLogConverter()
        converter.validate(output_dir)
        captured = capsys.readouterr()
        assert "No .npz files found" in captured.out


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_file_produces_no_output(self, input_dir, output_dir):
        path = os.path.join(input_dir, "empty.jsonl")
        with open(path, "w"):
            pass
        converter = NTTLogConverter()
        created = converter.convert_file(path, output_dir)
        assert created == []

    def test_single_frame_episode(self, input_dir, output_dir):
        """A file with exactly one frame should produce one episode with 1 transition."""
        config = EnvConfig()
        frames = [_make_frame(frame=0, my_health=8)]
        path = os.path.join(input_dir, "single_frame.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        assert len(created) == 1

        data = np.load(created[0])
        assert data["obs"].shape[0] == 1
        assert data["actions"].shape == (1, 4)
        assert data["dones"].shape == (1,)

    def test_truncated_episode_done_never_true(self, input_dir, output_dir):
        """If player is alive at end of file, dones should all be False."""
        config = EnvConfig()
        frames = [_make_frame(frame=i, my_health=10, maxhealth=10)
                  for i in range(20)]
        path = os.path.join(input_dir, "truncated.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        data = np.load(created[0])

        assert not np.any(data["dones"])

    def test_no_enemies(self, input_dir, output_dir):
        """Frames with no enemies should produce zero-padded enemy features."""
        config = EnvConfig()
        frames = [_make_frame(frame=i, enemies=[]) for i in range(3)]
        path = os.path.join(input_dir, "no_enemies.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        data = np.load(created[0])

        # Enemy features (indices 12 onwards) should all be zero
        enemy_features = data["obs"][:, config.player_features:]
        np.testing.assert_array_equal(enemy_features, 0.0)

    def test_max_enemies_exceeded(self, input_dir, output_dir):
        """More enemies than max_enemies should be truncated in observation."""
        config = EnvConfig()
        n_enemies = config.max_enemies + 5
        enemies = [
            {"x": 5000 + i * 10, "y": 5000, "my_health": 4, "maxhealth": 4,
             "type_id": "ref object 14"}
            for i in range(n_enemies)
        ]
        frames = [_make_frame(frame=0, enemies=enemies)]
        path = os.path.join(input_dir, "many_enemies.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        data = np.load(created[0])

        # obs should still be (1, obs_dim), not larger
        assert data["obs"].shape == (1, config.obs_dim)

    def test_all_malformed_lines_produces_no_output(self, input_dir, output_dir):
        """A file where every line is malformed should produce no output."""
        path = os.path.join(input_dir, "all_bad.jsonl")
        _write_raw_ntt_jsonl([
            "not json at all",
            "{broken",
            "}{}{}{",
        ], path)

        converter = NTTLogConverter()
        created = converter.convert_file(path, output_dir)
        assert created == []

    def test_no_human_action_field(self, input_dir, output_dir):
        """Frames missing human_action should still convert with default actions."""
        config = EnvConfig()
        frame = _make_frame(frame=0)
        del frame["human_action"]
        path = os.path.join(input_dir, "no_action.jsonl")
        _write_jsonl([frame], path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        assert len(created) == 1
        data = np.load(created[0])
        # With empty human_action, discretize_action returns defaults
        assert data["actions"].shape == (1, 4)

    def test_no_reward_signals_field(self, input_dir, output_dir):
        """Frames missing reward_signals should still convert with base reward."""
        config = EnvConfig()
        frame = _make_frame(frame=0)
        del frame["reward_signals"]
        path = os.path.join(input_dir, "no_reward.jsonl")
        _write_jsonl([frame], path)

        converter = NTTLogConverter(config=config)
        created = converter.convert_file(path, output_dir)
        assert len(created) == 1
        data = np.load(created[0])
        # Should get survival reward only
        assert data["rewards"][0] == pytest.approx(config.reward_survival_per_step)

    def test_convert_does_not_mutate_input_frames(self, input_dir, output_dir):
        """Converter should not mutate the original frame dicts."""
        config = EnvConfig()
        frames = [_make_frame(frame=0, my_health=8, maxhealth=8)]
        # Save originals for comparison
        import copy
        originals = copy.deepcopy(frames)

        path = os.path.join(input_dir, "nomutate.jsonl")
        _write_jsonl(frames, path)

        converter = NTTLogConverter(config=config)
        # Parse and convert directly to test _convert_episode
        parsed = converter._parse_jsonl(path)
        original_parsed = copy.deepcopy(parsed)
        converter._convert_episode(parsed)

        # The parsed frames should not be mutated
        assert parsed == original_parsed
