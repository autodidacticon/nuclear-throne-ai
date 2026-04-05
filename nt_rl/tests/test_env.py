"""Tests for NuclearThroneEnv against the mock socket server."""

import time

import gymnasium
import numpy as np
import pytest

from nt_rl.config import EnvConfig
from nt_rl.env import NuclearThroneEnv
from nt_rl.tests.mock_server import MockNuclearThroneServer


def _get_free_port():
    """Find a free port for testing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_port():
    return _get_free_port()


@pytest.fixture
def mock_server(mock_port):
    server = MockNuclearThroneServer(port=mock_port, max_steps=50)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def env(mock_port, mock_server):
    config = EnvConfig(socket_timeout=5.0, step_timeout=2.0)
    e = NuclearThroneEnv(port=mock_port, config=config)
    yield e
    e.close()


class TestReset:
    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.config.obs_dim,)
        assert obs.dtype == np.float32

    def test_reset_returns_info_dict(self, env):
        obs, info = env.reset()
        assert isinstance(info, dict)


class TestStep:
    def test_step_returns_correct_shape(self, env):
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.config.obs_dim,)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space_sample_is_valid(self, env):
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (env.config.obs_dim,)
            if terminated:
                env.reset()

    def test_observation_space_bounds(self, env):
        env.reset()
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs), (
                f"Obs out of bounds: min={obs.min()}, max={obs.max()}"
            )
            if terminated:
                env.reset()

    def test_episode_terminates_on_done(self, env):
        env.reset()
        terminated = False
        for _ in range(200):  # Mock server terminates at 50 steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated


class TestErrorResilience:
    def test_env_survives_socket_disconnect(self, mock_port):
        server = MockNuclearThroneServer(port=mock_port, max_steps=100)
        server.disconnect_after_steps = 5
        server.start()

        config = EnvConfig(socket_timeout=3.0, step_timeout=2.0)
        e = NuclearThroneEnv(port=mock_port, config=config)

        try:
            e.reset()
            terminated = False
            for i in range(20):
                action = e.action_space.sample()
                obs, reward, terminated, truncated, info = e.step(action)
                if terminated:
                    break
            # Should not raise — env handles disconnect gracefully
            assert terminated  # env sets terminated=True on connection error
        finally:
            e.close()
            server.stop()

    def test_env_survives_malformed_json(self, mock_port):
        server = MockNuclearThroneServer(port=mock_port, max_steps=100)
        server.send_malformed_json = True
        server.start()

        config = EnvConfig(socket_timeout=3.0, step_timeout=2.0)
        e = NuclearThroneEnv(port=mock_port, config=config)

        try:
            e.reset()
            terminated = False
            for i in range(10):
                action = e.action_space.sample()
                obs, reward, terminated, truncated, info = e.step(action)
                if terminated:
                    break
            # Should not raise — env handles malformed JSON gracefully
        finally:
            e.close()
            server.stop()


class TestGymnasiumCompliance:
    def test_gymnasium_api_compliance(self, mock_port, mock_server):
        config = EnvConfig(socket_timeout=5.0, step_timeout=2.0)
        e = NuclearThroneEnv(port=mock_port, config=config)
        try:
            from gymnasium.utils.env_checker import check_env
            check_env(e, skip_render_check=True)
        finally:
            e.close()
