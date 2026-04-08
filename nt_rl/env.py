"""NuclearThroneEnv — Gymnasium-compatible environment wrapping the GML UDP bridge."""

import json
import socket
import sys
import time

import gymnasium
import numpy as np

from nt_rl.config import EnvConfig
from nt_rl.obs_utils import encode_observation


# Direction angles for 8-directional movement (index 0-7), index 8 = no movement
_MOVE_DIRS = [0, 45, 90, 135, 180, 225, 270, 315]


class NuclearThroneEnv(gymnasium.Env):
    """Gymnasium environment for Nuclear Throne RL training via UDP bridge."""

    metadata = {"render_modes": []}

    def __init__(self, port: int = 7777, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.port = port

        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.config.obs_dim,),
            dtype=np.float32,
        )

        # [move_dir (0-8), aim_bin (0-23), shoot (0-1), special (0-1)]
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [self.config.n_move_dirs, self.config.n_aim_angles, 2, 2]
        )

        self._socket: socket.socket | None = None
        self._step_count = 0
        self._game_addr = (self.config.host, self.port)
        self._prev_cumulative_reward = 0.0  # For computing reward deltas

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_cumulative_reward = 0.0

        try:
            if self._socket is None:
                self._create_socket()

            # Drain any stale datagrams
            self._drain()

            # Send reset. The game may need time to process (room_restart).
            # Keep sending pings until we get a state back, since the game
            # needs a received datagram to know our address.
            self._send_json({"type": "reset"})

            deadline = time.time() + self.config.socket_timeout
            noop = json.dumps({"type": "action", "move_dir": 0, "moving": False,
                               "aim_dir": 0, "fire": False, "spec": False})
            while time.time() < deadline:
                try:
                    state = self._recv_state(timeout=1.0)
                    obs = encode_observation(state, self.config)
                    info = {"game": state.get("game", {}), "frame": state.get("frame", 0)}
                    return obs, info
                except (TimeoutError, OSError):
                    # Re-send a ping so the game knows our address after room_restart
                    self._socket.sendto(noop.encode(), self._game_addr)

            raise TimeoutError("No state received after reset")

        except Exception as e:
            print(f"NuclearThroneEnv: reset error: {e}", file=sys.stderr)
            return np.zeros(self.config.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1

        try:
            action_dict = self._decode_action(action)
            self._send_json(action_dict)
            state = self._recv_state()

            obs = encode_observation(state, self.config)
            # Reward is cumulative on the GML side — compute delta
            cumulative = float(state.get("reward", 0.0))
            reward = cumulative - self._prev_cumulative_reward
            self._prev_cumulative_reward = cumulative
            terminated = bool(state.get("done", False))
            truncated = self._step_count >= self.config.max_steps
            info = {
                "game": state.get("game", {}),
                "frame": state.get("frame", 0),
                "mutation_screen": state.get("mutation_screen", False),
            }

            return obs, reward, terminated, truncated, info

        except (TimeoutError, OSError) as e:
            print(f"NuclearThroneEnv: error at step {self._step_count}: {e}",
                  file=sys.stderr)
            return (
                np.zeros(self.config.obs_dim, dtype=np.float32),
                0.0, True, False,
                {"error": str(e)},
            )
        except (json.JSONDecodeError, ValueError) as e:
            print(f"NuclearThroneEnv: parse error at step {self._step_count}: {e}",
                  file=sys.stderr)
            return (
                np.zeros(self.config.obs_dim, dtype=np.float32),
                0.0, True, False,
                {"error": str(e)},
            )

    def close(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def _decode_action(self, action) -> dict:
        """Convert MultiDiscrete action to GML JSON action format."""
        move_idx = int(action[0])
        aim_idx = int(action[1])
        shoot = bool(action[2])
        spec = bool(action[3])

        moving = move_idx < 8
        move_dir = _MOVE_DIRS[move_idx] if moving else 0
        aim_dir = aim_idx * (360.0 / self.config.n_aim_angles)

        return {
            "type": "action",
            "move_dir": move_dir,
            "moving": moving,
            "aim_dir": aim_dir,
            "fire": shoot,
            "spec": spec,
            "swap": False,
            "pick": False,
        }

    def _create_socket(self):
        """Create a UDP socket. No connection needed."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        sock.settimeout(self.config.step_timeout)
        self._socket = sock
        print(f"NuclearThroneEnv: UDP socket ready, target {self._game_addr}",
              file=sys.stderr)

    def _send_json(self, data: dict):
        """Send a JSON message as a UDP datagram."""
        if self._socket is None:
            raise ConnectionError("Socket not created")
        msg = json.dumps(data).encode("utf-8")
        self._socket.sendto(msg, self._game_addr)

    def _recv_state(self, timeout: float | None = None) -> dict:
        """Receive the latest state datagram, draining any older ones.

        UDP datagrams are complete messages — no fragmentation. We drain
        all queued datagrams and return the most recent one.
        """
        if self._socket is None:
            raise ConnectionError("Socket not created")

        self._socket.settimeout(timeout or self.config.step_timeout)

        # Block until at least one datagram arrives
        data, _addr = self._socket.recvfrom(65536)
        latest = data

        # Drain any additional queued datagrams (non-blocking) to get the latest
        self._socket.setblocking(False)
        try:
            while True:
                data, _addr = self._socket.recvfrom(65536)
                latest = data
        except (BlockingIOError, OSError):
            pass
        self._socket.setblocking(True)

        return json.loads(latest.decode("utf-8").strip())

    def _drain(self):
        """Drain all queued UDP datagrams."""
        if self._socket is None:
            return
        self._socket.setblocking(False)
        try:
            while True:
                self._socket.recvfrom(65536)
        except (BlockingIOError, OSError):
            pass
        self._socket.setblocking(True)
