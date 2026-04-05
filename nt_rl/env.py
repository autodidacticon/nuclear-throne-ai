"""NuclearThroneEnv — Gymnasium-compatible environment wrapping the GML TCP socket bridge."""

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
    """Gymnasium environment for Nuclear Throne RL training via TCP socket bridge."""

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
        self._recv_buffer = ""
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._recv_buffer = ""

        try:
            if self._socket is None:
                self._connect()

            self._send_json({"type": "reset"})
            state = self._recv_json()
            obs = encode_observation(state, self.config)
            info = {"game": state.get("game", {}), "frame": state.get("frame", 0)}
            return obs, info

        except Exception as e:
            print(f"NuclearThroneEnv: reset error: {e}", file=sys.stderr)
            self._close_socket()
            return np.zeros(self.config.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1

        try:
            action_dict = self._decode_action(action)
            self._send_json(action_dict)
            state = self._recv_json()

            obs = encode_observation(state, self.config)
            reward = float(state.get("reward", 0.0))
            terminated = bool(state.get("done", False))
            truncated = self._step_count >= self.config.max_steps
            info = {
                "game": state.get("game", {}),
                "frame": state.get("frame", 0),
                "mutation_screen": state.get("mutation_screen", False),
            }

            return obs, reward, terminated, truncated, info

        except (ConnectionError, TimeoutError, OSError) as e:
            print(f"NuclearThroneEnv: connection error at step {self._step_count}: {e}",
                  file=sys.stderr)
            self._close_socket()
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
        self._close_socket()

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

    def _connect(self, max_retries: int = 10, retry_delay: float = 3.0):
        """Connect to the GML socket bridge with retry logic."""
        for attempt in range(1, max_retries + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.socket_timeout)
                sock.connect((self.config.host, self.port))
                self._socket = sock
                self._recv_buffer = ""
                print(f"NuclearThroneEnv: connected to {self.config.host}:{self.port}",
                      file=sys.stderr)
                return
            except (ConnectionRefusedError, TimeoutError, OSError) as e:
                print(f"NuclearThroneEnv: connect attempt {attempt}/{max_retries} "
                      f"failed: {e}", file=sys.stderr)
                if attempt < max_retries:
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"Failed to connect to {self.config.host}:{self.port} "
            f"after {max_retries} attempts"
        )

    def _send_json(self, data: dict):
        """Send a newline-delimited JSON message."""
        if self._socket is None:
            raise ConnectionError("Socket not connected")
        msg = json.dumps(data) + "\n"
        self._socket.sendall(msg.encode("utf-8"))

    def _recv_json(self) -> dict:
        """Read one newline-delimited JSON message from the socket."""
        if self._socket is None:
            raise ConnectionError("Socket not connected")

        self._socket.settimeout(self.config.step_timeout)

        while "\n" not in self._recv_buffer:
            try:
                chunk = self._socket.recv(65536)
            except socket.timeout:
                raise TimeoutError("No state received within step_timeout")

            if not chunk:
                raise ConnectionError("Socket closed by remote")

            self._recv_buffer += chunk.decode("utf-8")

        line, self._recv_buffer = self._recv_buffer.split("\n", 1)
        return json.loads(line)

    def _close_socket(self):
        """Close socket and reset state."""
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        self._recv_buffer = ""
