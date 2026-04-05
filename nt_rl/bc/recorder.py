"""Record human gameplay demonstrations for behavioural cloning.

Connects to the GML socket bridge in recording mode and logs
(observation, action, reward, done) tuples as per-episode .npz files.

Usage:
    python -m nt_rl.bc.recorder [--port 7777] [--output demonstrations]
"""

import argparse
import json
import math
import os
import signal
import socket
import sys
import time

import numpy as np

from nt_rl.config import EnvConfig
from nt_rl.obs_utils import encode_observation


# Direction angles matching env.py _MOVE_DIRS
_MOVE_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def discretize_action(human_action: dict, n_aim_angles: int = 24) -> np.ndarray:
    """Convert continuous human action to MultiDiscrete([9, 24, 2, 2])."""
    # Movement: snap to nearest 45-degree bin, or 8 for no movement
    if human_action.get("moving", False):
        move_dir = human_action.get("move_dir", 0) % 360
        move_idx = round(move_dir / 45) % 8
    else:
        move_idx = 8

    # Aim: snap to nearest bin
    aim_dir = human_action.get("aim_dir", 0) % 360
    bin_size = 360.0 / n_aim_angles
    aim_idx = round(aim_dir / bin_size) % n_aim_angles

    shoot = 1 if human_action.get("fire", False) else 0
    special = 1 if human_action.get("spec", False) else 0

    return np.array([move_idx, aim_idx, shoot, special], dtype=np.int32)


class DemonstrationRecorder:
    """Records human gameplay from the GML bridge into .npz demonstration files."""

    def __init__(self, port: int = 7777, output_dir: str = "demonstrations",
                 config: EnvConfig | None = None):
        self.port = port
        self.output_dir = output_dir
        self.config = config or EnvConfig()
        self._socket: socket.socket | None = None
        self._recv_buffer = ""
        self._running = False

        # Current episode accumulators
        self._obs_list: list[np.ndarray] = []
        self._action_list: list[np.ndarray] = []
        self._reward_list: list[float] = []
        self._raw_action_list: list[dict] = []

        # Stats
        self._episode_count = 0
        self._total_frames = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """Main recording loop. Blocks until Ctrl+C."""
        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)

        print(f"Recorder: connecting to bridge on port {self.port}...")
        self._connect()
        print("Recorder: connected. Play the game — press Ctrl+C to stop recording.")

        try:
            while self._running:
                state = self._recv_state()
                if state is None:
                    continue

                if "human_action" not in state:
                    print("Recorder: state missing 'human_action' — is the game in record mode?",
                          file=sys.stderr)
                    continue

                obs = encode_observation(state, self.config)
                action = discretize_action(state["human_action"], self.config.n_aim_angles)
                reward = float(state.get("reward", 0.0))
                done = bool(state.get("done", False))

                self._obs_list.append(obs)
                self._action_list.append(action)
                self._reward_list.append(reward)
                self._raw_action_list.append(state["human_action"])
                self._total_frames += 1

                if done:
                    self._save_episode(terminated=True)

        except (ConnectionError, OSError) as e:
            print(f"\nRecorder: connection lost: {e}")
        finally:
            self._save_episode(terminated=False)
            self._close()
            self._print_summary()

    def _save_episode(self, terminated: bool):
        """Save current episode to an .npz file and reset accumulators."""
        if not self._obs_list:
            return

        n = len(self._obs_list)
        dones = np.zeros(n, dtype=bool)
        if terminated:
            dones[-1] = True

        timestamp = int(time.time())
        filename = f"episode_{timestamp}_{self._episode_count:04d}.npz"
        filepath = os.path.join(self.output_dir, filename)

        np.savez_compressed(
            filepath,
            obs=np.array(self._obs_list, dtype=np.float32),
            actions=np.array(self._action_list, dtype=np.int32),
            rewards=np.array(self._reward_list, dtype=np.float32),
            dones=dones,
        )

        ep_reward = sum(self._reward_list)
        print(f"  Episode {self._episode_count}: {n} frames, "
              f"reward={ep_reward:.1f}, {'terminated' if terminated else 'truncated'} "
              f"-> {filename}")

        self._episode_count += 1
        self._obs_list.clear()
        self._action_list.clear()
        self._reward_list.clear()
        self._raw_action_list.clear()

    def _connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.config.socket_timeout)
        sock.connect((self.config.host, self.port))
        self._socket = sock
        self._recv_buffer = ""

    def _recv_state(self) -> dict | None:
        """Read one newline-delimited JSON state message."""
        if self._socket is None:
            raise ConnectionError("Not connected")

        self._socket.settimeout(1.0)

        while "\n" not in self._recv_buffer:
            try:
                chunk = self._socket.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                raise ConnectionError("Socket closed by remote")
            self._recv_buffer += chunk.decode("utf-8")

        line, self._recv_buffer = self._recv_buffer.split("\n", 1)
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _close(self):
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def _signal_handler(self, signum, frame):
        print("\nRecorder: stopping...")
        self._running = False

    def _print_summary(self):
        print(f"\nRecording complete:")
        print(f"  Episodes:     {self._episode_count}")
        print(f"  Total frames: {self._total_frames}")
        if self._episode_count > 0:
            print(f"  Mean length:  {self._total_frames / self._episode_count:.0f} frames")
        print(f"  Saved to:     {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Record human Nuclear Throne gameplay")
    parser.add_argument("--port", type=int, default=7777, help="Bridge port (default: 7777)")
    parser.add_argument("--output", type=str, default="demonstrations",
                        help="Output directory (default: demonstrations)")
    args = parser.parse_args()

    recorder = DemonstrationRecorder(port=args.port, output_dir=args.output)
    recorder.run()


if __name__ == "__main__":
    main()
