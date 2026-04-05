"""Mock Nuclear Throne GML socket server for testing.

Simulates the GML bridge without requiring Docker or a running game.
Uses the exact JSON schema from GML_BRIDGE_SUMMARY.md.

Two modes:
  - Agent mode (default): expects action messages from client, responds with state.
  - Recording mode: streams state with human_action, no client input expected.
"""

import json
import math
import random
import socket
import threading
import time


class MockNuclearThroneServer:
    """Fake GML socket server that responds with plausible game state."""

    def __init__(self, port: int = 7777, max_steps: int = 50):
        self.port = port
        self.max_steps = max_steps
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        # Configurable behaviors for testing edge cases
        self.send_malformed_json = False
        self.disconnect_after_steps = None  # Set to int to disconnect after N steps

    def start(self):
        """Start the mock server in a background thread."""
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=5.0)

    def stop(self):
        """Stop the mock server."""
        self._stop_event.set()
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)
        self._server_socket.bind(("localhost", self.port))
        self._server_socket.listen(1)
        self._ready_event.set()

        while not self._stop_event.is_set():
            try:
                client, _ = self._server_socket.accept()
                client.settimeout(1.0)
                self._handle_client(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_client(self, client: socket.socket):
        recv_buffer = ""
        step = 0
        player_x = 5000.0
        player_y = 5000.0
        player_hp = 8
        kills = 0
        area = 1
        subarea = 1
        done = False

        try:
            while not self._stop_event.is_set():
                # Read incoming message
                try:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    recv_buffer += chunk.decode("utf-8")
                except socket.timeout:
                    continue

                while "\n" in recv_buffer:
                    line, recv_buffer = recv_buffer.split("\n", 1)
                    if not line.strip():
                        continue

                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "reset":
                        step = 0
                        player_x = 5000.0
                        player_y = 5000.0
                        player_hp = 8
                        kills = 0
                        area = 1
                        subarea = 1
                        done = False

                    elif msg_type == "action":
                        step += 1
                        # Simulate simple movement
                        if msg.get("moving", False):
                            import math
                            move_dir = msg.get("move_dir", 0)
                            player_x += 3.0 * math.cos(math.radians(move_dir))
                            player_y -= 3.0 * math.sin(math.radians(move_dir))
                            player_x = max(0, min(player_x, 10080))
                            player_y = max(0, min(player_y, 10080))

                        # Simulate kills occasionally
                        if msg.get("fire", False) and random.random() < 0.1:
                            kills += 1

                        # Simulate damage occasionally
                        if random.random() < 0.02 and player_hp > 0:
                            player_hp -= 1

                        done = player_hp <= 0 or step >= self.max_steps

                    # Check for test disconnect behavior
                    if (self.disconnect_after_steps is not None
                            and step >= self.disconnect_after_steps):
                        client.close()
                        return

                    # Build and send state
                    reward = 0.01  # survival reward
                    if kills > 0 and msg_type == "action":
                        if msg.get("fire", False) and random.random() < 0.1:
                            reward += 5.0

                    # Malformed JSON test mode
                    if self.send_malformed_json and step == 3:
                        client.sendall(b"NOT VALID JSON{{{}\n")
                        continue

                    n_enemies = random.randint(0, 5)
                    enemies = []
                    for _ in range(n_enemies):
                        enemies.append({
                            "x": player_x + random.uniform(-200, 200),
                            "y": player_y + random.uniform(-200, 200),
                            "hp": random.randint(1, 10),
                            "max_hp": 10,
                            "hitid": random.randint(1, 20),
                        })

                    state = {
                        "type": "state",
                        "frame": step,
                        "done": done,
                        "reward": reward,
                        "player": {
                            "x": player_x,
                            "y": player_y,
                            "hp": player_hp,
                            "max_hp": 8,
                            "hspeed": 0.0,
                            "vspeed": 0.0,
                            "gunangle": 90.0,
                            "wep": 1,
                            "bwep": 5,
                            "ammo": [0, 20, 8, 0, 0, 0],
                            "reload": 0,
                            "can_shoot": True,
                            "roll": False,
                            "race": 1,
                            "nexthurt": 0,
                            "current_frame": step,
                        },
                        "enemies": enemies,
                        "game": {
                            "area": area,
                            "subarea": subarea,
                            "level": min(step // 10 + 1, 20),
                            "loops": 0,
                            "kills": kills,
                            "hard": 0,
                        },
                        "mutation_screen": False,
                    }

                    response = json.dumps(state) + "\n"
                    client.sendall(response.encode("utf-8"))

        except (ConnectionError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass


class MockRecordingServer:
    """Mock server that simulates recording mode.

    Streams state messages with simulated human_action at ~30 FPS.
    Does not expect any input from the client.
    """

    def __init__(self, port: int = 7777, max_steps: int = 100, n_episodes: int = 3,
                 fps: float = 30.0):
        self.port = port
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.frame_delay = 1.0 / fps
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=5.0)

    def stop(self):
        self._stop_event.set()
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)
        self._server_socket.bind(("localhost", self.port))
        self._server_socket.listen(1)
        self._ready_event.set()

        while not self._stop_event.is_set():
            try:
                client, _ = self._server_socket.accept()
                client.settimeout(1.0)
                self._stream_episodes(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def _stream_episodes(self, client: socket.socket):
        try:
            for ep in range(self.n_episodes):
                if self._stop_event.is_set():
                    break

                player_x, player_y = 5000.0, 5000.0
                player_hp, kills = 8, 0
                aim_dir = 90.0

                for step in range(self.max_steps):
                    if self._stop_event.is_set():
                        break

                    # Simulate human-like input
                    moving = random.random() > 0.2
                    move_dir = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
                    aim_dir = (aim_dir + random.uniform(-30, 30)) % 360
                    firing = random.random() > 0.4

                    if moving:
                        player_x += 3.0 * math.cos(math.radians(move_dir))
                        player_y -= 3.0 * math.sin(math.radians(move_dir))
                        player_x = max(0, min(player_x, 10080))
                        player_y = max(0, min(player_y, 10080))

                    if firing and random.random() < 0.1:
                        kills += 1
                    if random.random() < 0.02 and player_hp > 0:
                        player_hp -= 1

                    done = player_hp <= 0 or step == self.max_steps - 1

                    n_enemies = random.randint(0, 5)
                    enemies = [
                        {
                            "x": player_x + random.uniform(-200, 200),
                            "y": player_y + random.uniform(-200, 200),
                            "hp": random.randint(1, 10),
                            "max_hp": 10,
                            "hitid": random.randint(1, 20),
                        }
                        for _ in range(n_enemies)
                    ]

                    state = {
                        "type": "state",
                        "frame": step,
                        "done": done,
                        "reward": 0.01 + (5.0 if firing and random.random() < 0.05 else 0),
                        "player": {
                            "x": player_x, "y": player_y,
                            "hp": player_hp, "max_hp": 8,
                            "hspeed": 3.0 if moving else 0.0,
                            "vspeed": 0.0,
                            "gunangle": aim_dir, "wep": 1, "bwep": 5,
                            "ammo": [0, 20, 8, 0, 0, 0],
                            "reload": 0, "can_shoot": True,
                            "roll": False, "race": 1, "nexthurt": 0,
                            "current_frame": step,
                        },
                        "enemies": enemies,
                        "game": {
                            "area": 1, "subarea": 1,
                            "level": min(step // 30 + 1, 5),
                            "loops": 0, "kills": kills, "hard": 0,
                        },
                        "mutation_screen": False,
                        "human_action": {
                            "move_dir": float(move_dir),
                            "moving": moving,
                            "aim_dir": float(aim_dir),
                            "fire": firing,
                            "spec": random.random() > 0.95,
                            "swap": False,
                            "pick": False,
                        },
                    }

                    msg = json.dumps(state) + "\n"
                    client.sendall(msg.encode("utf-8"))

                    if done:
                        break

                    time.sleep(self.frame_delay)

        except (ConnectionError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass
