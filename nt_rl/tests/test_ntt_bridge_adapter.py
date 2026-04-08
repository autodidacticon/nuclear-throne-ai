"""Tests for the NTT file-to-TCP bridge adapter.

Tests the adapter using mock files (no game needed). The adapter translates
between file-based IPC (NTT mod) and TCP (NuclearThroneEnv).
"""

import json
import os
import socket
import tempfile
import threading
import time

import pytest

from nt_rl.ntt_bridge_adapter import NTTBridgeAdapter, STATE_FILE, STATE_READY, ACTION_FILE, ACTION_READY


def _get_free_port():
    """Find a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture
def ipc_dir():
    """Create a temporary directory for IPC files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def adapter_port():
    return _get_free_port()


@pytest.fixture
def adapter(ipc_dir, adapter_port):
    """Create and start an adapter in a background thread."""
    a = NTTBridgeAdapter(ipc_dir=ipc_dir, port=adapter_port)
    a._cleanup_ipc_files()
    a._start_server()
    a._running = True
    # Run the accept+bridge loop in a background thread
    t = threading.Thread(target=_run_adapter_loop, args=(a,), daemon=True)
    t.start()
    yield a
    a._running = False
    a.stop()
    t.join(timeout=3.0)


def _run_adapter_loop(adapter: NTTBridgeAdapter):
    """Run the adapter's accept-and-run loop, catching expected errors."""
    try:
        adapter._accept_and_run()
    except OSError:
        pass  # Expected when server socket is closed during shutdown


def _connect_tcp(port: int, timeout: float = 5.0) -> socket.socket:
    """Connect a TCP client to the adapter, retrying until success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(("localhost", port))
            return sock
        except (ConnectionRefusedError, OSError):
            sock.close()
            time.sleep(0.05)
    raise RuntimeError(f"Could not connect to adapter on port {port}")


def _tcp_send(sock: socket.socket, data: dict):
    """Send a newline-delimited JSON message over TCP."""
    msg = json.dumps(data) + "\n"
    sock.sendall(msg.encode("utf-8"))


def _tcp_recv(sock: socket.socket, timeout: float = 3.0) -> dict:
    """Receive one newline-delimited JSON message from TCP."""
    sock.settimeout(timeout)
    buf = ""
    while "\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk.decode("utf-8")
    line, _ = buf.split("\n", 1)
    return json.loads(line)


def _write_state_files(ipc_dir: str, state: dict):
    """Simulate the NTT mod writing state files."""
    state_path = os.path.join(ipc_dir, STATE_FILE)
    ready_path = os.path.join(ipc_dir, STATE_READY)
    with open(state_path, "w") as f:
        json.dump(state, f)
    with open(ready_path, "w") as f:
        f.write("1")


def _read_action_files(ipc_dir: str, timeout: float = 3.0) -> dict:
    """Wait for and read the action files written by the adapter."""
    ready_path = os.path.join(ipc_dir, ACTION_READY)
    action_path = os.path.join(ipc_dir, ACTION_FILE)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(ready_path):
            with open(action_path, "r") as f:
                data = json.loads(f.read())
            os.unlink(action_path)
            os.unlink(ready_path)
            return data
        time.sleep(0.01)
    raise TimeoutError("Action files not written within timeout")


def _make_mock_state(frame: int = 0, done: bool = False, reward: float = 0.01) -> dict:
    """Create a minimal valid game state matching NuclearThroneEnv protocol."""
    return {
        "type": "state",
        "frame": frame,
        "done": done,
        "reward": reward,
        "player": {
            "x": 5000.0, "y": 5000.0,
            "hp": 8, "max_hp": 8,
            "hspeed": 0.0, "vspeed": 0.0,
            "gunangle": 90.0,
            "wep": 1, "bwep": 5,
            "ammo": [0, 20, 8, 0, 0, 0],
            "reload": 0, "can_shoot": True, "roll": False,
        },
        "enemies": [],
        "game": {
            "area": 1, "subarea": 1, "level": 1,
            "loops": 0, "kills": 0, "hard": 0,
        },
        "mutation_screen": False,
    }


class TestTCPServer:
    """Test that the adapter accepts TCP connections."""

    def test_server_starts_and_accepts_connection(self, adapter, adapter_port):
        sock = _connect_tcp(adapter_port)
        try:
            assert sock.fileno() != -1
        finally:
            sock.close()

    def test_server_accepts_after_client_disconnect(self, adapter, adapter_port):
        # First connection
        sock1 = _connect_tcp(adapter_port)
        sock1.close()
        time.sleep(0.2)

        # Second connection should also work
        sock2 = _connect_tcp(adapter_port)
        try:
            assert sock2.fileno() != -1
        finally:
            sock2.close()


class TestFileToTCP:
    """Test state file -> TCP forwarding."""

    def test_state_forwarded_to_tcp_client(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)  # Let adapter register client

            state = _make_mock_state(frame=42, reward=5.01)
            _write_state_files(ipc_dir, state)

            received = _tcp_recv(sock)
            assert received["frame"] == 42
            assert received["player"]["hp"] == 8
            assert received["type"] == "state"
        finally:
            sock.close()

    def test_state_files_deleted_after_read(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            _write_state_files(ipc_dir, _make_mock_state())
            _tcp_recv(sock)  # Wait for adapter to process

            time.sleep(0.1)  # Give adapter time to clean up
            assert not os.path.exists(os.path.join(ipc_dir, STATE_FILE))
            assert not os.path.exists(os.path.join(ipc_dir, STATE_READY))
        finally:
            sock.close()

    def test_multiple_states_forwarded_in_order(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            for i in range(3):
                _write_state_files(ipc_dir, _make_mock_state(frame=i))
                received = _tcp_recv(sock)
                assert received["frame"] == i
        finally:
            sock.close()

    def test_done_state_forwarded(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            state = _make_mock_state(done=True, reward=-15.0)
            state["player"]["hp"] = 0
            _write_state_files(ipc_dir, state)

            received = _tcp_recv(sock)
            assert received["done"] is True
            assert received["reward"] == -15.0
        finally:
            sock.close()


class TestTCPToFile:
    """Test TCP -> action file forwarding."""

    def test_action_written_to_files(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            action = {
                "type": "action",
                "move_dir": 90,
                "moving": True,
                "aim_dir": 45.0,
                "fire": True,
                "spec": False,
                "swap": False,
                "pick": False,
            }
            _tcp_send(sock, action)

            received = _read_action_files(ipc_dir)
            assert received["type"] == "action"
            assert received["move_dir"] == 90
            assert received["fire"] is True
        finally:
            sock.close()

    def test_action_sentinel_created(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            _tcp_send(sock, {"type": "action", "move_dir": 0, "moving": False,
                             "aim_dir": 0, "fire": False, "spec": False,
                             "swap": False, "pick": False})

            # Wait for files to appear
            ready_path = os.path.join(ipc_dir, ACTION_READY)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if os.path.exists(ready_path):
                    break
                time.sleep(0.01)

            assert os.path.exists(ready_path)
            assert os.path.exists(os.path.join(ipc_dir, ACTION_FILE))
        finally:
            sock.close()


class TestResetMessage:
    """Test reset command forwarding."""

    def test_reset_written_to_action_file(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            _tcp_send(sock, {"type": "reset"})
            received = _read_action_files(ipc_dir)
            assert received["type"] == "reset"
        finally:
            sock.close()


class TestRoundTrip:
    """Test full round-trip: action -> state cycle."""

    def test_action_then_state_round_trip(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            # 1. Send action from "env"
            action = {
                "type": "action",
                "move_dir": 45,
                "moving": True,
                "aim_dir": 180.0,
                "fire": False,
                "spec": False,
                "swap": False,
                "pick": False,
            }
            _tcp_send(sock, action)

            # 2. Simulate NTT mod reading action and writing state
            action_data = _read_action_files(ipc_dir)
            assert action_data["move_dir"] == 45

            state = _make_mock_state(frame=1, reward=0.01)
            _write_state_files(ipc_dir, state)

            # 3. Env should receive state
            received = _tcp_recv(sock)
            assert received["frame"] == 1
        finally:
            sock.close()

    def test_multiple_steps(self, adapter, adapter_port, ipc_dir):
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            for step in range(5):
                # Send action
                _tcp_send(sock, {"type": "action", "move_dir": 0, "moving": True,
                                 "aim_dir": step * 15.0, "fire": step % 2 == 0,
                                 "spec": False, "swap": False, "pick": False})

                # Read action from file and respond with state
                _read_action_files(ipc_dir)
                _write_state_files(ipc_dir, _make_mock_state(frame=step))

                # Verify state received
                received = _tcp_recv(sock)
                assert received["frame"] == step
        finally:
            sock.close()


class TestStaleFileCleanup:
    """Test that stale files from previous sessions are cleaned up."""

    def test_stale_files_cleaned_on_start(self, ipc_dir, adapter_port):
        # Create stale files before adapter starts
        for fname in [STATE_FILE, STATE_READY, ACTION_FILE, ACTION_READY]:
            with open(os.path.join(ipc_dir, fname), "w") as f:
                f.write("stale")

        a = NTTBridgeAdapter(ipc_dir=ipc_dir, port=adapter_port)
        a._cleanup_ipc_files()

        for fname in [STATE_FILE, STATE_READY, ACTION_FILE, ACTION_READY]:
            assert not os.path.exists(os.path.join(ipc_dir, fname))


class TestPartialFileHandling:
    """Test edge cases with IPC files."""

    def test_sentinel_without_data_file(self, adapter, adapter_port, ipc_dir):
        """Sentinel exists but data file is missing -- adapter should not crash."""
        sock = _connect_tcp(adapter_port)
        try:
            time.sleep(0.1)

            # Write only the sentinel, no data file
            ready_path = os.path.join(ipc_dir, STATE_READY)
            with open(ready_path, "w") as f:
                f.write("1")

            # Give adapter time to try reading (should handle gracefully)
            time.sleep(0.2)

            # Sentinel should be cleaned up
            assert not os.path.exists(ready_path)

            # Adapter should still be functional -- write a proper state
            _write_state_files(ipc_dir, _make_mock_state(frame=99))
            received = _tcp_recv(sock)
            assert received["frame"] == 99
        finally:
            sock.close()


class TestShutdownCleanup:
    """Test that shutdown cleans up IPC files."""

    def test_stop_deletes_ipc_files(self, ipc_dir, adapter_port):
        a = NTTBridgeAdapter(ipc_dir=ipc_dir, port=adapter_port)
        a._start_server()
        a._running = True

        # Write some IPC files as if in mid-operation
        for fname in [STATE_FILE, ACTION_FILE]:
            with open(os.path.join(ipc_dir, fname), "w") as f:
                f.write('{"test": true}')

        a.stop()

        for fname in [STATE_FILE, STATE_READY, ACTION_FILE, ACTION_READY]:
            assert not os.path.exists(os.path.join(ipc_dir, fname))
