"""File-to-TCP adapter bridging the NTT agent bridge mod to NuclearThroneEnv.

The NTT mod communicates via file-based IPC (writing/reading JSON files in the
game's data directory). This adapter translates those file operations to the
newline-delimited TCP/JSON protocol that NuclearThroneEnv expects.

Architecture:
    NTT Mod (official NT) <-> File IPC <-> ntt_bridge_adapter.py <-> TCP <-> NuclearThroneEnv

Usage:
    python -m nt_rl.ntt_bridge_adapter --ipc-dir /path/to/nt/data --port 7777
"""

import argparse
import json
import os
import selectors
import signal
import socket
import sys
import time
from pathlib import Path


# IPC file names (must match nt_agent_bridge.mod.gml)
STATE_FILE = "agent_state.json"
STATE_READY = "agent_state.ready"
ACTION_FILE = "agent_action.json"
ACTION_READY = "agent_action.ready"

# Polling interval in seconds (1ms for fast enough 30 FPS interaction)
POLL_INTERVAL = 0.001

# How long to wait for the env to send data before checking files again (seconds)
TCP_RECV_TIMEOUT = 0.001


class NTTBridgeAdapter:
    """Bridges file-based IPC from the NTT mod to a TCP socket for NuclearThroneEnv."""

    def __init__(self, ipc_dir: str, port: int = 7777, host: str = "localhost"):
        self.ipc_dir = Path(ipc_dir)
        self.port = port
        self.host = host

        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._recv_buffer = ""
        self._running = False
        self._selector = selectors.DefaultSelector()

    @property
    def state_file(self) -> Path:
        return self.ipc_dir / STATE_FILE

    @property
    def state_ready(self) -> Path:
        return self.ipc_dir / STATE_READY

    @property
    def action_file(self) -> Path:
        return self.ipc_dir / ACTION_FILE

    @property
    def action_ready(self) -> Path:
        return self.ipc_dir / ACTION_READY

    def start(self):
        """Start the TCP server and main bridge loop."""
        self._cleanup_ipc_files()
        self._start_server()
        self._running = True

        print(f"NTT Bridge Adapter listening on {self.host}:{self.port}",
              file=sys.stderr)
        print(f"IPC directory: {self.ipc_dir}", file=sys.stderr)
        print("Waiting for NuclearThroneEnv to connect...", file=sys.stderr)

        try:
            self._accept_and_run()
        except KeyboardInterrupt:
            print("\nShutting down...", file=sys.stderr)
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown: close sockets, delete IPC files."""
        self._running = False
        self._close_client()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        self._selector.close()
        self._cleanup_ipc_files()
        print("NTT Bridge Adapter stopped.", file=sys.stderr)

    def _start_server(self):
        """Bind and listen on the TCP port."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.setblocking(False)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)

    def _accept_and_run(self):
        """Accept client connections and run the bridge loop for each."""
        while self._running:
            # Wait for a client connection (NuclearThroneEnv)
            self._client_socket = self._wait_for_client()
            if self._client_socket is None:
                continue

            print(f"NuclearThroneEnv connected from "
                  f"{self._client_socket.getpeername()}", file=sys.stderr)
            self._recv_buffer = ""

            try:
                self._bridge_loop()
            except (ConnectionError, BrokenPipeError, OSError) as e:
                print(f"Connection lost: {e}", file=sys.stderr)
            finally:
                self._close_client()
                print("Client disconnected, waiting for reconnection...",
                      file=sys.stderr)

    def _wait_for_client(self) -> socket.socket | None:
        """Block until a client connects or we're told to stop."""
        sel = selectors.DefaultSelector()
        sel.register(self._server_socket, selectors.EVENT_READ)
        try:
            while self._running:
                events = sel.select(timeout=0.5)
                for key, _ in events:
                    client, addr = self._server_socket.accept()
                    client.setblocking(False)
                    return client
        finally:
            sel.close()
        return None

    def _bridge_loop(self):
        """Main loop: shuttle data between IPC files and TCP socket.

        Flow each iteration:
        1. Check for state file from NTT mod -> forward to TCP client
        2. Check for data from TCP client -> write to action file for NTT mod
        """
        while self._running:
            did_work = False

            # 1. Check if the NTT mod has written a state file
            if self.state_ready.exists():
                state_json = self._read_and_delete_state()
                if state_json is not None:
                    self._tcp_send(state_json + "\n")
                    did_work = True

            # 2. Check for incoming data from NuclearThroneEnv
            action_json = self._tcp_recv_line()
            if action_json is not None:
                self._write_action(action_json)
                did_work = True

            # Avoid busy-spinning when idle
            if not did_work:
                time.sleep(POLL_INTERVAL)

    def _read_and_delete_state(self) -> str | None:
        """Read the state JSON file and delete both state files.

        Returns the raw JSON string, or None if reading fails.
        """
        try:
            state_json = self.state_file.read_text(encoding="utf-8").strip()
            # Delete files: data first, then sentinel
            try:
                self.state_file.unlink()
            except FileNotFoundError:
                pass
            try:
                self.state_ready.unlink()
            except FileNotFoundError:
                pass
            return state_json
        except (FileNotFoundError, OSError) as e:
            # Race condition: sentinel existed but data file gone
            try:
                self.state_ready.unlink()
            except FileNotFoundError:
                pass
            print(f"Warning: failed to read state file: {e}", file=sys.stderr)
            return None

    def _write_action(self, action_json: str):
        """Write action JSON to IPC files (data first, then sentinel)."""
        self.action_file.write_text(action_json, encoding="utf-8")
        self.action_ready.write_text("1", encoding="utf-8")

    def _tcp_send(self, data: str):
        """Send data to the connected TCP client."""
        if self._client_socket is None:
            raise ConnectionError("No client connected")
        self._client_socket.sendall(data.encode("utf-8"))

    def _tcp_recv_line(self) -> str | None:
        """Try to read one newline-delimited message from the TCP client.

        Returns the line (without newline) or None if no complete line available.
        Non-blocking: returns immediately if no data.
        """
        if self._client_socket is None:
            return None

        # Try to read available data
        try:
            data = self._client_socket.recv(65536)
            if not data:
                raise ConnectionError("Client disconnected")
            self._recv_buffer += data.decode("utf-8")
        except BlockingIOError:
            pass  # No data available right now
        except (ConnectionError, OSError):
            raise

        # Check if we have a complete line
        if "\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split("\n", 1)
            return line.strip()

        return None

    def _close_client(self):
        """Close the client socket."""
        if self._client_socket is not None:
            try:
                self._client_socket.close()
            except OSError:
                pass
            self._client_socket = None
        self._recv_buffer = ""

    def _cleanup_ipc_files(self):
        """Delete all IPC files (safe to call even if they don't exist)."""
        for path in [self.state_file, self.state_ready,
                     self.action_file, self.action_ready]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Bridge NTT agent mod file IPC to TCP for NuclearThroneEnv"
    )
    parser.add_argument(
        "--ipc-dir", required=True,
        help="Directory where NTT writes IPC files (game's save/data directory)"
    )
    parser.add_argument(
        "--port", type=int, default=7777,
        help="TCP port for NuclearThroneEnv to connect (default: 7777)"
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Host to bind TCP server (default: localhost)"
    )
    args = parser.parse_args()

    # Validate IPC directory exists
    ipc_path = Path(args.ipc_dir)
    if not ipc_path.is_dir():
        print(f"Error: IPC directory does not exist: {args.ipc_dir}",
              file=sys.stderr)
        sys.exit(1)

    adapter = NTTBridgeAdapter(
        ipc_dir=args.ipc_dir,
        port=args.port,
        host=args.host,
    )

    # Handle SIGINT/SIGTERM gracefully
    def _signal_handler(signum, frame):
        adapter.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    adapter.start()


if __name__ == "__main__":
    main()
