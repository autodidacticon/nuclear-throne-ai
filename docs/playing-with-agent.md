# Playing Nuclear Throne with a Trained Agent

A behavioral cloning (BC) policy trained on human Nuclear Throne gameplay can take control of the player character. This document covers how to set it up with either the open-source rebuild or the official game via Nuclear Throne Together (NTT).

## Architecture

There are two paths to connect the agent to the game. Both end with the same TCP protocol on port 7777:

```
Official NT path:
  Nuclear Throne + NTT mod  <-->  Files  <-->  ntt_bridge_adapter.py  <-->  TCP:7777  <-->  play.py

Rebuild path:
  nt-recreated-public (GML bridge)  <-->  TCP:7777  <-->  play.py
```

## Prerequisites

- **Python 3.12+** with project dependencies installed:
  ```bash
  uv pip install -r requirements.txt
  ```
  Or with standard pip:
  ```bash
  pip install -r requirements.txt
  ```
- **A trained checkpoint** -- the best BC checkpoint is at `checkpoints/bc_run_c/final` (a [256,256] tanh network, 70.4% action accuracy, 0.636 val loss). PPO checkpoints saved via SB3 also work.
- **The game**, either:
  - The open-source rebuild (`nt-recreated-public`) built in GameMaker Studio 2, OR
  - Official Nuclear Throne on Steam with Nuclear Throne Together (NTT) installed

## Option A: Playing with the Rebuild

The rebuild has a built-in TCP socket bridge (`obj_AgentBridge`) that communicates directly with the Python environment.

### Steps

1. **Open the rebuild project** in GameMaker Studio 2:
   - Open `nt-recreated-public/NuclearThrone.yyp`

2. **Enable agent mode** by placing a file named `agent_mode.txt` in the game's working directory. The contents of the file do not matter -- the game only checks for its presence.
   ```bash
   touch nt-recreated-public/datafiles/agent_mode.txt
   ```

3. **Build and run** the game for macOS (or your target platform). The GML bridge will start a TCP server on port 7777 and wait for a connection.

4. **Run the agent** in a terminal:
   ```bash
   python3 scripts/play.py --port 7777 --stats
   ```

5. The agent should connect and begin controlling the player. You will see "AGENT BRIDGE [CONNECTED]" in the game window.

### Notes

- The rebuild can run faster than 30 FPS, which means faster training/evaluation cycles.
- You can run multiple instances on different ports (7777, 7778, etc.) for parallel evaluation.

## Option B: Playing with Official Nuclear Throne (NTT)

The official game does not have a built-in TCP bridge. Instead, an NTT mod writes game state to files, and a Python adapter translates that file IPC into TCP messages.

### Step 1: Install Nuclear Throne Together

1. Download NTT from [https://yellowafterlife.itch.io/nuclear-throne-together](https://yellowafterlife.itch.io/nuclear-throne-together)
2. Follow NTT's installation instructions to patch your Nuclear Throne installation
3. Verify NTT works by launching Nuclear Throne and checking for the NTT overlay

### Step 2: Install the Agent Bridge Mod

Copy the agent bridge mod into NTT's mods directory:

```bash
cp ntt_mods/nt_agent_bridge.mod.gml /path/to/NuclearThrone/mods/
```

The mods directory location depends on your platform:
- **macOS**: `~/Library/Application Support/com.vlambeer.nuclearthrone/mods/` or alongside the game executable
- **Windows**: typically in the Nuclear Throne installation directory under `mods/`
- **Linux**: `~/.local/share/NuclearThrone/mods/` or alongside the game executable

### Step 3: Find the IPC Directory

The NTT mod reads and writes JSON files for communication. These files are written to Nuclear Throne's save/data directory. Common locations:

- **macOS**: `~/Library/Application Support/com.vlambeer.nuclearthrone/`
- **Windows**: `%LOCALAPPDATA%/nuclearthrone/`
- **Linux**: `~/.local/share/NuclearThrone/`

You can verify you have the right directory by looking for `nuclearthrone.sav` or other game save files there.

### Step 4: Start the Bridge Adapter

In a terminal, start the file-to-TCP bridge adapter:

```bash
python3 -m nt_rl.ntt_bridge_adapter --ipc-dir /path/to/ipc --port 7777
```

Replace `/path/to/ipc` with the actual IPC directory from Step 3. The adapter will:
- Start a TCP server on port 7777
- Wait for the agent script to connect
- Poll the IPC directory for game state files and relay them over TCP
- Write action files back for the NTT mod to read

### Step 5: Launch the Game and Load the Mod

1. Launch Nuclear Throne (with NTT active)
2. In the NTT console (press the designated key to open it), load the mod:
   ```
   /load nt_agent_bridge
   ```
3. You should see "nt_agent_bridge loaded" in the console and "AGENT BRIDGE [WAITING]" on screen

### Step 6: Run the Agent

In another terminal:

```bash
python3 scripts/play.py --port 7777 --stats
```

The agent will connect through the bridge adapter to the game. The HUD indicator should change to "AGENT BRIDGE [CONNECTED]".

## play.py Usage Reference

```
python3 scripts/play.py [OPTIONS]

Options:
  --checkpoint PATH     Path to SB3 policy checkpoint
                        (default: checkpoints/bc_run_c/final)
  --port PORT           Game bridge TCP port (default: 7777)
  --episodes N          Number of episodes to play (default: 0 = infinite)
  --deterministic       Use deterministic action selection (default)
  --stochastic          Use stochastic action selection (samples from the
                        policy distribution instead of taking the argmax)
  --render-actions      Print each action to stdout (for debugging)
  --stats               Print per-episode statistics (reward, length, kills,
                        levels) and a session summary on exit
```

### Examples

Play indefinitely with stats:
```bash
python3 scripts/play.py --stats
```

Play 10 episodes with a specific checkpoint:
```bash
python3 scripts/play.py --checkpoint checkpoints/bc_run_c/final --episodes 10 --stats
```

Debug action output:
```bash
python3 scripts/play.py --render-actions --episodes 1
```

Use stochastic sampling (adds randomness, may help with diversity):
```bash
python3 scripts/play.py --stochastic --stats
```

Press Ctrl+C at any time to stop. The script will print a session summary with cumulative statistics before exiting.

## Troubleshooting

### Connection refused

The game is not running, or the port does not match.
- **Rebuild**: make sure the game is running and `agent_mode.txt` is present
- **NTT**: make sure `ntt_bridge_adapter.py` is running and listening on the correct port

### Agent stands still / no actions applied

- **NTT**: verify the mod is loaded (`/load nt_agent_bridge` in NTT console)
- **NTT**: verify the IPC directory path is correct -- the adapter should print "IPC directory: /path" on startup
- Check that the bridge adapter shows "NuclearThroneEnv connected" after the agent script starts

### Agent only moves in one direction

This was a known bug with early demonstration data where action encoding was incorrect. If you see this with newer checkpoints, verify the checkpoint was trained on data converted with the latest `ntt_converter.py`.

### NTT button_set not working

Some NTT versions have issues with `button_set()` for certain inputs. The mod uses `button_set()` for movement and shooting. If the agent connects but inputs are not applying:
- Try a different version of NTT
- Check the NTT console for error messages
- As a fallback, the mod could be modified to write directly to player variables instead of using `button_set()`

### Policy fails to load

- Make sure the checkpoint file exists (default: `checkpoints/bc_run_c/final`)
- The checkpoint must be saved via SB3's `policy.save()` (zip format). PPO checkpoints saved via `model.save()` require loading with `PPO.load()` instead -- modify the script accordingly if using a full PPO model.

### High latency / slow response

- The NTT file IPC path is inherently slower than the rebuild's direct TCP -- expect roughly 30 FPS
- The bridge adapter polls at 1ms intervals; if the game runs faster, actions may lag by a frame or two
- The rebuild's direct TCP connection has lower latency

## Known Limitations

- **Limited level coverage**: the BC policy was trained on human gameplay from levels 1-7 only. It has not seen later levels, loop content, or most bosses.
- **No special ability usage**: the training data contains ~99% not-special actions, so the policy almost never uses the special ability.
- **30 FPS cap with official NT**: the NTT file IPC path is limited to roughly 30 FPS. The rebuild can run faster.
- **No mutation selection**: the policy does not handle mutation selection screens. The game will pause on those screens until manually advanced.
- **Single character**: the training data was collected with a specific character (likely Crystal or Fish). The policy may not generalize well to other characters.
