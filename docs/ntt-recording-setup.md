# NTT Recording Setup — Capturing Human Demos from Official Nuclear Throne

## Prerequisites

- Nuclear Throne on Steam
- Windows PC (NTT does not support macOS natively — see [macOS section](#macos) below)

## 1. Install Nuclear Throne Together

NTT is distributed as a Steam beta branch. No separate download required.

1. Right-click **Nuclear Throne** in your Steam library
2. **Properties → Betas** → select `ntt_development - NTT test build`
3. Steam downloads the NTT build (~50MB update)
4. Launch the game — the title screen should show "Nuclear Throne Together"

## 2. Deploy the Recorder Mod

1. Find your Nuclear Throne install directory:
   - Steam → right-click Nuclear Throne → **Properties → Installed Files → Browse**
2. Create a `mods` folder in that directory if it doesn't exist
3. Copy `ntt_mods/nt_recorder.mod.gml` from this repo into the `mods` folder:

```
Nuclear Throne/
├── mods/
│   └── nt_recorder.mod.gml    ← place here
├── NuclearThroneTogether.exe
└── ...
```

## 3. Load the Mod

Launch Nuclear Throne Together, then:

1. Press **T** to open in-game chat
2. Type `/loadmod nt_recorder` and press Enter
3. A red dot and "REC" indicator appears in the top-left corner — recording is active

**Auto-load on startup (optional):** Create `mods/startup.txt` with the line:
```
/loadmod nt_recorder
```

## 4. Record Gameplay

Just play the game normally. The mod records in the background:

- A red dot in the top-left shows recording status with episode and frame count
- Each time you die, the episode is saved as a `.jsonl` file
- Data also flushes every ~10 seconds as a safety net against crashes
- There is no noticeable performance impact

**Recommended recording goals:**
- At least 3–5 hours of gameplay
- Play through multiple areas (at least areas 1–3) for observation coverage
- Use varied strategies (different weapons, characters) for action diversity
- Don't worry about dying — deaths are valuable terminal state data

## 5. Locate the Recorded Files

NTT writes mod files to a per-mod data directory under the game's appdata location:

- **Windows:** `%LOCALAPPDATA%\nuclearthrone\` (typically `C:\Users\<you>\AppData\Local\nuclearthrone\`)

Look for files named `ntt_demo_TIMESTAMP_NNNN.jsonl`. If you can't find them, search your system for `ntt_demo_*.jsonl`.

**Tip:** Run `/gmlapi` in the NTT chat to generate an API reference at `%LOCALAPPDATA%\nuclearthrone\api\` — this confirms the appdata path on your system.

## 6. Convert to Training Data

Copy the `.jsonl` files to your development machine (this repo), then run:

```bash
# Convert all NTT logs to .npz training format
python -m nt_rl.bc.ntt_converter --input /path/to/jsonl/files --output demonstrations

# Verify the dataset
python -m nt_rl.bc.ntt_converter --input /path/to/jsonl/files --output demonstrations --validate

# Check what was produced
ls demonstrations/*.npz
```

The converter:
- Maps NTT variable names (`my_health` → `hp`, `maxhealth` → `max_hp`, etc.)
- Encodes observations to the same 112-float vector the RL agent uses
- Discretizes continuous human input to `MultiDiscrete([9, 24, 2, 2])` actions
- Computes rewards from raw signal deltas using `EnvConfig` weights
- Splits multi-episode files automatically

## 7. Train

Once demonstrations are converted:

```bash
python -m nt_rl.bc.train --demo-dir demonstrations
```

## macOS

NTT does not run natively on macOS. Options:

1. **WINE / CrossOver** — NTT works under WINE, but you must install Windows Steam inside the WINE prefix to get the Windows game build (which includes `data.win`). This is functional but adds friction.

2. **Windows PC or VM** — Use a separate Windows machine for recording. Only the `.jsonl` files need to be transferred back.

3. **Skip NTT, use the rebuild** — Record demos from the nt-recreated-public rebuild instead (runs natively on macOS). Place `agent_record.txt` in the game's working directory and run:
   ```bash
   python -m nt_rl.bc.recorder --port 7777
   ```
   This uses the socket bridge rather than NTT file logging.

The rebuild recorder and NTT recorder produce identical `.npz` files. Demonstrations from both sources can be mixed in the same `demonstrations/` directory for training.
