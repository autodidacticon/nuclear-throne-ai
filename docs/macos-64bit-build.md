# macOS 64-bit Standalone Build Analysis

## 1. Current State

### Architecture

The GameMaker runtime (2024.14.4.268) ships a **universal binary** Mac_Runner that includes both x86_64 and arm64 slices:

```
Mac_Runner: Mach-O universal binary with 2 architectures:
  [x86_64: Mach-O 64-bit executable x86_64]
  [arm64:  Mach-O 64-bit executable arm64]
```

All bundled frameworks are also universal:
- `libYoYoGamepad.dylib` — universal (x86_64 + arm64)
- `libYoYoIAP.dylib` — universal (x86_64 + arm64)

The Igor CLI itself is native arm64. The `Application Oven.app` (used during packaging) is also universal.

### Build mode

The current workflow (`scripts/gm_build.sh run`) uses Igor with `--runtime=VM mac Run`. This:
1. Compiles GML scripts into bytecode via the Asset Compiler
2. Copies bytecode + assets into a temp directory
3. Launches the shared `YoYo Runner.app` from the runtime directory, which interprets the bytecode

The game runs natively on Apple Silicon via the arm64 slice of Mac_Runner. There are **zero 32-bit components** anywhere in the toolchain.

### Project settings (options_mac.yy)

| Setting | Value | Notes |
|---|---|---|
| `option_mac_arm64` | `true` | arm64 slice included |
| `option_mac_x86_64` | `true` | x86_64 slice included |
| `option_mac_min_version` | `"10.10"` | Yosemite minimum |
| `option_mac_disable_sandbox` | `false` | Sandbox enabled |
| `option_mac_allow_incoming_network` | `false` | **Needs changing for agent bridge** |
| `option_mac_allow_outgoing_network` | `false` | **Needs changing for agent bridge** |
| `option_mac_enable_steam` | `false` | No Steam dependency |
| `option_mac_signing_identity` | `"Developer ID Application:"` | **Needs ad-hoc signing** |

## 2. Feasibility Verdict: YES

A standalone 64-bit macOS `.app` bundle can be built from the rebuild. In fact, the `PackageZip` command already produces one. During this analysis, a package build was successfully executed:

```
$ scripts/gm_build.sh package
```

This produced:
- **`nuclearthrone_exe.app`** (209 MB) — a self-contained .app bundle at `~/GameMaker-Studio/nuclearthrone_exe/GM_MAC/nuclearthronemobile/`
- **`nuclearthronemobile.zip`** (145 MB) — the same app as a zip archive
- The app contains:
  - `Contents/MacOS/Mac_Runner` — universal binary (x86_64 + arm64)
  - `Contents/Resources/game.ios` — compiled game bytecode
  - `Contents/Resources/` — all game assets (sounds, textures, language files)
  - `Contents/Resources/agent_mode.txt` — the agent bridge activation file
  - `Contents/Frameworks/libYoYoGamepad.dylib` — gamepad support

The build failed only at the code-signing step because no Developer ID certificate is installed. This was resolved with ad-hoc signing:

```bash
codesign --force --deep --sign - nuclearthrone_exe.app
```

The app does **not** require GameMaker IDE to run. It is fully self-contained.

## 3. Steps Required

### 3.1 Fix macOS options for agent bridge networking

The agent bridge uses UDP sockets (`network_create_socket_ext(network_socket_udp, ...)`). The macOS sandbox options must allow network access.

Edit `nt-recreated-public/options/mac/options_mac.yy`:
```json
"option_mac_allow_incoming_network": true,
"option_mac_allow_outgoing_network": true,
```

Alternatively, disable the sandbox entirely:
```json
"option_mac_disable_sandbox": true,
```

### 3.2 Update the build script

The existing `package` command in `scripts/gm_build.sh` needs correct output paths and an ad-hoc signing step. Updated command:

```bash
package)
    echo "==> Packaging as zip..."
    "$IGOR" "${COMMON_ARGS[@]}" --temp="$TEMP" -j=8 --runtime=VM \
      --of="$HOME/GameMaker-Studio/nuclearthrone_exe/GM_MAC/nuclearthronemobile/NuclearThrone" \
      --tf="$HOME/GameMaker-Studio/nuclearthrone_exe/GM_MAC/nuclearthronemobile/NuclearThrone.zip" \
      mac PackageZip

    echo "==> Ad-hoc signing..."
    codesign --force --deep --sign - \
      "$HOME/GameMaker-Studio/nuclearthrone_exe/GM_MAC/nuclearthronemobile/nuclearthrone_exe.app"
    ;;
```

### 3.3 Fix the signing identity

Change `option_mac_signing_identity` in `options_mac.yy` to empty string `""` or to `"-"` (ad-hoc) to prevent the certificate lookup error. Or leave it and handle signing manually after the build.

### 3.4 Parallel instances for training

To run N parallel copies for RL training, each instance needs:
1. A separate copy of the `.app` bundle (or at minimum, a separate `agent_mode.txt` with the correct port number)
2. Each instance must bind to a unique UDP port

Strategy for parallel deployment:
```bash
for i in $(seq 0 $((N-1))); do
    PORT=$((7777 + i))
    APP_DIR="/tmp/nt_instance_$i"
    cp -R nuclearthrone_exe.app "$APP_DIR/NuclearThrone.app"
    # Write port config into the app's Resources
    echo "$PORT" > "$APP_DIR/NuclearThrone.app/Contents/Resources/agent_port.txt"
    # Launch
    open -n "$APP_DIR/NuclearThrone.app"
done
```

Since each `.app` is ~209 MB, running 4 parallel instances requires ~836 MB of disk. This is manageable.

## 4. Blockers

### 4.1 No showstoppers found

All identified issues have straightforward solutions:

| Issue | Severity | Solution |
|---|---|---|
| Code signing fails (no Developer ID cert) | Low | Use ad-hoc signing (`codesign --force --deep --sign -`) |
| Network options set to `false` | Low | Set `allow_incoming_network` and `allow_outgoing_network` to `true` |
| Sandbox may block localhost UDP | Low | Set `option_mac_disable_sandbox: true` or enable network entitlements |

### 4.2 Extensions are not a problem

The project uses three extensions:
1. **`execute_shell_simple_ext`** — Windows-only DLL (PE32/PE32+). Only used by `scrWindowOpenSecondary` for multi-window on Windows. Has a GML wrapper that calls the DLL. On macOS, the DLL is simply not loaded (TargetMask: 6 = Windows only). Not needed for RL training.
2. **`native_cursor_ext`** — Has a `.dll` reference but the actual DLL files are `.gitignore`d. The extension provides GML-only fallbacks. The game already runs on macOS without native cursor DLLs.
3. **`YYExtra`** — Android-only Java extension (copyToTargets: 8 = Android). Has GML fallbacks. Irrelevant on macOS.

No extensions require native macOS `.dylib` files that are missing.

### 4.3 License

The GameMaker license (`licence.plist`) is type "Personal" with `Mac.build_module` feature enabled, expiring `2026-05-05`. This license explicitly supports macOS builds. The `ci_build` feature is also present, which may be relevant for headless/automated builds.

## 5. Recommendation

### Best path forward for RL training

1. **Enable network entitlements** — Set `option_mac_allow_incoming_network` and `option_mac_allow_outgoing_network` to `true` in `options_mac.yy`. Consider disabling sandbox entirely for training builds.

2. **Use `PackageZip` for standalone builds** — The command `mac PackageZip` (already in `gm_build.sh`) produces a fully self-contained `.app` that runs without GameMaker IDE. After fixing the output path issue and adding ad-hoc signing, this becomes a single-command operation.

3. **Continue using `Run` for development** — The `mac Run` workflow is faster for iteration since it skips packaging and uses the shared runtime. Use `PackageZip` only when creating distributable copies for parallel training.

4. **Parallel training deployment** — Copy the `.app` bundle N times with per-instance port configuration. Each 209 MB instance runs natively on arm64 without Rosetta overhead. The M4 Max should handle 4-8 parallel instances comfortably.

5. **No architecture changes needed** — The game already builds as a universal binary with native arm64 support. The current `option_mac_arm64: true` and `option_mac_x86_64: true` settings produce a universal binary. For training-only builds, you could set `option_mac_x86_64: false` to reduce binary size, but this is optional.

### Summary

The Nuclear Throne rebuild **already builds and runs as a native 64-bit macOS application**. The `PackageZip` Igor command produces a standalone `.app` bundle that:
- Runs without GameMaker IDE installed
- Runs natively on Apple Silicon (arm64) without Rosetta
- Can be copied for parallel training instances
- Includes the RL agent bridge (agent_mode.txt + compiled GML bridge code)

The only required changes are enabling network entitlements and handling code signing, both of which are configuration-only fixes requiring no code changes.
