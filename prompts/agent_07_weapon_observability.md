# Agent 07 — Weapon Observability and Selection

## Goal
Give the RL agent the information it needs to make intelligent weapon decisions: what weapon it's holding, what's available to pick up, and when swapping is beneficial.

## Background

Nuclear Throne is a roguelike where weapon choice dominates moment-to-moment strategy. Players juggle two weapon slots and constantly decide whether to swap to a better weapon dropped by an enemy or in a pickup chest. The current RL agent has minimal weapon awareness:

**What the obs already has:**
- `wep` (current weapon ID, normalized to [0, 1] by max_weapon_id=128) at index 6
- `bwep` (backup weapon ID) is exported by the rebuild but NOT currently in the obs vector
- `ammo[]` total normalized at index 7 (sum across all 6 ammo types)
- `reload_norm` and `can_shoot` flags at indices 8-9

**What's missing:**
- The backup weapon is invisible to the agent
- The agent has no idea what weapons are on the floor (pickups, dropped from enemies, in chests)
- Per-ammo-type counts are aggregated into one number — the agent can't tell if it's out of bullets but full on shells
- No notion of "this weapon is melee, that one is ranged" — weapon ID alone gives no semantic information
- No reward signal for picking up better weapons

**What's already wired up:**
- The action space already has `swap` (binary) and `pick` (binary) — the agent CAN swap and pick up
- The bridge already triggers these via `KeyCont.press_swap` / `KeyCont.press_pick`
- The agent just doesn't have enough information to use them intelligently

## Strategies (ranked by complexity)

### Strategy 1: Minimal — expose backup weapon and per-ammo-type ammo

Smallest possible change. Just add fields the bridge already collects.

**GML side**: backup weapon ID and the full 6-element ammo array are already in `scr_agent_build_state.gml` — they're passed through but not used by the encoder.

**Python side**:
- Add `bwep_norm` to obs (1 float)
- Add 6 per-type ammo counts to obs (6 floats)
- Total: +7 features → obs_dim grows from 240 to 247

**Cost**: trivial. ~10 lines of Python.

**Benefit**: agent knows its full inventory. Doesn't help with pickups at all.

### Strategy 2: Add nearest weapon pickups to observation

Expose the nearest 3-5 weapon pickups in the world as variable-length entities (similar to how enemies are handled).

**GML side**:
- In `scr_agent_build_state.gml`, iterate over `WepPickup` (or whatever the parent class is) and emit nearest N
- Per pickup: relative x/y to player, weapon ID, distance, "type tag" (melee/ranged/explosive/heavy — derive from weapon ID via lookup table)

**Python side**:
- Add a new entity set: 5 pickup slots × 5 features = 25 floats
- Update `DeepSetsExtractor` to add a third pooled set: enemies + projectiles + pickups
- DeepSets output grows from 147 features to 147 + 32 + 32 = 211

**Cost**: medium. ~50 lines GML + ~30 lines Python + extractor update + BC retraining.

**Benefit**: agent can reason about "should I walk to that pickup?" Enables strategic weapon hunting.

### Strategy 3: Add weapon-quality reward shaping

Reward the agent for picking up weapons that are objectively better than what it's currently holding. Requires a "weapon quality" lookup table.

**GML side** (in `scr_agent_compute_reward.gml`):
- Track previous-frame weapon IDs
- On weapon swap or pickup, compare new weapon's quality tier vs old
- Reward proportional to quality delta: `+1.0` for upgrade, `-0.5` for downgrade (player might pick up by accident)

**Quality table**: hand-craft a tier list mapping weapon ID → tier (1-5). Nuclear Throne has ~100 weapons. The community has a well-known tier list.

**Cost**: medium. The quality table is ~100 lines but is one-time work.

**Benefit**: makes the kill reward less dominant. Encourages active inventory management.

### Strategy 4: Encode weapon type as one-hot semantic vector

Replace the single `wep` ID float with a one-hot vector encoding weapon TYPE rather than identity.

**Categories** (~10):
- Melee
- Pistol
- Shotgun
- SMG / machine gun
- Rifle / sniper
- Bow / crossbow
- Grenade launcher
- Rocket launcher
- Special (laser cannon, flamethrower, etc.)
- None / empty

**GML side**: lookup table mapping weapon ID → category (1-10).

**Python side**: replace 1 weapon ID float with 10 one-hot floats. Same for backup.

**Cost**: medium. Categorization is hand-crafted.

**Benefit**: agent learns weapon-type-specific strategies (melee = approach, sniper = kite). Generalizes across specific weapons within a category.

### Strategy 5: Full strategic inventory — combine 1+2+3+4

Do everything: per-ammo counts, pickups in obs, weapon-quality rewards, and one-hot weapon types.

**Cost**: high. Requires rebuilding obs encoding, retraining BC.

**Benefit**: complete weapon awareness. Agent can answer "should I switch from my reloading shotgun to the pistol with full ammo and an enemy 30px away?"

## Recommended approach

Start with **Strategy 1+4** as a single PR:
- Adds 7 features (backup weapon one-hot + per-ammo counts) and changes the existing weapon ID to a one-hot type vector
- Total obs change: ~16 features added, 1 removed → obs_dim grows from 240 to ~255
- Single BC retrain
- Establishes the lookup tables that Strategies 2/3 will need later

If after PPO training the agent shows weapon-related failure modes (e.g., refuses to swap, ignores pickups), add Strategy 2 in a follow-up.

## Tasks

1. Define a weapon ID → category lookup table (`nt_rl/weapon_categories.py`) covering all NT weapons. Use the official weapon enum from `nt-recreated-public/scripts/scrWeapons/`.

2. Update the GML state builder to expose the backup weapon and per-ammo array (the data is already collected, just needs to be in the JSON output).

3. Update `nt_rl/obs_utils.py` to:
   - Replace `wep / max_weapon_id` (1 float) with a 10-dim one-hot weapon type vector
   - Add a 10-dim one-hot for backup weapon
   - Add 6 per-ammo-type floats normalized by max_ammo
   - Update the docstring

4. Update `nt_rl/config.py` to bump `player_features` accordingly.

5. Run the test suite and fix any obs dimension hardcoding.

6. Document the new layout.

## Out of scope (defer to a future cycle)

- Strategy 2 (pickups in obs) — adds complexity to DeepSets extractor
- Strategy 3 (quality rewards) — add only if weapon-related failures persist after Strategy 1+4
- Mouse-tracking for human pickup events in the recorder — current keyboard tracking doesn't capture mouse-driven weapon swaps. The bridge can detect swaps post-hoc by watching `wep` change between frames.

## Success criteria

- Trained PPO agent shows >5% weapon swap rate (currently <1% because the agent doesn't know swapping exists)
- Agent visibly walks toward weapon pickups in eval rollouts (qualitative check)
- BC training remains stable with the new obs dimensions
