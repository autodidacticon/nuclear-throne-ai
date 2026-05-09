# Agent 08 — Mutation Selection (Hierarchical Sub-Policy)

## Goal
Give the RL agent a real ability to choose mutations during level-up screens, instead of the current random-pick fallback. Mutations are the highest-leverage long-term decisions in Nuclear Throne — they shape the rest of the run.

## Background

Mutations in Nuclear Throne are passive abilities you select between levels. Each level-up offers 4 random mutations from a pool of ~30. They fundamentally change playstyle:

- **Rhino Skin**: +50% HP — pure tank
- **Eagle Eyes**: +50% accuracy — sniper builds
- **Hammerhead**: melee damage doubled — melee specialist
- **Lucky Shot**: 20% damage refund as ammo — ammo conservation
- **Plutonium Hunger**: pickups give more rads (XP)
- ... etc.

The choice is **strategic, low-frequency, high-impact**. Picking Rhino Skin early might let you survive a tough boss; picking Eagle Eyes might let your sniper builds shred faster.

## Current State

**The agent doesn't choose mutations.** Period. Here's what's actually happening:

1. **Rebuild bridge** (`AgentBridge/Step_0.gml`): when `LevCont` exists with `SkillIcon` instances, it calls a hardcoded random-pick:
   ```gml
   var _icons = []
   with (SkillIcon) { array_push(_icons, id) }
   if (array_length(_icons) > 0) {
       var _pick = _icons[irandom(array_length(_icons) - 1)]
       with (_pick) { selected = true; event_user(0) }
   }
   ```
2. **NTT recorder**: doesn't capture mutation choices at all (the user clicks with mouse, the recorder only watches keyboard state)
3. **Observation space**: has no information about owned mutations or offered mutations
4. **Action space**: no mutation pick action

This means: across 16M training steps, the agent's mutation choices are **uniformly random**. Whatever build the agent ends up with is pure luck, and the same build pattern is repeated regardless of what's working.

## Why this matters

A skilled human picks mutations based on:
- Current weapon loadout (Hammerhead is useless without melee weapons)
- Current HP situation (Rhino Skin is great if you've been taking damage)
- Current level / planned playstyle (early-game vs late-game priorities)
- What's left in the mutation pool (some are obviously better)

Random selection achieves none of this. The agent's effective skill ceiling is bounded by random mutations regardless of how good the in-combat policy is.

## Strategies (ranked by complexity)

### Strategy 1: Heuristic priority list

Hardcode a tier list. When LevCont appears, pick the highest-tier offered mutation.

**Implementation**: Replace the random pick in `AgentBridge/Step_0.gml` with a tier lookup table.

**Cost**: trivial. ~50 lines of GML.

**Benefit**: dramatically better than random. A reasonable tier list outperforms ~80% of random picks. Doesn't help training quality but improves inference.

**Downside**: ignores context (HP, loadout). Same picks every run.

### Strategy 2: Multi-armed bandit with offline learning

Treat each mutation as an arm. Track average run length / kill count when each mutation is owned. Pick the mutation with the highest expected reward.

**Implementation**:
- Add a Python script that analyzes existing eval episode data
- For each mutation: compute mean episode reward across episodes where the agent owned it
- Use UCB or Thompson sampling for exploration
- Export the resulting priority table to GML

**Cost**: medium. Pure Python analysis + GML lookup.

**Benefit**: data-driven, adaptive over time, no training needed.

**Downside**: ignores context. Same picks every run. Overfits to early-game mutations because they appear more often.

### Strategy 3: Hierarchical sub-policy

Add a separate small neural network ("meta-policy") that runs ONLY when LevCont exists. It takes the current game state + offered mutations as input and outputs a probability over picks.

**Architecture**:
- Input: current player state (HP, weapons, ammo, level, owned mutations as 30-bool vector) + 4 offered mutation IDs as one-hot
- Network: 2-layer MLP, 128 hidden
- Output: 4-way categorical over offered slots
- Trained via REINFORCE or PPO with episode-end reward signal (final kills, area reached)

**Implementation**:
- Add `owned_mutations` (30 bools) to the observation
- Add a new action dim or a separate meta-policy class
- Capture human mutation picks in the recorder for BC seed
- Train the meta-policy with episode-end credit assignment

**Cost**: high. New observation features, new policy class, recorder changes, BC + RL training loop modifications.

**Benefit**: context-aware mutation selection. Adapts to current build. Highest theoretical performance.

**Downside**: data-hungry. Mutations are rare events (4-8 per episode), so credit assignment is hard. May need many more training episodes than a flat RL approach.

### Strategy 4: AlphaStar-style value estimation bandit

DeepMind's approach for similar low-frequency, high-impact decisions. Train a separate value network that estimates "expected long-term reward given this build choice." Pick the option with highest predicted value.

**Difference from Strategy 3**: instead of predicting actions directly, predict VALUES of each option. The actor is just argmax over those values. Easier to train via TD learning since you can use single-step bootstrapping.

**Cost**: high. Same complexity as Strategy 3.

**Benefit**: more sample-efficient than direct policy training for low-frequency decisions.

## Recommended approach

**Phase A (immediate, low-risk)**: Strategy 1 + add owned mutations to observation.

1. Build a tier list of all 30 mutations (rebuild has the full list in `scrSkills.gml`)
2. Replace random pick with tier-based pick in `AgentBridge/Step_0.gml`
3. Add `owned_mutations` (30 bools) to the observation so the in-combat policy at least knows what build it has

This decouples the mutation problem from the training quality problem. Even if mutation selection is heuristic, the in-combat agent benefits from knowing its build.

**Phase B (after Phase A)**: Strategy 2 (data-driven priority) — refine the tier list using eval data.

**Phase C (long-term)**: Strategy 3 or 4 (learned hierarchical sub-policy) — only if mutation strategy becomes the dominant bottleneck after the in-combat policy is solid.

## Tasks for Phase A

1. **Catalog mutations**: read `nt-recreated-public/scripts/scrSkills/scrSkills.gml` and any related files. Build a complete list of mutation IDs and names.

2. **Hand-craft a tier list**: based on community knowledge, assign each mutation a quality score (1-10). Reasonable defaults:
   - S-tier (10): Rhino Skin, Bloodlust, Lucky Shot
   - A-tier (8): Eagle Eyes, Hammerhead, Stress, Throne Butt
   - B-tier (6): Boiling Veins, Rabbit Paw, Strong Spirit
   - C-tier (4): Trigger Fingers, Long Arms, Open Mind
   - D-tier (2): everything else

   Save as `nt-recreated-public/scripts/scr_agent_mutation_tiers/scr_agent_mutation_tiers.gml`.

3. **Update the auto-pick logic** in `AgentBridge/Step_0.gml`:
   ```gml
   if (instance_exists(LevCont) && instance_exists(SkillIcon)) {
       var _best_id = noone
       var _best_tier = -1
       with (SkillIcon) {
           var _t = scr_agent_mutation_tier(skill)  // returns 1-10
           if (_t > _best_tier) {
               _best_tier = _t
               _best_id = id
           }
       }
       if (_best_id != noone) {
           with (_best_id) { selected = true; event_user(0) }
       }
   }
   ```

4. **Add owned mutations to obs**:
   - GML: in `scr_agent_build_state.gml`, iterate over the player's `skill[]` array (or whatever NT calls it) and emit a 30-bool array
   - Python: add 30 floats to the player feature section. Update `obs_utils.py` and `config.py`.
   - obs_dim grows from 240 to 270

5. **Test rebuilds and re-train BC** with the new obs dim.

## Capturing human mutation picks (for Phase C)

Required if you ever want to learn mutation selection from demonstrations:

- The NTT recorder needs to detect when LevCont closes and which mutation was added
- Compare `Player.skill` array before/after the LevCont disappears
- Emit a `mutation_pick` field in the JSONL with the chosen mutation ID
- The converter and BC pipeline need to consume this

Defer this until Phase C is actually needed. Phase A doesn't require it.

## Success criteria for Phase A

- Mutation auto-pick is no longer random — picks the highest-tier available mutation
- Owned mutations are in the observation
- BC retrains successfully at obs_dim 270
- Trained agent shows visible build coherence (e.g., picks Rhino Skin if available, not Trigger Fingers as the 4th choice)

## Out of scope

- Mouse position tracking in the recorder (would let us capture human mutation clicks but isn't needed for Phase A)
- Modifying the NT mutation pool itself (cheating)
- Per-character mutation tier lists (Robot has different optimal builds than Crystal — defer to Phase C)
