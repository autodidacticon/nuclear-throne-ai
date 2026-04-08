# AlphaStar vs PPO-Pragmatist: Joint Recommendation

**Participants:** alphastar-architect, ppo-pragmatist
**Date:** 2026-04-07
**Context:** Cycle 1 of BC+PPO produced a 10M-step agent exhibiting Robot-special-spam
(44-53% usage vs 0.4% human baseline), corner-camping, BC policy collapse on rare actions,
and a reward-hacking ammo-regeneration loop. This document is the consensus output of a
5-round discussion on what to change for cycle 2.

The discussion proceeded in two phases. First, alphastar-architect drafted the root-cause
analysis and staged plan (Sections "Framing" through "Implementation Order"). Then
ppo-pragmatist reviewed the draft against `prompts/agent_06_ppo_training.md`, the observed
failure history, and the existing demonstration data, and pushed back on three substantive
points. The "Pragmatist Pushback Round" section captures the disagreements and the
resolutions that are reflected in the final staged plan.

## Framing

Nuclear Throne is PvE, not adversarial. Many AlphaStar techniques (League Training,
distributed IMPALA across hundreds of actors, full Population-Based Training) are either
unnecessary or infeasible on a single M4 Max with 8 parallel game instances at ~750 FPS
combined. We adopted AlphaStar ideas *selectively*, prioritizing changes that directly
address the observed failure modes and fit the compute budget.

The discussion converged on the principle: **fix the observation and the reward function
before touching the network architecture.** Every observed failure has at least one
upstream cause in the obs/reward pipeline that is cheaper to fix than a model rewrite.

## Root-Cause Analysis of Cycle 1 Failures

| Failure | Proximate Cause | Root Cause |
|---|---|---|
| Robot special spam (44-53%) | No anchor to BC distribution; PPO drifted freely | No KL regularization against π_SL |
| Corner camping / circling | Optimal given partial obs | Obs contains no projectiles; no portal-approach reward |
| Ammo-regen exploit | `-0.05` per-shot penalty made "don't shoot" optimal | Miscalibrated shaping term in `scr_agent_compute_reward.gml` |
| BC policy collapse on rare actions | Nothing preserves the BC distribution once PPO starts | No KL regularization |
| Agent can't dodge bullets | Obs has enemies but no projectiles | GML `scr_agent_build_state` never collected bullets |

Three of five failures trace back to two root causes: **missing projectiles in obs** and
**no KL regularization**. Both are cheap to fix.

## Agreed Cycle 2 Scope (Consensus)

Execute in the following order so we can attribute wins to specific changes.

### Stage A — Minimal reward cleanup (touch only `scr_agent_compute_reward.gml`)

**Scope was reduced after pragmatist pushback.** Original Stage A proposed deleting
the per-shot penalty and adding a new stationary penalty. Both were rejected (see
Pragmatist Pushback Round #1 and #2). What remains:

1. **Leave the per-shot penalty (−0.05) in place for now.** It is a band-aid but it is
   not the root cause of the ammo-regen loop — the root cause is that Robot's special
   regenerates ammo, which the kill reward then incentivizes spending. Stage D
   (KL-to-BC) is the correct fix for that exploit. Removing the penalty before Stage D
   lands risks *worsening* the spam, not fixing it. Re-evaluate after Stage D: if the
   policy no longer spams special, the penalty can be removed in a follow-up cycle
   with attribution.
2. **Add portal-approach shaping**: small positive reward proportional to
   `(prev_portal_dist - curr_portal_dist)` when the portal is visible and not yet used.
   Scale ~0.01/frame peak so it cannot dominate kill rewards. This addresses the
   separate failure mode where the agent never *reaches* the portal even when the
   level is clear. Unlike a stationary penalty, portal-approach is a positive gradient
   toward the actual objective, which is much less prone to producing a new exploit.
3. **No new stationary penalty.** We already tried survival bonuses and they caused
   corner camping. A stationary *penalty* is in the same family of hand-crafted
   shaping that has failed four times. The whole point of Stage D is to REPLACE this
   kind of shaping with a learned prior over human behavior. Adding another shaping
   term contradicts that principle and repeats a known failure mode.

Stage A is now a single small change (portal-approach shaping). It is NOT expected to
fix the ammo-regen loop on its own — that is Stage D's job.

### Stage B — Projectile observations (GML + Python)

1. **GML side (`scr_agent_build_state.gml`):** collect up to 20 projectiles via
   `with (projectile)` — emit `[x, y, hspeed, vspeed, team, damage]` per projectile.
   Sort by distance to Player, nearest first.
2. **Python side (`obs_utils.py`, `config.py`):** extend observation with 20
   projectile slots x 6 features = 120 new floats. New `obs_dim = 119 + 120 = 239`.
   Use the same normalization conventions as enemies.
3. **Document carefully**: the docstring at the top of `obs_utils.py` must be updated
   to reflect the new layout. This is a hard incompat — BC checkpoints must be retrained.

### Stage C — DeepSets pooling (lightweight entity encoder)

We explicitly **defer** a full Transformer with self-attention per alphastar.md §2.
Instead adopt a DeepSets-style permutation-invariant encoder as a stepping stone:

```
per-entity MLP (6 → 32) → [mean_pool, max_pool] (2 x 32 = 64) → concat with scalar feats
```

- Apply separately to the enemy set and the projectile set (two independent DeepSets).
- Output concatenated with the 19 scalar player features → flat vector fed into the
  existing `[256, 256]` tanh MLP.
- Parameter cost: ~4k params per set, ~8k total. Negligible vs. the existing 70k policy.
- Benefits: permutation invariant, handles variable length gracefully, no "slot 0"
  positional leakage.

This is implemented as a custom `torch.nn.Module` that wraps SB3's `ActorCriticPolicy`
via `features_extractor_class`. No SB3 fork required.

**If and only if** Stage C fails to improve threat prioritization, promote to a single
self-attention layer (`nn.MultiheadAttention`, d_model=64, num_heads=4) in cycle 3.

### Stage D — KL regularization against frozen BC policy

The single most important training-time change. Prevents the exact failure we observed.

1. Keep the BC checkpoint loaded as a **frozen reference policy** `π_SL`.
2. Modify the PPO loss to add `β · KL(π_RL || π_SL)` computed over rollout states.
3. Schedule: start `β = 0.5`, anneal linearly to `β = 0.05` over first 2M steps.
4. This is a ~40-line override of SB3's `PPO._train()` method via a subclass.
5. RAM cost: the frozen BC policy is ~2 MB. Trivial on 128 GB.

This is *not* equivalent to raising `ent_coef`. Entropy pushes toward uniform action
distributions, which would *increase* spam of rare actions. KL against π_SL pushes
toward the *shape* of human behavior, where special ≈ 0.4%.

### Deferred (explicitly out of scope for cycle 2)

These are good ideas from alphastar.md we agreed to defer, with trigger conditions:

| Technique | Defer because | Promote if |
|---|---|---|
| Full Transformer entity encoder | DeepSets cheaper and likely sufficient | DeepSets fails to improve threat prioritization |
| LSTM / GRU core memory | Current obs has reload counter; episodes are short | Agent still fails on multi-room memory tasks |
| Dense → sparse reward decay curriculum | Stage A may fix reward hacking directly | Agent reward-hacks a new exploit post-Stage A |
| Hierarchical RL for mutation selection | Low-frequency decision, small cycle 2 surface | Agent's mutation choices visibly hurt runs |
| Spatial ResNet over tile grid | Walls encoded as 4 raycasts currently | Agent fails to use cover / gets stuck on geometry |
| League Training / constraint specialists | PvE, not adversarial; single-machine | Agent overfits one weapon class |
| Distributed IMPALA + V-Trace | 8 actors is not "distributed" | Scale beyond one machine |
| Full PBT | Overkill for hyperparameter search at this scale | Need automated hyperparam tuning |
| Autoregressive action head | Action coupling not yet shown to be the bottleneck | Move/aim correlation remains broken after B+C+D |
| 150 ms artificial reaction delay | No evidence agent is exploiting superhuman reflexes | Agent clearly out-reflexes humans on hard content |

## Success Metrics for Cycle 2

Before declaring cycle 2 successful, the agent must hit all of the following at the end
of a 10M-step PPO run:

1. **Robot special usage rate < 5%** (cycle 1: 44-53%, human baseline: 0.4%)
2. **Mean portals reached per episode > 0.5** (cycle 1: ~0 — agent doesn't progress)
3. **Mean KL(π_RL || π_SL) stays < 2.0 nats** throughout training (measurable in the loss)
4. **Damage-taken-per-kill ratio improves ≥ 30% vs cycle 1** (proxy for projectile-awareness)
5. **No new reward-hacking exploit** visible in episode replays (manual qualitative check)

If any metric fails, trigger the matching "Promote if" in the deferred table for cycle 3.

## Points of Consensus

- The current obs is wrong, not just suboptimal. Projectiles are mandatory.
- KL regularization against BC is the single highest-leverage training change.
- DeepSets is the right cost/benefit point today; full attention is cycle 3+.
- Fix reward before touching the network — attribution matters.
- PvE changes the calculus: League Training, IMPALA, PBT are not cycle 2 priorities.
- Compute constraints are real and acknowledged. Nothing proposed here needs a GPU.

## Points of Respectful Disagreement (Parked)

- **LSTM/GRU urgency**: alphastar-architect believes temporal memory will be needed for
  weapon cooldowns and multi-room tracking. ppo-pragmatist thinks explicit reload
  features in the scalar obs are sufficient. Resolution: measure after cycle 2; if the
  agent shows reload-timing failures, add a 64-unit GRU.
- **Autoregressive actions**: alphastar-architect wants aim conditioned on movement.
  ppo-pragmatist argues independent sampling has not been shown to be the bottleneck.
  Resolution: add diagnostic logging for `corr(move_dir, aim_dir)` vs. human data in
  cycle 2; act in cycle 3 only if correlation is visibly broken.

## Implementation Order

1. Stage A (GML reward fixes) — 1 day, testable in isolation against a replay
2. Stage B (projectile obs) — 2 days, requires BC retraining
3. Stage C (DeepSets features extractor) — 1 day, pure Python
4. Stage D (KL-regularized PPO subclass) — 1 day, pure Python
5. Retrain BC on projectile-augmented obs — 1 day
6. PPO cycle 2 run with KL reg — 2-3 days wall clock on 8 instances

Total: ~1.5 weeks from start to measurable results.

## Sign-off

Both agents agree this document represents the consensus of the discussion. Neither
side got everything they advocated for, and neither side was asked to surrender core
positions. Disagreements are parked with concrete triggers for re-opening.

— alphastar-architect
— ppo-pragmatist
