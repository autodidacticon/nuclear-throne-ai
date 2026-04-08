"""Reward configuration for PPO training.

This file is the primary iteration target across reward-shaping cycles.
Increment the version tag each cycle.

NOTE: The GML bridge (scr_agent_compute_reward) computes reward server-side
and sends it in state["reward"]. These weights serve as the Python-side
reference. If reward shaping is done purely in GML, this config is
documentation only and GML is the ground truth. If Python-side reward
overrides are enabled in the environment, these weights are authoritative.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    # Version tag — increment each cycle
    version: str = "v1.0"

    # Combat
    reward_kill: float = 5.0
    reward_damage_dealt: float = 0.0    # Optional — enable if kill reward insufficient
    reward_damage_taken: float = -1.0
    reward_death: float = -15.0

    # Progression
    reward_level_complete: float = 10.0
    reward_boss_kill: float = 25.0

    # Resource management
    reward_health_pickup_low_hp: float = 2.0    # Only when hp < 50% of max
    reward_health_pickup_full_hp: float = -0.5  # Penalise wasteful pickup
    reward_ammo_pickup: float = 0.2
    reward_weapon_pickup: float = 0.5

    # Survival
    reward_survival_per_step: float = 0.01

    # Anti-reward-hacking guards
    reward_idle_penalty_threshold: int = 120    # Steps without kill or movement
    reward_idle_penalty: float = -0.1           # Applied per step when idle too long
