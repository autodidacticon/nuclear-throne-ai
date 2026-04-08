"""DeepSets-style entity features extractor for Nuclear Throne RL.

The flat 239-dim observation contains three semantically distinct groups:

  1. Player scalars (19 features) — global state, fed through unchanged.
  2. Enemy set (20 enemies x 5 features) — variable-cardinality, padded with
     zeros. Order is "nearest first" but we still want a permutation-invariant
     pooled representation so the policy generalizes across enemy populations.
  3. Projectile set (20 projectiles x 6 features) — same idea as enemies.

The DeepSets construction (Zaheer et al. 2017) applies a per-entity MLP and
then pools across entities. We use mean+max pooling so the policy can attend
both to "average threat" and "worst-case" entities. Padding slots (all-zero
features) are masked out so they don't dilute the pooled representation.

Output dimensions:
    player(19) + enemy_mean(32) + enemy_max(32) + proj_mean(32) + proj_max(32)
    = 147 features
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Layout constants — must mirror nt_rl.config.EnvConfig defaults.
PLAYER_DIM = 19
N_ENEMIES = 20
ENEMY_DIM = 5
N_PROJECTILES = 20
PROJECTILE_DIM = 6

ENEMY_OFFSET = PLAYER_DIM
PROJECTILE_OFFSET = PLAYER_DIM + N_ENEMIES * ENEMY_DIM
TOTAL_OBS_DIM = PROJECTILE_OFFSET + N_PROJECTILES * PROJECTILE_DIM  # 239

# Per-entity hidden dim. Mean + max pool doubles this for the output.
ENTITY_HIDDEN = 32
DEEPSETS_FEATURES_DIM = (
    PLAYER_DIM
    + 2 * ENTITY_HIDDEN  # enemy mean + max
    + 2 * ENTITY_HIDDEN  # projectile mean + max
)  # 19 + 64 + 64 = 147


class DeepSetsExtractor(BaseFeaturesExtractor):
    """Permutation-invariant features extractor for player + enemies + projectiles.

    Splits the flat observation, encodes each entity through a small MLP, then
    mean+max pools over the entity dimension with proper masking for padding.
    """

    def __init__(self, observation_space, features_dim: int = DEEPSETS_FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)

        if features_dim != DEEPSETS_FEATURES_DIM:
            raise ValueError(
                f"DeepSetsExtractor requires features_dim={DEEPSETS_FEATURES_DIM}, "
                f"got {features_dim}"
            )

        # Per-entity encoders (shared weights across slots within a set)
        self.enemy_mlp = nn.Sequential(
            nn.Linear(ENEMY_DIM, ENTITY_HIDDEN),
            nn.Tanh(),
            nn.Linear(ENTITY_HIDDEN, ENTITY_HIDDEN),
            nn.Tanh(),
        )
        self.projectile_mlp = nn.Sequential(
            nn.Linear(PROJECTILE_DIM, ENTITY_HIDDEN),
            nn.Tanh(),
            nn.Linear(ENTITY_HIDDEN, ENTITY_HIDDEN),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: [batch, 239]
        player = obs[:, :ENEMY_OFFSET]
        enemies = obs[:, ENEMY_OFFSET:PROJECTILE_OFFSET].view(
            -1, N_ENEMIES, ENEMY_DIM
        )
        projectiles = obs[:, PROJECTILE_OFFSET:TOTAL_OBS_DIM].view(
            -1, N_PROJECTILES, PROJECTILE_DIM
        )

        # Padding masks: an entity slot is "real" if any feature is non-zero.
        # Shape: [batch, N, 1] for broadcasting against per-entity features.
        enemy_mask = (enemies.abs().sum(-1, keepdim=True) > 0).float()
        proj_mask = (projectiles.abs().sum(-1, keepdim=True) > 0).float()

        # Per-entity encoding, zeroed for padding slots.
        enemy_feats = self.enemy_mlp(enemies) * enemy_mask  # [B, N, H]
        proj_feats = self.projectile_mlp(projectiles) * proj_mask  # [B, N, H]

        # Mean pool with masked-count denominator (avoids dividing by 20 when
        # only a few entities exist).
        enemy_count = enemy_mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        proj_count = proj_mask.sum(dim=1).clamp(min=1.0)
        enemy_mean = enemy_feats.sum(dim=1) / enemy_count  # [B, H]
        proj_mean = proj_feats.sum(dim=1) / proj_count

        # Max pool with masked padding pushed to -inf so it never wins.
        enemy_max = (enemy_feats - 1e9 * (1.0 - enemy_mask)).max(dim=1).values
        proj_max = (proj_feats - 1e9 * (1.0 - proj_mask)).max(dim=1).values

        # If a set is entirely empty the max pool would be -1e9 — replace with 0
        # so the trunk MLP isn't fed wild values.
        empty_enemies = (enemy_mask.sum(dim=1) == 0)  # [B, 1]
        empty_proj = (proj_mask.sum(dim=1) == 0)
        enemy_max = torch.where(empty_enemies, torch.zeros_like(enemy_max), enemy_max)
        proj_max = torch.where(empty_proj, torch.zeros_like(proj_max), proj_max)

        return torch.cat(
            [player, enemy_mean, enemy_max, proj_mean, proj_max],
            dim=-1,
        )
