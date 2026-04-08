"""KL-regularized PPO for Nuclear Throne RL.

Subclasses Stable Baselines3's PPO to add a KL divergence penalty against a
frozen reference policy (typically the BC baseline). This prevents the trained
policy from drifting arbitrarily far from the BC behaviour, which previously
produced degenerate strategies (e.g. special-ability spam).

This is the AlphaStar-style approach: penalize KL(current || reference)
during PPO updates, with a coefficient that linearly anneals from
``kl_coef_start`` down to ``kl_coef_end`` over ``kl_anneal_steps`` env steps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.distributions import (
    MultiCategoricalDistribution,
    kl_divergence,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance


class KLRegularizedPPO(PPO):
    """PPO with KL divergence regularization against a frozen reference policy.

    Adds ``kl_coef * KL(current_policy || reference_policy)`` to the PPO loss.
    The coefficient is linearly annealed from ``kl_coef_start`` at step 0 to
    ``kl_coef_end`` at ``kl_anneal_steps`` (and then held constant).

    Parameters mirror :class:`stable_baselines3.PPO` plus:

    :param reference_policy: A frozen ``ActorCriticPolicy`` to regularize against.
        If ``None``, behaves identically to vanilla PPO.
    :param kl_coef_start: Initial KL coefficient (default 0.5).
    :param kl_coef_end: Final KL coefficient after annealing (default 0.05).
    :param kl_anneal_steps: Number of environment steps over which to anneal
        the KL coefficient (default 2,000,000).
    """

    def __init__(
        self,
        *args: Any,
        reference_policy: ActorCriticPolicy | None = None,
        kl_coef_start: float = 0.5,
        kl_coef_end: float = 0.05,
        kl_anneal_steps: int = 2_000_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.reference_policy = reference_policy
        if self.reference_policy is not None:
            # Freeze every parameter of the reference policy
            for p in self.reference_policy.parameters():
                p.requires_grad = False
            self.reference_policy.eval()
            # Move to the same device as the trainer policy
            self.reference_policy.to(self.device)

        self.kl_coef_start = float(kl_coef_start)
        self.kl_coef_end = float(kl_coef_end)
        self.kl_anneal_steps = int(kl_anneal_steps)

    # ------------------------------------------------------------------
    # Coefficient schedule
    # ------------------------------------------------------------------
    def _current_kl_coef(self) -> float:
        """Linearly anneal the KL coefficient from start to end."""
        if self.kl_anneal_steps <= 0:
            return self.kl_coef_end
        if self.num_timesteps >= self.kl_anneal_steps:
            return self.kl_coef_end
        progress = float(self.num_timesteps) / float(self.kl_anneal_steps)
        return self.kl_coef_start + progress * (self.kl_coef_end - self.kl_coef_start)

    # ------------------------------------------------------------------
    # KL against the reference policy
    # ------------------------------------------------------------------
    def _compute_kl_to_reference(self, obs: th.Tensor) -> th.Tensor:
        """Compute the mean ``KL(current_policy || reference_policy)``.

        Returns a 0-D tensor on the trainer device. If no reference policy is
        attached, returns 0.0.
        """
        if self.reference_policy is None:
            return th.zeros((), device=self.device)

        with th.no_grad():
            ref_dist = self.reference_policy.get_distribution(obs)

        current_dist = self.policy.get_distribution(obs)

        # SB3's `kl_divergence` handles MultiCategoricalDistribution properly
        # (sums over action dimensions for us). Wrap in a try/except so we can
        # gracefully fall back if the distribution combination is not
        # supported by SB3 / torch.
        try:
            kl = kl_divergence(current_dist, ref_dist)
        except (NotImplementedError, AssertionError):
            # Manual fallback for MultiCategoricalDistribution
            if isinstance(current_dist, MultiCategoricalDistribution) and isinstance(
                ref_dist, MultiCategoricalDistribution
            ):
                kl_terms = [
                    th.distributions.kl.kl_divergence(curr, ref)
                    for curr, ref in zip(
                        current_dist.distribution, ref_dist.distribution
                    )
                ]
                kl = th.stack(kl_terms, dim=1).sum(dim=1)
            else:
                # Last-ditch fallback: zero penalty (still differentiable wrt
                # current policy via the kl tensor below)
                kl = th.zeros(
                    (obs.shape[0],), device=self.device, dtype=th.float32
                )

        return kl.mean()

    # ------------------------------------------------------------------
    # Training loop (mirrors SB3 PPO.train(), with the KL term added)
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Update policy using the rollout buffer, with KL regularization."""
        # Switch to train mode (affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        kl_coef = self._current_kl_coef()

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        kl_to_ref_losses: list[float] = []
        clip_fractions: list[float] = []

        continue_training = True
        loss = th.zeros((), device=self.device)

        for epoch in range(self.n_epochs):
            approx_kl_divs: list[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # PPO clipped surrogate loss
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # KL to frozen reference policy
                kl_to_ref = self._compute_kl_to_reference(rollout_data.observations)
                kl_to_ref_losses.append(float(kl_to_ref.detach().cpu().item()))

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + kl_coef * kl_to_ref
                )

                # Approximate reverse KL for early stopping (vs. old policy)
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: "
                            f"{approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Logs
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record(
            "train/policy_gradient_loss", float(np.mean(pg_losses))
        )
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        self.logger.record("train/loss", float(loss.item()))
        self.logger.record("train/explained_variance", float(explained_var))
        self.logger.record(
            "train/kl_to_reference",
            float(np.mean(kl_to_ref_losses)) if kl_to_ref_losses else 0.0,
        )
        self.logger.record("train/kl_coef", float(kl_coef))
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", float(th.exp(self.policy.log_std).mean().item())
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", float(clip_range))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", float(clip_range_vf))
