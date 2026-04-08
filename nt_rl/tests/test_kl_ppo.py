"""Smoke tests for the KL-regularized PPO subclass.

These tests use a tiny dummy Gymnasium env with a MultiDiscrete action space
that mirrors the Nuclear Throne action space. They verify that:

  1. KLRegularizedPPO instantiates with a frozen reference policy.
  2. The reference policy's parameters have ``requires_grad == False``.
  3. ``model.train()`` runs end-to-end after a rollout, producing a finite
     loss and a non-negative ``train/kl_to_reference`` log entry.
  4. The reference policy's weights are unchanged after training (frozen).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch as th

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from nt_rl.kl_ppo import KLRegularizedPPO


# ---------------------------------------------------------------------------
# Dummy MultiDiscrete env (matches NT action shape: [9, 24, 2, 2])
# ---------------------------------------------------------------------------
class _DummyMultiDiscreteEnv(gym.Env):
    """Tiny env with continuous obs and a MultiDiscrete action.

    The reward is constant; the only purpose is to drive PPO's rollout
    collection and training loop.
    """

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 8, episode_len: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.episode_len = episode_len
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9, 24, 2, 2])
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        obs = self.observation_space.sample().astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = self.observation_space.sample().astype(np.float32)
        reward = float(np.random.randn()) * 0.1
        terminated = self._step >= self.episode_len
        return obs, reward, terminated, False, {}


def _make_dummy_env():
    return _DummyMultiDiscreteEnv()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.fixture
def vec_env():
    return DummyVecEnv([_make_dummy_env])


@pytest.fixture
def policy_kwargs():
    return {
        "net_arch": [16, 16],
        "activation_fn": th.nn.Tanh,
    }


@pytest.fixture
def reference_policy(vec_env, policy_kwargs):
    """A frozen ActorCriticPolicy with the same arch as the trainer."""
    return ActorCriticPolicy(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        lr_schedule=lambda _: 0.0,
        **policy_kwargs,
    )


def test_kl_ppo_instantiation(vec_env, policy_kwargs, reference_policy):
    """KLRegularizedPPO should construct without error and freeze the ref."""
    model = KLRegularizedPPO(
        policy=ActorCriticPolicy,
        env=vec_env,
        reference_policy=reference_policy,
        kl_coef_start=0.5,
        kl_coef_end=0.05,
        kl_anneal_steps=1_000,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        learning_rate=3e-4,
        device="cpu",
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    # Reference policy parameters must be frozen
    for p in model.reference_policy.parameters():
        assert p.requires_grad is False

    # Reference policy should be in eval mode
    assert model.reference_policy.training is False


def test_kl_coef_anneal(vec_env, policy_kwargs, reference_policy):
    """KL coefficient should linearly anneal from start to end."""
    model = KLRegularizedPPO(
        policy=ActorCriticPolicy,
        env=vec_env,
        reference_policy=reference_policy,
        kl_coef_start=1.0,
        kl_coef_end=0.0,
        kl_anneal_steps=1000,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        device="cpu",
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    model.num_timesteps = 0
    assert model._current_kl_coef() == pytest.approx(1.0)

    model.num_timesteps = 500
    assert model._current_kl_coef() == pytest.approx(0.5, abs=1e-6)

    model.num_timesteps = 1000
    assert model._current_kl_coef() == pytest.approx(0.0, abs=1e-6)

    model.num_timesteps = 5000
    assert model._current_kl_coef() == pytest.approx(0.0, abs=1e-6)


def test_kl_ppo_one_training_step(vec_env, policy_kwargs, reference_policy):
    """End-to-end: collect a rollout, run train(), verify ref is unchanged."""
    model = KLRegularizedPPO(
        policy=ActorCriticPolicy,
        env=vec_env,
        reference_policy=reference_policy,
        kl_coef_start=0.5,
        kl_coef_end=0.05,
        kl_anneal_steps=1_000,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        learning_rate=3e-4,
        device="cpu",
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    # Snapshot reference policy weights so we can confirm they don't move
    ref_snapshot = {
        name: p.detach().clone()
        for name, p in model.reference_policy.named_parameters()
    }

    # Run a single PPO learn() iteration: one rollout (n_steps) + train()
    model.learn(total_timesteps=16, progress_bar=False)

    # Reference policy parameters must be unchanged after training
    for name, p in model.reference_policy.named_parameters():
        assert th.allclose(p, ref_snapshot[name]), (
            f"Reference policy parameter {name} changed during training"
        )

    # Logger should contain KL-related metrics from the train() override
    name_values = dict(model.logger.name_to_value)
    assert "train/kl_to_reference" in name_values
    assert "train/kl_coef" in name_values
    assert np.isfinite(name_values["train/kl_to_reference"])
    assert name_values["train/kl_to_reference"] >= 0.0
    # KL coef should be within [end, start]
    assert 0.05 - 1e-6 <= name_values["train/kl_coef"] <= 0.5 + 1e-6


def test_kl_to_reference_zero_when_policies_identical(
    vec_env, policy_kwargs, reference_policy
):
    """If the trainer policy is initialised to match the reference, KL ~ 0."""
    model = KLRegularizedPPO(
        policy=ActorCriticPolicy,
        env=vec_env,
        reference_policy=reference_policy,
        kl_coef_start=0.5,
        kl_coef_end=0.05,
        kl_anneal_steps=1_000,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        device="cpu",
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    # Copy reference weights into the trainer policy
    model.policy.load_state_dict(reference_policy.state_dict(), strict=False)

    obs = th.as_tensor(
        vec_env.observation_space.sample()[None, :], dtype=th.float32
    )
    kl = model._compute_kl_to_reference(obs)
    assert float(kl.detach().cpu().item()) == pytest.approx(0.0, abs=1e-5)
