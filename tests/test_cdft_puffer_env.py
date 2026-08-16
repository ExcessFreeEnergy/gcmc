"""
Tests for the PufferLib-compatible cDFT fluid manipulation environment.
"""

import time
import numpy as np
import pytest

from gcmc.envs.cdft_puffer import (
    CdftFluidEnv,
    BatchedCdftVecEnv,
    CDFT_OBS_SIZE,
    CDFT_NUM_ACTIONS,
)


def test_cdft_env_single_instance():
    """Verify single instance reset, stepping, shapes, and reward bounds."""
    env = CdftFluidEnv(max_ticks=50, seed=42)
    obs, info = env.reset()

    assert obs.shape == (CDFT_OBS_SIZE,)
    assert np.all(np.isfinite(obs))
    assert "T" in info
    assert "mu" in info
    assert "target_theta" in info

    total_reward = 0.0
    for step in range(50):
        # Apply sample continuous action in [-1, 1]
        action = np.array([0.2, -0.1, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (CDFT_OBS_SIZE,)
        assert np.isfinite(reward)
        total_reward += reward

        if step == 49:
            assert terminated is True
        else:
            assert terminated is False

    assert np.isfinite(total_reward)


def test_cdft_vec_env_batched_throughput():
    """Verify batched vector environment throughput and seamless stepping."""
    num_envs = 256
    vec_env = BatchedCdftVecEnv(num_envs=num_envs, max_ticks=100)
    obs, _ = vec_env.reset()

    assert obs.shape == (num_envs, CDFT_OBS_SIZE)

    # Benchmark 200 batched steps
    steps = 200
    actions = np.zeros((num_envs, CDFT_NUM_ACTIONS), dtype=np.float32)
    actions[:, 0] = 0.1 # phi0 nudge
    actions[:, 1] = -0.05 # mode nudge

    t0 = time.perf_counter()
    for _ in range(steps):
        obs, rewards, terminals, _ = vec_env.step(actions)

    elapsed = time.perf_counter() - t0
    total_env_steps = steps * num_envs
    steps_per_sec = total_env_steps / elapsed

    assert obs.shape == (num_envs, CDFT_OBS_SIZE)
    assert rewards.shape == (num_envs,)
    assert terminals.shape == (num_envs,)
    assert steps_per_sec > 50000.0, f"Vectorized throughput was {steps_per_sec:.0f} steps/s"
    print(f"\nBatched cDFT Environment throughput: {steps_per_sec:,.1f} steps/s across {num_envs} environments")
