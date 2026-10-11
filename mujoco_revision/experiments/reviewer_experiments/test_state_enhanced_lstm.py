"""Information-boundary and initialization tests for State-Enhanced-LSTM."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments" / "h1_deck_balance"
sys.path[:0] = [str(ROOT), str(BASE), str(HERE)]

from train_recurrent_finetune_ppo import load_common  # noqa: E402
from state_enhanced_lstm import (  # noqa: E402
    BASE_OBSERVATION_DIM, ENHANCED_BLOCKS, OBSERVATION_DIM, CausalStateEnhancer,
    StateEnhancedActorCritic, StateEnhancedH12Env, schema_manifest,
)


def test_schema_dimension_is_derived_from_named_blocks():
    expected = BASE_OBSERVATION_DIM + sum(block.dimension for block in ENHANCED_BLOCKS)
    assert OBSERVATION_DIM == expected == schema_manifest()["dimension"]
    assert OBSERVATION_DIM != 62


def test_schema_declares_only_causal_robot_sources():
    schema = schema_manifest()
    assert all(block["causal"] for block in schema["blocks"])
    assert schema["uses_forbidden_actor_sources"] is False
    text = " ".join(block["source"].lower() for block in schema["blocks"])
    assert "deck truth" not in text
    assert "contact truth" not in text
    assert "com truth" not in text


def test_causal_feature_values_and_reset_history():
    enhancer = CausalStateEnhancer(action_scale=0.25)
    first = np.arange(BASE_OBSERVATION_DIM, dtype=np.float32) / 100.0
    reset = enhancer.reset(first)
    assert reset.shape == (OBSERVATION_DIM,)
    np.testing.assert_array_equal(reset[47:53], np.zeros(6, dtype=np.float32))
    np.testing.assert_allclose(reset[53:], first[9:21] - 0.25 * first[33:45])

    second = first.copy()
    second[0:3] += np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    second[3:6] += np.asarray([-0.1, 0.05, 0.2], dtype=np.float32)
    stepped = enhancer.step(second)
    np.testing.assert_allclose(stepped[47:50], second[0:3] - first[0:3])
    np.testing.assert_allclose(stepped[50:53], second[3:6] - first[3:6])


def test_zero_initialized_expansion_is_exactly_base_equivalent():
    torch.manual_seed(20260924)
    base = load_common().eval()
    enhanced = StateEnhancedActorCritic(base).eval()
    obs47 = torch.randn(1, BASE_OBSERVATION_DIM)
    obs65 = torch.cat((obs47, torch.zeros(1, OBSERVATION_DIM - BASE_OBSERVATION_DIM)), dim=1)
    base_state = base.initial_state(1)
    enhanced_state = enhanced.initial_state(1)
    with torch.inference_mode():
        base_action, _, base_value, base_next = base.act_step(obs47, base_state, deterministic=True)
        enhanced_action, _, enhanced_value, enhanced_next = enhanced.act_step(
            obs65, enhanced_state, deterministic=True
        )
    torch.testing.assert_close(enhanced_action, base_action, rtol=0.0, atol=0.0)
    # The wider critic GEMM may accumulate the same zero-padded dot product in
    # a different order; policy actions and recurrent state remain bit exact.
    torch.testing.assert_close(enhanced_value, base_value, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(enhanced_next[0], base_next[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(enhanced_next[1], base_next[1], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(enhanced.actor.memory.weight_ih_l0[:, 47:]) == 0
    assert torch.count_nonzero(enhanced.critic[0].weight[:, 47:]) == 0


def test_wrapped_environment_preserves_action_and_expands_observation():
    env = StateEnhancedH12Env(
        episode_duration=0.04, randomize_phase=False, reward_mode="base",
        deck_roll_amp_deg=1.0, deck_pitch_amp_deg=1.0,
    )
    try:
        obs, _ = env.reset(seed=17)
        assert obs.shape == (OBSERVATION_DIM,)
        assert env.action_space.shape == (12,)
        next_obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        assert next_obs.shape == (OBSERVATION_DIM,)
        assert np.isfinite(next_obs).all() and np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "fall_flag" in info
    finally:
        env.close()
