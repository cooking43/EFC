"""Tests for information boundaries and fair action paths in C2--C5."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fair_controller_ablation import (  # noqa: E402
    KNEE_DIRECTION_SIGN, METHODS, OBSERVATION_DIM, RESIDUAL_LIMIT_RAD, FairAblationEnv,
    PlatformRollTaylorPredictor, knee_residual_target,
    load_predictor_augmented_common, observation_schema,
)
from train_recurrent_finetune_ppo import load_common  # noqa: E402


def test_actor_schema_is_causal_and_shared():
    schema = observation_schema()
    assert schema["dimension"] == OBSERVATION_DIM == 48
    assert schema["actor_deck_truth"] is False
    assert all(block["causal"] for block in schema["blocks"])


def test_zero_initialized_prediction_inputs_preserve_base_policy():
    torch.manual_seed(20260926)
    base = load_common().eval()
    expanded = load_predictor_augmented_common().eval()
    obs47 = torch.randn(1, 47)
    obs48 = torch.cat((obs47, torch.randn(1, 1)), dim=1)
    with torch.inference_mode():
        base_action, _, _, base_state = base.act_step(obs47, base.initial_state(1), deterministic=True)
        new_action, _, _, new_state = expanded.act_step(obs48, expanded.initial_state(1), deterministic=True)
    torch.testing.assert_close(new_action, base_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(new_state[0], base_state[0], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(expanded.actor.memory.weight_ih_l0[:, 47:]) == 0


def test_knee_mapping_is_bilateral_bounded_and_uses_no_other_joint():
    assert KNEE_DIRECTION_SIGN == 1.0
    residual = knee_residual_target(compensation_rad=1.0)
    assert np.max(np.abs(residual)) <= RESIDUAL_LIMIT_RAD
    assert np.count_nonzero(residual) == 2
    np.testing.assert_array_equal(np.nonzero(residual)[0], np.asarray([3, 9]))
    assert residual[3] == -residual[9]


def test_taylor_predictor_order_and_horizon_change_actual_output():
    predictor = PlatformRollTaylorPredictor(horizon_s=0.1, control_dt_s=0.02,
                                            filter_new_weight=1.0)
    predictor.reset(np.deg2rad(0.0))
    predictor.step(np.deg2rad(1.0))
    state = predictor.step(np.deg2rad(3.0))
    assert np.isclose(state["persistence_roll_deg"], 3.0)
    assert state["first_order_roll_deg"] != state["persistence_roll_deg"]
    assert state["second_order_roll_deg"] != state["first_order_roll_deg"]


def test_all_variants_share_prediction_and_residual_limit_paths():
    envs = [FairAblationEnv(method=method, episode_duration=0.04, randomize_phase=False,
                            reward_mode="base", deck_roll_amp_deg=3.0,
                            deck_pitch_amp_deg=2.0) for method in METHODS]
    try:
        observations = []
        for env in envs:
            obs, _ = env.reset(seed=17)
            observations.append(obs)
        for obs in observations[1:]:
            np.testing.assert_allclose(obs, observations[0])
        policy_action = np.linspace(-0.25, 0.25, 12, dtype=np.float32)
        for env in envs:
            _, _, _, _, info = env.step(policy_action)
            assert env.last_applied_action.shape == (12,)
            assert info["ablation"]["residual_max_abs_rad"] <= RESIDUAL_LIMIT_RAD
        assert np.count_nonzero(envs[0].last_residual_rad) == 0
        np.testing.assert_array_equal(envs[0].last_applied_action, policy_action.astype(np.float64))
        assert envs[-1].method == "Oracle-Predictive-Residual"
        assert envs[-1].last_residual_rad.shape == envs[1].last_residual_rad.shape
    finally:
        for env in envs:
            env.close()


def test_roll_sensor_stress_is_causal_seeded_and_zero_setting_is_exact():
    clean_a = FairAblationEnv(method="Predictive-Residual", episode_duration=0.08,
                              randomize_phase=False, reward_mode="base",
                              deck_roll_amp_deg=3.0, deck_pitch_amp_deg=3.0)
    clean_b = FairAblationEnv(method="Predictive-Residual", episode_duration=0.08,
                              randomize_phase=False, reward_mode="base",
                              deck_roll_amp_deg=3.0, deck_pitch_amp_deg=3.0,
                              roll_sensor_noise_std_deg=0.0,
                              roll_sensor_bias_deg=0.0,
                              roll_sensor_latency_ms=0)
    stressed_a = FairAblationEnv(method="Predictive-Residual", episode_duration=0.08,
                                 randomize_phase=False, reward_mode="base",
                                 deck_roll_amp_deg=3.0, deck_pitch_amp_deg=3.0,
                                 roll_sensor_noise_std_deg=0.2,
                                 roll_sensor_bias_deg=0.5,
                                 roll_sensor_latency_ms=40)
    stressed_b = FairAblationEnv(method="Predictive-Residual", episode_duration=0.08,
                                 randomize_phase=False, reward_mode="base",
                                 deck_roll_amp_deg=3.0, deck_pitch_amp_deg=3.0,
                                 roll_sensor_noise_std_deg=0.2,
                                 roll_sensor_bias_deg=0.5,
                                 roll_sensor_latency_ms=40)
    envs = (clean_a, clean_b, stressed_a, stressed_b)
    try:
        observations = [env.reset(seed=123)[0] for env in envs]
        np.testing.assert_array_equal(observations[0], observations[1])
        np.testing.assert_array_equal(observations[2], observations[3])
        action = np.zeros(12, dtype=np.float32)
        outputs = [env.step(action) for env in envs]
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[2][0], outputs[3][0])
        assert not np.isclose(stressed_a.last_true_roll_rad,
                              stressed_a.last_measured_roll_rad)
        assert stressed_a.roll_sensor_latency_samples == 2
    finally:
        for env in envs:
            env.close()


def test_roll_sensor_latency_must_align_with_control_rate():
    with np.testing.assert_raises(ValueError):
        FairAblationEnv(method="Predictive-Residual", episode_duration=0.04,
                        randomize_phase=False, reward_mode="base",
                        deck_roll_amp_deg=3.0, deck_pitch_amp_deg=3.0,
                        roll_sensor_latency_ms=30)
