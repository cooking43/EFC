"""Contract tests for the corrected roll/pitch-only H1-2 deck."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments" / "h1_deck_balance"
sys.path[:0] = [str(BASE), str(HERE)]

from deck_motion import get_deck_pose  # noqa: E402
from h1_2_official_deck_env import H12OfficialDeckEnv, SCENE  # noqa: E402
from fair_controller_ablation import LEFT_KNEE_NAME, RIGHT_KNEE_NAME  # noqa: E402


def test_deck_geometry_is_50_by_50_by_02_and_has_no_other_collision_plane():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    deck = model.geom("oscillating_deck_geom").id
    np.testing.assert_allclose(model.geom_size[deck], np.asarray([25.0, 25.0, 0.1]))
    collidable_planes = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0:
            collidable_planes.append(geom_id)
    assert collidable_planes == []


def test_motion_is_roll_pitch_only_and_startup_ramp_is_single_clock_based():
    cfg = {"deck": {"roll_amp_deg": 5.0, "pitch_amp_deg": 3.0,
                    "roll_freq_hz": 0.2, "pitch_freq_hz": 0.15,
                    "roll_phase_rad": math.pi / 2.0,
                    "pitch_phase_rad": math.pi / 2.0,
                    "surface_height_m": 0.0, "thickness": 0.2,
                    "startup_ramp_s": 1.0}}
    start = get_deck_pose(0.0, cfg)
    middle = get_deck_pose(0.5, cfg)
    settled = get_deck_pose(1.5, cfg)
    assert start["startup_ramp"] == 0.0
    assert 0.0 < middle["startup_ramp"] < 1.0
    assert settled["startup_ramp"] == 1.0
    for pose in (start, middle, settled):
        np.testing.assert_allclose(pose["position"][:2], 0.0)
        assert pose["position"][2] == -0.1
        assert pose["heave"] == 0.0 and pose["heave_rate"] == 0.0


def test_robot_reset_does_not_reset_deck_time_or_resample_phase():
    env = H12OfficialDeckEnv(episode_duration=0.2, randomize_phase=True,
                            reward_mode="base", deck_roll_amp_deg=5.0,
                            deck_pitch_amp_deg=3.0)
    try:
        env.reset(seed=101)
        phase = (env.deck_cfg["deck"]["roll_phase_rad"],
                 env.deck_cfg["deck"]["pitch_phase_rad"])
        for _ in range(5):
            env.step(np.zeros(12, dtype=np.float32))
        time_before = env.deck_time
        env.reset(seed=202)
        assert env.deck_time == time_before
        assert phase == (env.deck_cfg["deck"]["roll_phase_rad"],
                         env.deck_cfg["deck"]["pitch_phase_rad"])
    finally:
        env.close()


def test_reward_reports_support_relative_height():
    env = H12OfficialDeckEnv(episode_duration=0.04, randomize_phase=False,
                            reward_mode="base", deck_roll_amp_deg=2.0,
                            deck_pitch_amp_deg=2.0)
    try:
        env.reset(seed=11)
        _, reward, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        value = info["reward_components"]["support_relative_base_height_m"]
        assert np.isfinite(reward) and np.isfinite(value)
        assert 0.5 < value < 1.5
    finally:
        env.close()


def test_corrected_reward_is_reconstructable_and_not_nominally_saturated():
    env = H12OfficialDeckEnv(episode_duration=0.04, randomize_phase=False,
                            reward_mode="corrected", deck_roll_amp_deg=3.0,
                            deck_pitch_amp_deg=3.0)
    try:
        env.reset(seed=19)
        _, reward, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        component = info["reward_components"]
        keys = ("alive_reward", "world_upright_reward",
                "support_relative_height_reward", "joint_deviation_penalty",
                "action_delta_penalty", "angular_velocity_penalty",
                "slip_penalty", "fall_penalty")
        assert math.isclose(reward, sum(float(component[key]) for key in keys), abs_tol=1e-12)
        assert reward < 0.98
        assert -0.4 <= float(component["slip_penalty"]) <= 0.0
    finally:
        env.close()


def test_named_knee_sign_matches_positive_roll_geometry_at_standing_pose():
    env = H12OfficialDeckEnv(episode_duration=0.04, randomize_phase=False,
                            reward_mode="corrected", deck_roll_amp_deg=0.0,
                            deck_pitch_amp_deg=0.0)
    try:
        env.reset(seed=23)
        env.data.qpos[7:] = env.default
        left = int(env.model.joint(LEFT_KNEE_NAME).qposadr[0])
        right = int(env.model.joint(RIGHT_KNEE_NAME).qposadr[0])
        env.data.qpos[left] += math.radians(1.0)
        env.data.qpos[right] -= math.radians(1.0)
        mujoco.mj_forward(env.model, env.data)
        # Positive deck roll raises its +y side; H1-2's named left leg is +y.
        # Left-knee flexion must shorten that leg, seen here as the left ankle
        # moving upward relative to a fixed pelvis.
        assert env.data.xpos[env.left_body, 2] > env.data.xpos[env.right_body, 2]
    finally:
        env.close()
