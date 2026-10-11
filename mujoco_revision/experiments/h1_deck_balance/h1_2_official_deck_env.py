"""Unified 12-DoF H1-2 standing environment based on Unitree's interfaces."""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import yaml

from deck_motion import get_deck_pose
from validate_unitree_pretrained_h1_2 import gravity_orientation, quat_roll_pitch

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "external" / "unitree_rl_gym"
DEPLOY_CONFIG = UPSTREAM / "deploy" / "deploy_mujoco" / "configs" / "h1_2.yaml"
LEGACY_SCENE = UPSTREAM / "resources" / "robots" / "h1_2" / "deck_scene.xml"
# Default visualization now uses the paper-style scene requested for the
# experiment.  This file is dynamics-equivalent to LEGACY_SCENE.
SCENE = UPSTREAM / "resources" / "robots" / "h1_2" / "deck_scene_paper.xml"
DECK_CONFIG = HERE / "configs" / "deck_stand.yaml"


class H12OfficialDeckEnv(gym.Env):
    """Static-command balance task with no gait phase or pretrained reflex."""

    metadata = {"render_modes": []}

    def __init__(self, episode_duration: float = 10.0, randomize_phase: bool = True,
                 reward_mode: str = "base", deck_roll_amp_deg: float = 1.0,
                 deck_pitch_amp_deg: float = 1.0, scene_path: str | Path | None = None):
        super().__init__()
        deploy = yaml.safe_load(DEPLOY_CONFIG.read_text(encoding="utf-8"))
        self.deck_cfg = yaml.safe_load(DECK_CONFIG.read_text(encoding="utf-8"))
        self.deck_cfg["deck"].update(roll_amp_deg=float(deck_roll_amp_deg),
                                     pitch_amp_deg=float(deck_pitch_amp_deg), heave_amp_m=0.0)
        # A caller may select a visualization-only scene.  Training and all
        # existing evaluations retain the original scene by default.
        self.scene_path = Path(scene_path) if scene_path is not None else SCENE
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(deploy["simulation_dt"])
        self.data = mujoco.MjData(self.model)
        self.decimation = int(deploy["control_decimation"])
        self.max_steps = int(round(episode_duration / (self.model.opt.timestep * self.decimation)))
        self.kp = np.asarray(deploy["kps"], dtype=np.float64)
        self.kd = np.asarray(deploy["kds"], dtype=np.float64)
        self.default = np.asarray(deploy["default_angles"], dtype=np.float64)
        self.action_scale = float(deploy["action_scale"])
        self.randomize_phase = bool(randomize_phase)
        if reward_mode not in {"base", "corrected", "base_plus_com",
                               "base_plus_multi_stability"}:
            raise ValueError(f"unsupported reward_mode: {reward_mode}")
        self.reward_mode = reward_mode
        self.previous_action = np.zeros(12, dtype=np.float64)
        self.steps = 0
        self.deck_time = 0.0
        self._deck_phase_initialized = False
        deck_body = self.model.body("oscillating_deck").id
        self.deck_body = deck_body
        self.deck_mocap = int(self.model.body_mocapid[deck_body])
        self.deck_geom = self.model.geom("oscillating_deck_geom").id
        self.pelvis_body = self.model.body("pelvis").id
        self.left_body = self.model.body("left_ankle_roll_link").id
        self.right_body = self.model.body("right_ankle_roll_link").id
        # Unitree's deploy_mujoco.py applies the network output directly and
        # does not clip it to [-1, 1].  PPO actors may still impose their own
        # bounded distribution, but the plant interface must not alter actions.
        self.action_space = spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
        # Match Unitree's 47-D deployment interface exactly: omega(3),
        # gravity(3), zero velocity command(3), q(12), dq(12), previous
        # action(12), gait clock(2).  Plain PPO does not observe deck truth.
        self.observation_space = spaces.Box(-np.inf, np.inf, (47,), dtype=np.float32)

    def _set_deck(self) -> dict:
        pose = get_deck_pose(self.deck_time, self.deck_cfg)
        self.data.mocap_pos[self.deck_mocap] = pose["position"]
        self.data.mocap_quat[self.deck_mocap] = pose["quaternion"]
        return pose

    def _observation(self, pose: dict) -> np.ndarray:
        phase = (float(self.data.time) % 0.8) / 0.8
        return np.concatenate((
            self.data.qvel[3:6] * 0.25,
            gravity_orientation(self.data.qpos[3:7]),
            np.zeros(3, dtype=np.float64),
            self.data.qpos[7:] - self.default,
            self.data.qvel[6:] * 0.05,
            self.previous_action,
            np.asarray([math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)]),
        )).astype(np.float32)

    def _contacts_and_slip(self) -> tuple[bool, bool, float]:
        left = right = False
        slip_squared: list[float] = []
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if self.deck_geom not in (int(contact.geom1), int(contact.geom2)):
                continue
            other = int(contact.geom2 if int(contact.geom1) == self.deck_geom else contact.geom1)
            body = int(self.model.geom_bodyid[other])
            if body not in (self.left_body, self.right_body):
                continue
            left |= body == self.left_body; right |= body == self.right_body
            point = np.asarray(contact.pos)
            foot_velocity = (self.data.cvel[body, 3:6]
                             + np.cross(self.data.cvel[body, :3], point - self.data.xpos[body]))
            deck_velocity = (self.data.cvel[self.deck_body, 3:6]
                             + np.cross(self.data.cvel[self.deck_body, :3], point - self.data.xpos[self.deck_body]))
            relative = foot_velocity - deck_velocity
            normal = np.asarray(contact.frame[:3])
            tangent = relative - np.dot(relative, normal) * normal
            slip_squared.append(float(np.dot(tangent, tangent)))
        return left, right, float(np.mean(slip_squared)) if slip_squared else 0.0

    def _com_support_error(self) -> np.ndarray:
        """CoM error from the feet midpoint, expressed in deck-local x/y."""
        support_center = 0.5 * (self.data.xpos[self.left_body] + self.data.xpos[self.right_body])
        deck_rotation = self.data.xmat[self.deck_body].reshape(3, 3)
        return (deck_rotation.T @ (self.data.subtree_com[self.pelvis_body] - support_center))[:2]

    def _constraint_axis_margins(self) -> tuple[float, float]:
        """Return fore-aft and lateral signed CoM support margins."""
        deck_pos = self.data.xpos[self.deck_body]
        deck_rot = self.data.xmat[self.deck_body].reshape(3, 3)
        left, right, _ = self._contacts_and_slip()
        active_bodies = [body for active, body in ((left, self.left_body), (right, self.right_body)) if active]
        if not active_bodies:
            active_bodies = [self.left_body, self.right_body]
        foot_local = np.asarray([deck_rot.T @ (self.data.xpos[body] - deck_pos) for body in active_bodies])
        support_min = np.asarray([np.mean(foot_local[:, 0]) - 0.12, np.min(foot_local[:, 1]) - 0.06])
        support_max = np.asarray([np.mean(foot_local[:, 0]) + 0.12, np.max(foot_local[:, 1]) + 0.06])
        com_local = deck_rot.T @ (self.data.subtree_com[self.pelvis_body] - deck_pos)
        margins = np.minimum(com_local[:2] - support_min, support_max - com_local[:2])
        return float(margins[0]), float(margins[1])

    def constraint_costs(self) -> dict[str, float]:
        """Original costs retained exactly for all existing checkpoints."""
        x_margin, y_margin = self._constraint_axis_margins()
        com_cost = max(0.0, 0.03 - min(x_margin, y_margin))

        pose = get_deck_pose(self.deck_time, self.deck_cfg)
        deck_pos = self.data.xpos[self.deck_body]
        deck_rot = self.data.xmat[self.deck_body].reshape(3, 3)
        left, right, _ = self._contacts_and_slip()
        active_bodies = [body for active, body in ((left, self.left_body), (right, self.right_body)) if active]
        if not active_bodies:
            active_bodies = [self.left_body, self.right_body]
        deck_omega = np.asarray([pose["roll_rate"] * math.cos(pose["pitch"]),
                                 pose["pitch_rate"],
                                 -pose["roll_rate"] * math.sin(pose["pitch"])])
        deck_origin_velocity = np.asarray([0.0, 0.0, pose["heave_rate"]])
        speeds = []
        for body in active_bodies:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                     body, velocity, 0)
            point = self.data.xpos[body]
            deck_surface_velocity = deck_origin_velocity + np.cross(deck_omega, point - pose["position"])
            relative_local = deck_rot.T @ (velocity[3:6] - deck_surface_velocity)
            speeds.append(float(np.linalg.norm(relative_local[:2])))
        slip_cost = max(0.0, max(speeds) - 0.05)

        pelvis_up = self.data.xmat[self.pelvis_body].reshape(3, 3)[:, 2]
        relative_tilt = math.acos(float(np.clip(np.dot(pelvis_up, deck_rot[:, 2]), -1.0, 1.0)))
        torso_cost = max(0.0, relative_tilt - math.radians(15.0))
        return {"com": com_cost, "slip": slip_cost, "torso": torso_cost}

    def axis_separated_constraint_costs(self) -> dict[str, float]:
        """Experimental axis split; slip and torso remain bit-identical."""
        x_margin, y_margin = self._constraint_axis_margins()
        original = self.constraint_costs()
        return {"com_x": max(0.0, 0.03 - x_margin),
                "com_y": max(0.0, 0.03 - y_margin),
                "slip": original["slip"], "torso": original["torso"]}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.randomize_phase and not self._deck_phase_initialized:
            for key in ("roll_phase_rad", "pitch_phase_rad"):
                self.deck_cfg["deck"][key] = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            self._deck_phase_initialized = True
        self.previous_action[:] = 0.0
        self.steps = 0
        pose = self._set_deck()
        mujoco.mj_forward(self.model, self.data)
        return self._observation(pose), self._info(False)

    def _info(self, fall: bool) -> dict:
        roll, pitch = quat_roll_pitch(self.data.qpos[3:7])
        left, right, slip = self._contacts_and_slip()
        pose = get_deck_pose(self.deck_time, self.deck_cfg)
        pelvis_up = self.data.xmat[self.pelvis_body].reshape(3, 3)[:, 2]
        deck_up = self.data.xmat[self.deck_body].reshape(3, 3)[:, 2]
        relative_tilt = math.acos(float(np.clip(np.dot(pelvis_up, deck_up), -1.0, 1.0)))
        return {
            "time": self.deck_time, "episode_time": float(self.data.time), "fall_flag": fall,
            "base_roll": roll, "base_pitch": pitch,
            "world_tilt_rad": float(math.hypot(roll, pitch)),
            "deck_relative_tilt_rad": relative_tilt,
            "deck_roll_rad": float(pose["roll"]),
            "deck_pitch_rad": float(pose["pitch"]),
            "deck_roll_rate_rad_s": float(pose["roll_rate"]),
            "deck_pitch_rate_rad_s": float(pose["pitch_rate"]),
            "base_height": float(self.data.qpos[2]),
            "left_foot_contact": left, "right_foot_contact": right,
            "foot_slip_squared_speed": slip,
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        target = self.default + self.action_scale * action
        for _ in range(self.decimation):
            self._set_deck()
            self.data.ctrl[:] = ((target - self.data.qpos[7:]) * self.kp
                                 - self.data.qvel[6:] * self.kd)
            mujoco.mj_step(self.model, self.data)
            self.deck_time += float(self.model.opt.timestep)
        self.steps += 1
        pose = get_deck_pose(self.deck_time, self.deck_cfg)
        roll, pitch = quat_roll_pitch(self.data.qpos[3:7])
        height = float(self.data.qpos[2])
        deck_rotation = self.data.xmat[self.deck_body].reshape(3, 3)
        relative_height = float((deck_rotation.T @
                                 (self.data.xpos[self.pelvis_body] - self.data.xpos[self.deck_body]))[2]
                                - 0.5 * float(self.deck_cfg["deck"]["thickness"]))
        fall = bool(height < 0.6 or abs(roll) > math.pi / 4 or abs(pitch) > math.pi / 4
                    or not np.isfinite(self.data.qpos).all())
        left, right, slip = self._contacts_and_slip()
        joint_error = self.data.qpos[7:] - self.default
        action_delta = action - self.previous_action
        # Plain PPO task reward: no CoM, prediction, multiplier, or progressive
        # constraint term.  Keep each signed contribution explicit so that the
        # reward range and any saturation can be reconstructed from event-level
        # logs rather than inferred from the total alone.
        fall_penalty = -10.0 * float(fall)
        if self.reward_mode == "corrected":
            # Non-saturating, dimensionless standing objective.  Physical
            # errors are normalized by fixed, manifest-visible scales so the
            # common initialization retains a measurable learning signal.
            alive_reward = 1.0
            world_upright_reward = -(
                (roll / math.radians(5.0)) ** 2
                + (pitch / math.radians(5.0)) ** 2
            )
            support_relative_height_reward = (
                -0.25 * ((relative_height - 1.0) / 0.05) ** 2
            )
            joint_deviation_penalty = (
                -0.02 * float(np.mean((joint_error / 0.5) ** 2))
            )
            action_delta_penalty = (
                -0.005 * float(np.mean((action_delta / 0.25) ** 2))
            )
            angular_velocity_penalty = (
                -0.02 * float(np.dot(self.data.qvel[3:6], self.data.qvel[3:6]))
            )
            # Contact-point velocities can contain brief impact spikes.  Keep
            # the physically scaled slip objective, but bound its per-step
            # influence so it cannot erase all posture/height learning signal.
            slip_penalty = -0.10 * min(slip / (0.10 ** 2), 4.0)
            base_reward = (alive_reward + world_upright_reward
                           + support_relative_height_reward + joint_deviation_penalty
                           + action_delta_penalty + angular_velocity_penalty
                           + slip_penalty + fall_penalty)
        else:
            alive_reward = 1.0
            world_upright_reward = math.exp(-4.0 * (roll * roll + pitch * pitch))
            support_relative_height_reward = (
                0.5 * math.exp(-40.0 * (relative_height - 1.0) ** 2)
            )
            joint_deviation_penalty = -0.05 * float(np.mean(joint_error ** 2))
            action_delta_penalty = -0.01 * float(np.mean(action_delta ** 2))
            angular_velocity_penalty = 0.0
            slip_penalty = 0.0
            base_reward = (alive_reward + world_upright_reward
                           + support_relative_height_reward + joint_deviation_penalty
                           + action_delta_penalty + fall_penalty)
        com_error = self._com_support_error()
        com_support = math.exp(-float(np.dot(com_error, com_error)) / (2.0 * 0.12 ** 2))
        angular_velocity_squared = float(np.dot(self.data.qvel[3:6], self.data.qvel[3:6]))
        dual_contact = bool(left and right)
        if self.reward_mode in {"base", "corrected"}:
            reward = base_reward
        elif self.reward_mode == "base_plus_com":
            reward = base_reward + 0.5 * com_support
        else:
            reward = (base_reward + 0.5 * com_support - 0.05 * angular_velocity_squared
                      - 0.10 * slip + 0.10 * float(dual_contact))
        self.previous_action = action.copy()
        info = self._info(fall)
        info["reward_components"] = {
            "reward_mode": self.reward_mode, "base_reward": base_reward,
            "alive_reward": alive_reward,
            "world_upright_reward": world_upright_reward,
            "support_relative_height_reward": support_relative_height_reward,
            "joint_deviation_penalty": joint_deviation_penalty,
            "action_delta_penalty": action_delta_penalty,
            "angular_velocity_penalty": angular_velocity_penalty,
            "slip_penalty": slip_penalty,
            "fall_penalty": fall_penalty,
            "support_relative_base_height_m": relative_height,
            "com_support": com_support, "com_support_error_xy_m": com_error.tolist(),
            "base_angular_velocity_squared": angular_velocity_squared,
            "contacting_foot_slip_squared_speed": slip,
            "dual_foot_contact": float(dual_contact),
        }
        info["constraint_costs"] = self.constraint_costs()
        info["axis_separated_constraint_costs"] = self.axis_separated_constraint_costs()
        return self._observation(pose), float(reward), fall, self.steps >= self.max_steps, info

    def close(self) -> None:
        pass
