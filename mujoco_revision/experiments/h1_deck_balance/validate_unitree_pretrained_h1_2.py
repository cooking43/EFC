"""Headless validation of Unitree's unmodified H1-2 MuJoCo policy baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import yaml

from deck_motion import get_deck_pose

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "external" / "unitree_rl_gym"
CONFIG = UPSTREAM / "deploy" / "deploy_mujoco" / "configs" / "h1_2.yaml"


def gravity_orientation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.asarray([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float32)


def quat_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    qw, qx, qy, qz = map(float, q)
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    return roll, pitch


def resolve(value: str) -> Path:
    return Path(value.replace("{LEGGED_GYM_ROOT_DIR}", str(UPSTREAM)))


def run(command_x: float, duration: float, rocking_deck: bool = False,
        viewer_enabled: bool = False,
        deck_contact_timeconst_s: float | None = None,
        phase_seed: int | None = None) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    xml_path = (UPSTREAM / "resources" / "robots" / "h1_2" / "deck_scene.xml"
                if rocking_deck else resolve(cfg["xml_path"]))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(cfg["simulation_dt"])
    policy = torch.jit.load(str(resolve(cfg["policy_path"])), map_location="cpu")
    policy.eval()

    kp = np.asarray(cfg["kps"], dtype=np.float64)
    kd = np.asarray(cfg["kds"], dtype=np.float64)
    default = np.asarray(cfg["default_angles"], dtype=np.float64)
    command = np.asarray([command_x, 0.0, 0.0], dtype=np.float32)
    command_scale = np.asarray(cfg["cmd_scale"], dtype=np.float32)
    action = np.zeros(int(cfg["num_actions"]), dtype=np.float32)
    target = default.copy()
    obs = np.zeros(int(cfg["num_obs"]), dtype=np.float32)
    decimation = int(cfg["control_decimation"])
    action_scale = float(cfg["action_scale"])
    counter = 0
    max_roll = max_pitch = max_action = 0.0
    min_height = float("inf")
    fall_time = None
    roll_samples: list[float] = []
    pitch_samples: list[float] = []
    action_samples: list[float] = []
    joint_offset_samples: list[float] = []
    slip_samples: list[float] = []
    left_contact_steps = right_contact_steps = sampled_steps = 0
    deck_cfg = None
    deck_mocap_id = None
    support_geom_id = None
    foot_geom_ids: set[int] = set()
    contact_distances: list[float] = []
    settled_contact_distances: list[float] = []
    if rocking_deck:
        deck_cfg = yaml.safe_load((HERE / "configs" / "deck_stand.yaml").read_text(encoding="utf-8"))
        deck_cfg["deck"]["roll_amp_deg"] = 1.0
        deck_cfg["deck"]["pitch_amp_deg"] = 1.0
        deck_cfg["deck"]["heave_amp_m"] = 0.0
        if phase_seed is not None:
            rng = np.random.default_rng(phase_seed)
            for key in ("roll_phase_rad", "pitch_phase_rad", "heave_phase_rad"):
                deck_cfg["deck"][key] = float(rng.uniform(0.0, 2.0 * math.pi))
        deck_bid = model.body("oscillating_deck").id
        deck_mocap_id = int(model.body_mocapid[deck_bid])
        support_geom_id = model.geom("oscillating_deck_geom").id
        if deck_contact_timeconst_s is not None:
            if deck_contact_timeconst_s <= 0.0:
                raise ValueError("deck_contact_timeconst_s must be positive")
            model.geom_solref[support_geom_id] = (deck_contact_timeconst_s, 1.0)
    else:
        support_geom_id = model.geom("floor").id
    for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        body_id = model.body(body_name).id
        foot_geom_ids.update(
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
        )

    viewer = None
    if viewer_enabled:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
    try:
      while data.time < duration and (viewer is None or viewer.is_running()):
        step_start = time.perf_counter()
        if rocking_deck:
            pose = get_deck_pose(float(data.time), deck_cfg)
            data.mocap_pos[deck_mocap_id] = pose["position"]
            data.mocap_quat[deck_mocap_id] = pose["quaternion"]
        # Match Unitree's deploy_mujoco.py exactly.  These motors have
        # ctrllimited=false, so actuator_ctrlrange contains placeholder zeros
        # and must not be used for clipping.
        data.ctrl[:] = (target - data.qpos[7:]) * kp - data.qvel[6:] * kd
        mujoco.mj_step(model, data)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            pair = {int(contact.geom1), int(contact.geom2)}
            if support_geom_id in pair and pair.intersection(foot_geom_ids):
                contact_distances.append(float(contact.dist))
                if data.time >= 1.0:
                    settled_contact_distances.append(float(contact.dist))
        counter += 1
        if counter % decimation == 0:
            obs[:3] = data.qvel[3:6] * float(cfg["ang_vel_scale"])
            obs[3:6] = gravity_orientation(data.qpos[3:7])
            obs[6:9] = command * command_scale
            obs[9:21] = (data.qpos[7:] - default) * float(cfg["dof_pos_scale"])
            obs[21:33] = data.qvel[6:] * float(cfg["dof_vel_scale"])
            obs[33:45] = action
            phase = (counter * model.opt.timestep % 0.8) / 0.8
            obs[45:47] = (math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase))
            with torch.inference_mode():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target = action * action_scale + default
            max_action = max(max_action, float(np.max(np.abs(action))))

        roll, pitch = quat_roll_pitch(data.qpos[3:7])
        height = float(data.qpos[2])
        if counter % decimation == 0:
            roll_samples.append(roll); pitch_samples.append(pitch)
            action_samples.append(float(np.mean(np.square(action))))
            joint_offset_samples.append(float(np.mean(np.square(data.qpos[7:] - default))))
            if rocking_deck:
                left_active = right_active = False
                left_body = model.body("left_ankle_roll_link").id
                right_body = model.body("right_ankle_roll_link").id
                deck_body = model.body("oscillating_deck").id
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if support_geom_id not in pair:
                        continue
                    other = int(contact.geom2 if int(contact.geom1) == support_geom_id else contact.geom1)
                    body_id = int(model.geom_bodyid[other])
                    left_active |= body_id == left_body
                    right_active |= body_id == right_body
                    if body_id not in (left_body, right_body):
                        continue
                    point = np.asarray(contact.pos)
                    foot_point_velocity = (
                        data.cvel[body_id, 3:6]
                        + np.cross(data.cvel[body_id, :3], point - data.xpos[body_id])
                    )
                    deck_point_velocity = (
                        data.cvel[deck_body, 3:6]
                        + np.cross(data.cvel[deck_body, :3], point - data.xpos[deck_body])
                    )
                    relative = foot_point_velocity - deck_point_velocity
                    normal = np.asarray(contact.frame[:3])
                    tangential = relative - np.dot(relative, normal) * normal
                    slip_samples.append(float(np.dot(tangential, tangential)))
                left_contact_steps += int(left_active)
                right_contact_steps += int(right_active)
                sampled_steps += 1
        max_roll = max(max_roll, abs(roll)); max_pitch = max(max_pitch, abs(pitch))
        min_height = min(min_height, height)
        if height < 0.6 or abs(roll) > math.pi / 4 or abs(pitch) > math.pi / 4:
            fall_time = float(data.time)
            break
        if viewer is not None:
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()

    return {
        "command_x_m_s": command_x,
        "rocking_deck": rocking_deck,
        "phase_seed": phase_seed,
        "deck_contact_timeconst_s": deck_contact_timeconst_s,
        "elapsed_s": float(data.time),
        "fell": fall_time is not None,
        "fall_time_s": fall_time,
        "min_pelvis_height_m": min_height,
        "max_abs_roll_deg": math.degrees(max_roll),
        "max_abs_pitch_deg": math.degrees(max_pitch),
        "max_abs_policy_action": max_action,
        "base_roll_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(roll_samples))))),
        "base_pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(pitch_samples))))),
        "policy_action_rms": float(np.sqrt(np.mean(action_samples))),
        "joint_offset_rms_deg": float(np.degrees(np.sqrt(np.mean(joint_offset_samples)))),
        "contacting_foot_relative_slip_rms_m_s": (
            float(np.sqrt(np.mean(slip_samples))) if slip_samples else None
        ),
        "left_foot_contact_fraction": (
            left_contact_steps / sampled_steps if sampled_steps else None
        ),
        "right_foot_contact_fraction": (
            right_contact_steps / sampled_steps if sampled_steps else None
        ),
        "minimum_foot_support_contact_distance_m": (
            min(contact_distances) if contact_distances else None
        ),
        "mean_foot_support_contact_distance_m": (
            float(np.mean(contact_distances)) if contact_distances else None
        ),
        "minimum_settled_foot_support_contact_distance_m": (
            min(settled_contact_distances) if settled_contact_distances else None
        ),
        "mean_settled_foot_support_contact_distance_m": (
            float(np.mean(settled_contact_distances)) if settled_contact_distances else None
        ),
        "finite_state": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=HERE / "results" / "unitree_pretrained_h1_2" / "summary.json")
    args = parser.parse_args()
    if args.viewer:
        result = {
            "source": "unitreerobotics/unitree_rl_gym",
            "viewer": True,
            "cases": [run(0.0, args.duration, rocking_deck=True, viewer_enabled=True)],
        }
    else:
        result = {
        "source": "unitreerobotics/unitree_rl_gym",
        "upstream_files_modified": False,
        "cases": [
            run(0.0, args.duration),
            run(0.5, args.duration),
            run(0.0, args.duration, rocking_deck=True),
        ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
