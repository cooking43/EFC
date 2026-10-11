"""Auditable recurrent PPO trainer for the four fair controller ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import gymnasium
import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments" / "h1_deck_balance"
sys.path[:0] = [str(ROOT), str(BASE), str(HERE)]

from dmppo.algorithms.ppo import compute_gae  # noqa: E402
from fair_controller_ablation import (  # noqa: E402
    CONTROL_DT_S, FILTER_NEW_WEIGHT, KNEE_DIRECTION_SIGN, METHODS, OBSERVATION_DIM,
    PREDICTION_HORIZON_S, PREDICTOR_ORDER, RESIDUAL_LIMIT_RAD,
    ROLL_TO_KNEE_GAIN, FairAblationEnv, file_sha256,
    load_predictor_augmented_common, observation_schema,
)
from train_recurrent_finetune_ppo import recurrent_ppo_update  # noqa: E402

DEVELOPMENT_SEEDS = (303, 505, 1101, 1303)
REWARD_MODE = "corrected"
CSV_FIELDS = ("global_step", "update", "episodes", "mean_roll_deg", "mean_pitch_deg",
              "mean_rollout_reward", "mean_world_upright_reward",
              "mean_support_relative_height_reward", "mean_joint_deviation_penalty",
              "mean_action_delta_penalty", "mean_angular_velocity_penalty",
              "mean_slip_penalty", "mean_fall_penalty", "mean_residual_abs_rad",
              "policy_loss", "value_loss",
              "entropy", "grad_norm_before_clip", "approx_kl", "epochs", "elapsed_wall_s")
REWARD_COMPONENTS = ("world_upright_reward", "support_relative_height_reward",
                     "joint_deviation_penalty", "action_delta_penalty",
                     "angular_velocity_penalty", "slip_penalty", "fall_penalty")


def sample_and_reset(env: FairAblationEnv, rng: np.random.Generator, episode_seed: int,
                     minimum: float, maximum: float):
    roll = float(rng.uniform(minimum, maximum)); pitch = float(rng.uniform(minimum, maximum))
    env.unwrapped.deck_cfg["deck"]["roll_amp_deg"] = roll
    env.unwrapped.deck_cfg["deck"]["pitch_amp_deg"] = pitch
    obs, info = env.reset(seed=episode_seed)
    return obs, info, roll, pitch


def evaluate(model, method: str, seeds: tuple[int, ...], duration: float,
             roll_deg: float, pitch_deg: float) -> dict:
    rows = []; model.eval()
    for seed in seeds:
        env = FairAblationEnv(method=method,
            episode_duration=duration, randomize_phase=True, reward_mode=REWARD_MODE,
            deck_roll_amp_deg=roll_deg, deck_pitch_amp_deg=pitch_deg)
        state = model.initial_state(1)
        try:
            obs, _ = env.reset(seed=seed); total = 0.0
            for _ in range(env.unwrapped.max_steps):
                with torch.inference_mode():
                    action, _, _, state = model.act_step(torch.from_numpy(obs).unsqueeze(0),
                                                         state, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action.squeeze(0).numpy())
                total += reward
                if terminated or truncated: break
            rows.append({"seed": seed, "elapsed_s": float(info["episode_time"]), "return": total,
                         "success": bool(truncated and not terminated), "fell": bool(terminated)})
        finally:
            env.close()
    return {"success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_elapsed_s": float(np.mean([row["elapsed_s"] for row in rows])), "episodes": rows}


def save_checkpoint(path: Path, model, optimizer, global_step: int, update: int,
                    episodes: int, rng: np.random.Generator, args) -> None:
    torch.save({"model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "global_step": global_step,
        "update": update, "episodes": episodes, "numpy_generator_state": rng.bit_generator.state,
        "method": args.method, "observation_schema": observation_schema(),
        "predictor_definition": {"order": PREDICTOR_ORDER,
            "horizon_s": PREDICTION_HORIZON_S, "control_dt_s": CONTROL_DT_S,
            "filter_new_weight": FILTER_NEW_WEIGHT,
            "derivative_scheme": "three_point_backward"},
        "reward_mode": REWARD_MODE, "train_seed": args.seed,
        "train_amplitude_min_deg": args.train_amplitude_min_deg,
        "train_amplitude_max_deg": args.train_amplitude_max_deg,
        "rollout_steps": args.rollout_steps, "episode_duration_s": args.episode_duration}, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--episode-duration", type=float, default=20.0)
    parser.add_argument("--development-eval-duration", type=float, default=20.0)
    parser.add_argument("--train-amplitude-min-deg", type=float, default=0.0)
    parser.add_argument("--train-amplitude-max-deg", type=float, default=5.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=8192)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    model = load_predictor_augmented_common()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)
    global_step = update = episodes = 0
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        if payload.get("method") != args.method or int(payload.get("train_seed")) != args.seed:
            raise ValueError("resume checkpoint method/train seed mismatch")
        if payload.get("reward_mode") != REWARD_MODE:
            raise ValueError("resume checkpoint reward mode mismatch")
        expected_predictor = {"order": PREDICTOR_ORDER, "horizon_s": PREDICTION_HORIZON_S,
                              "control_dt_s": CONTROL_DT_S,
                              "filter_new_weight": FILTER_NEW_WEIGHT,
                              "derivative_scheme": "three_point_backward"}
        if payload.get("predictor_definition") != expected_predictor:
            raise ValueError("resume predictor definition mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        global_step = int(payload["global_step"]); update = int(payload["update"])
        episodes = int(payload["episodes"]); rng.bit_generator.state = payload["numpy_generator_state"]
    manifest = {"method": args.method, "observation_schema": observation_schema(),
        "observation_dimension_from_schema": OBSERVATION_DIM,
        "actor_architecture": "LSTM(48,64)-MLP(64,32,12)",
        "shared_causal_prediction_for_actor": True,
        "explicit_residual_source": {"Prediction-to-Policy": "none",
            "Current-State-Residual": "filtered current platform roll",
            "Predictive-Residual": "filtered second-order 100 ms platform-roll prediction",
            "Oracle-Predictive-Residual": "future platform roll; upper bound only"}[args.method],
        "predictor": {"input": "platform roll sensor", "order": PREDICTOR_ORDER,
            "horizon_s": PREDICTION_HORIZON_S, "control_dt_s": CONTROL_DT_S,
            "filter_new_weight": FILTER_NEW_WEIGHT,
            "velocity_difference": "three-point backward, second-order accurate",
            "acceleration_difference": "three-point backward second difference"},
        "residual_mapping": "double support: left_knee=+clip(0.1*roll_deg); right_knee=-clip(0.1*roll_deg)",
        "residual_limit_rad": RESIDUAL_LIMIT_RAD,
        "roll_to_knee_gain_deg_per_deg": ROLL_TO_KNEE_GAIN,
        "knee_direction_sign": KNEE_DIRECTION_SIGN,
        "knee_sign_audit": "runs/P1_final_reward_knee_audit_20260926",
        "action_interface": "unchanged 12-D Unitree target offset",
        "reward_mode": REWARD_MODE, "train_seed": args.seed,
        "formal_evaluation_seeds_used_for_training": False,
        "train_amplitude_range_deg": [args.train_amplitude_min_deg, args.train_amplitude_max_deg],
        "amplitude_sampling": "independent uniform roll and pitch per episode",
        "total_steps_target": args.total_steps, "rollout_steps": args.rollout_steps,
        "episode_duration_s": args.episode_duration,
        "source_sha256": {path.name: file_sha256(path) for path in
                          (Path(__file__), HERE / "fair_controller_ablation.py",
                           HERE / "formal_protocol.json")},
        "versions": {"python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "RUNNING.json").write_text(json.dumps(
        {"global_step": global_step, "target": args.total_steps}, indent=2), encoding="utf-8")
    initial = evaluate(model, args.method, DEVELOPMENT_SEEDS, args.development_eval_duration,
                       args.train_amplitude_max_deg, args.train_amplitude_max_deg)
    env = FairAblationEnv(method=args.method,
        episode_duration=args.episode_duration, randomize_phase=True, reward_mode=REWARD_MODE,
        deck_roll_amp_deg=0.0, deck_pitch_amp_deg=0.0)
    obs, _, roll_deg, pitch_deg = sample_and_reset(env, rng, args.seed + episodes,
        args.train_amplitude_min_deg, args.train_amplitude_max_deg)
    state = model.initial_state(1); episode_start = 1.0; start_wall = time.time()
    last_checkpoint_step = global_step
    update_path = args.output_dir / "train_updates.csv"
    with update_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS); writer.writeheader()
        while global_step < args.total_steps:
            count = min(args.rollout_steps, args.total_steps - global_step)
            rollout_initial_state = tuple(item.detach().clone().numpy() for item in state)
            buf = {name: [] for name in ("obs", "actions", "log_probs", "rewards", "dones",
                                         "values", "episode_starts")}
            sampled_roll, sampled_pitch, residuals = [], [], []
            reward_components = {key: [] for key in REWARD_COMPONENTS}
            model.eval()
            for _ in range(count):
                with torch.no_grad():
                    action, logp, value, next_state = model.act_step(
                        torch.from_numpy(obs).unsqueeze(0), state, deterministic=False)
                action_np = action.squeeze(0).numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                for name, item in (("obs", obs), ("actions", action_np), ("log_probs", float(logp)),
                    ("rewards", reward), ("dones", float(done)), ("values", float(value)),
                    ("episode_starts", episode_start)): buf[name].append(item)
                sampled_roll.append(roll_deg); sampled_pitch.append(pitch_deg)
                residuals.append(float(info["ablation"]["residual_max_abs_rad"]))
                for key in REWARD_COMPONENTS:
                    reward_components[key].append(float(info["reward_components"][key]))
                obs, state, episode_start = next_obs, next_state, 0.0; global_step += 1
                if done:
                    episodes += 1
                    obs, _, roll_deg, pitch_deg = sample_and_reset(env, rng, args.seed + episodes,
                        args.train_amplitude_min_deg, args.train_amplitude_max_deg)
                    state = model.initial_state(1); episode_start = 1.0
            with torch.no_grad():
                last_value = float(model.critic(torch.from_numpy(obs).unsqueeze(0)).squeeze())
            advantages, returns = compute_gae(np.asarray(buf["rewards"], np.float32),
                np.asarray(buf["values"], np.float32), np.asarray(buf["dones"], np.float32),
                last_value, 0.99, 0.95)
            batch = {name: np.asarray(value, np.float32) for name, value in buf.items()}
            batch["initial_state"] = rollout_initial_state
            batch["advantages"], batch["returns"] = advantages, returns
            metrics = recurrent_ppo_update(model, optimizer, batch, epochs=4, clip_epsilon=0.2)
            update += 1
            row = {"global_step": global_step, "update": update, "episodes": episodes,
                "mean_roll_deg": float(np.mean(sampled_roll)),
                "mean_pitch_deg": float(np.mean(sampled_pitch)),
                "mean_rollout_reward": float(np.mean(buf["rewards"])),
                **{f"mean_{key}": float(np.mean(values))
                   for key, values in reward_components.items()},
                "mean_residual_abs_rad": float(np.mean(residuals)), **metrics,
                "elapsed_wall_s": time.time() - start_wall}
            writer.writerow(row); stream.flush(); print(json.dumps(row), flush=True)
            (args.output_dir / "RUNNING.json").write_text(json.dumps(
                {"global_step": global_step, "target": args.total_steps, "update": update,
                 "episodes": episodes, "last_update": row}, indent=2), encoding="utf-8")
            if global_step - last_checkpoint_step >= args.checkpoint_every_steps:
                save_checkpoint(args.output_dir / f"checkpoint_step_{global_step}.pt", model,
                    optimizer, global_step, update, episodes, rng, args)
                last_checkpoint_step = global_step
    env.close()
    final_checkpoint = args.output_dir / "checkpoint_final.pt"
    save_checkpoint(final_checkpoint, model, optimizer, global_step, update, episodes, rng, args)
    final = evaluate(model, args.method, DEVELOPMENT_SEEDS, args.development_eval_duration,
                     args.train_amplitude_max_deg, args.train_amplitude_max_deg)
    subprocess.run([sys.executable, str(HERE / "plot_experiment_results.py"),
                    "--run-dir", str(args.output_dir)], check=True)
    completed = {"method": args.method, "train_seed": args.seed, "global_step": global_step,
        "updates": update, "episodes": episodes, "initial_development": initial,
        "final_development": final, "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": file_sha256(final_checkpoint),
        "elapsed_wall_s": time.time() - start_wall,
        "interpretation": "Development result only; not a formal method comparison."}
    (args.output_dir / "COMPLETED.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
    print(json.dumps(completed, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__, "message": str(exc)}),
              file=sys.stderr, flush=True)
        raise
