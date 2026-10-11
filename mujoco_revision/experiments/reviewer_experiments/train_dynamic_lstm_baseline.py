"""Auditable Dynamic-LSTM-47 baseline training on the MuJoCo H1-2 deck.

The 2,048-step setting is an engineering smoke only.  Pilot/formal runs are
resumable, write every PPO update to CSV, and sample deck amplitudes within the
pre-registered 0--5 degree training range.  Formal evaluation seeds are never
used here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
sys.path[:0] = [str(ROOT), str(BASE)]

from dmppo.algorithms.ppo import compute_gae  # noqa: E402
from h1_2_official_deck_env import H12OfficialDeckEnv  # noqa: E402
from train_recurrent_finetune_ppo import evaluate, load_common, recurrent_ppo_update  # noqa: E402

METHOD = "Dynamic-LSTM-47"
DEVELOPMENT_SEEDS = (303, 505, 1101, 1303)
REWARD_MODE = "corrected"
CSV_FIELDS = ("global_step", "update", "episodes", "mean_roll_deg", "mean_pitch_deg",
              "mean_rollout_reward", "mean_world_upright_reward",
              "mean_support_relative_height_reward", "mean_joint_deviation_penalty",
              "mean_action_delta_penalty", "mean_angular_velocity_penalty",
              "mean_slip_penalty", "mean_fall_penalty",
              "policy_loss", "value_loss", "entropy",
              "grad_norm_before_clip", "approx_kl", "epochs", "elapsed_wall_s")
REWARD_COMPONENTS = ("world_upright_reward", "support_relative_height_reward",
                     "joint_deviation_penalty", "action_delta_penalty",
                     "angular_velocity_penalty", "slip_penalty", "fall_penalty")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def sample_and_reset(env: H12OfficialDeckEnv, rng: np.random.Generator,
                     episode_seed: int, minimum: float, maximum: float):
    roll = float(rng.uniform(minimum, maximum))
    pitch = float(rng.uniform(minimum, maximum))
    env.deck_cfg["deck"]["roll_amp_deg"] = roll
    env.deck_cfg["deck"]["pitch_amp_deg"] = pitch
    obs, info = env.reset(seed=episode_seed)
    return obs, info, roll, pitch


def save_checkpoint(path: Path, model, optimizer, global_step: int, update: int,
                    episodes: int, rng: np.random.Generator, args: argparse.Namespace) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "update": update,
        "episodes": episodes,
        "numpy_generator_state": rng.bit_generator.state,
        "method": METHOD,
        "reward_mode": REWARD_MODE,
        "train_seed": args.seed,
        "train_amplitude_min_deg": args.train_amplitude_min_deg,
        "train_amplitude_max_deg": args.train_amplitude_max_deg,
        "rollout_steps": args.rollout_steps,
        "episode_duration_s": args.episode_duration,
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=262144)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--episode-duration", type=float, default=20.0)
    parser.add_argument("--development-eval-duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-amplitude-min-deg", type=float, default=0.0)
    parser.add_argument("--train-amplitude-max-deg", type=float, default=5.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=65536)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.total_steps <= 0 or args.rollout_steps <= 0:
        raise ValueError("step counts must be positive")
    if not 0.0 <= args.train_amplitude_min_deg <= args.train_amplitude_max_deg:
        raise ValueError("invalid amplitude range")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_common()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)
    global_step = update = episodes = 0
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        if payload.get("method") != METHOD or int(payload.get("train_seed")) != args.seed:
            raise ValueError("resume checkpoint method/train seed mismatch")
        if payload.get("reward_mode") != REWARD_MODE:
            raise ValueError("resume checkpoint reward mode mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        global_step = int(payload["global_step"])
        update = int(payload.get("update", 0))
        episodes = int(payload.get("episodes", 0))
        rng.bit_generator.state = payload["numpy_generator_state"]

    manifest = {
        "method": METHOD,
        "observation_schema": "Unitree proprioception 47-D",
        "actor_architecture": "LSTM(47,64)-MLP(64,32,12)",
        "actor_deck_truth": False,
        "reward_mode": REWARD_MODE,
        "reward_audit": "runs/P1_final_reward_knee_audit_20260926",
        "train_seed": args.seed,
        "formal_evaluation_seeds_used_for_training": False,
        "train_amplitude_range_deg": [args.train_amplitude_min_deg, args.train_amplitude_max_deg],
        "amplitude_sampling": "independent uniform roll and pitch per episode",
        "total_steps_target": args.total_steps,
        "rollout_steps": args.rollout_steps,
        "episode_duration_s": args.episode_duration,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "source_sha256": {path.name: file_sha256(path) for path in
                          (Path(__file__), HERE / "formal_protocol.json")},
        "versions": {"python": sys.version, "platform": platform.platform(),
                     "numpy": np.__version__, "torch": torch.__version__,
                     "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "RUNNING.json").write_text(json.dumps(
        {"global_step": global_step, "target": args.total_steps}, indent=2), encoding="utf-8")

    initial = evaluate(model, DEVELOPMENT_SEEDS, args.development_eval_duration, REWARD_MODE,
                       args.train_amplitude_max_deg, args.train_amplitude_max_deg)
    env = H12OfficialDeckEnv(episode_duration=args.episode_duration, randomize_phase=True,
                            reward_mode=REWARD_MODE, deck_roll_amp_deg=0.0, deck_pitch_amp_deg=0.0)
    obs, _, roll_deg, pitch_deg = sample_and_reset(
        env, rng, args.seed + episodes, args.train_amplitude_min_deg, args.train_amplitude_max_deg
    )
    state = model.initial_state(1)
    episode_start = 1.0
    start_wall = time.time()
    last_checkpoint_step = global_step
    update_path = args.output_dir / "train_updates.csv"
    with update_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        while global_step < args.total_steps:
            count = min(args.rollout_steps, args.total_steps - global_step)
            rollout_initial_state = tuple(item.detach().clone().numpy() for item in state)
            buf = {name: [] for name in
                   ("obs", "actions", "log_probs", "rewards", "dones", "values", "episode_starts")}
            sampled_roll, sampled_pitch = [], []
            reward_components = {key: [] for key in REWARD_COMPONENTS}
            model.eval()
            for _ in range(count):
                with torch.no_grad():
                    action, logp, value, next_state = model.act_step(
                        torch.from_numpy(obs).unsqueeze(0), state, deterministic=False
                    )
                action_np = action.squeeze(0).numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                for key in REWARD_COMPONENTS:
                    reward_components[key].append(float(info["reward_components"][key]))
                done = terminated or truncated
                for name, item in (("obs", obs), ("actions", action_np), ("log_probs", float(logp)),
                                   ("rewards", reward), ("dones", float(done)),
                                   ("values", float(value)), ("episode_starts", episode_start)):
                    buf[name].append(item)
                sampled_roll.append(roll_deg)
                sampled_pitch.append(pitch_deg)
                obs, state, episode_start = next_obs, next_state, 0.0
                global_step += 1
                if done:
                    episodes += 1
                    obs, _, roll_deg, pitch_deg = sample_and_reset(
                        env, rng, args.seed + episodes, args.train_amplitude_min_deg,
                        args.train_amplitude_max_deg,
                    )
                    state = model.initial_state(1)
                    episode_start = 1.0
            with torch.no_grad():
                last_value = float(model.critic(torch.from_numpy(obs).unsqueeze(0)).squeeze())
            advantages, returns = compute_gae(
                np.asarray(buf["rewards"], np.float32), np.asarray(buf["values"], np.float32),
                np.asarray(buf["dones"], np.float32), last_value, 0.99, 0.95,
            )
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
                   **metrics, "elapsed_wall_s": time.time() - start_wall}
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            (args.output_dir / "RUNNING.json").write_text(json.dumps(
                {"global_step": global_step, "target": args.total_steps, "update": update,
                 "episodes": episodes, "last_update": row}, indent=2), encoding="utf-8")
            if global_step - last_checkpoint_step >= args.checkpoint_every_steps:
                save_checkpoint(args.output_dir / f"checkpoint_step_{global_step}.pt", model, optimizer,
                                global_step, update, episodes, rng, args)
                last_checkpoint_step = global_step
    env.close()

    final_checkpoint = args.output_dir / "checkpoint_final.pt"
    save_checkpoint(final_checkpoint, model, optimizer, global_step, update, episodes, rng, args)
    final = evaluate(model, DEVELOPMENT_SEEDS, args.development_eval_duration, REWARD_MODE,
                     args.train_amplitude_max_deg, args.train_amplitude_max_deg)
    subprocess.run([sys.executable, str(HERE / "plot_experiment_results.py"),
                    "--run-dir", str(args.output_dir)], check=True)
    completed = {"method": METHOD, "train_seed": args.seed, "global_step": global_step,
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
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__, "message": str(exc)}),
              file=sys.stderr, flush=True)
        raise
