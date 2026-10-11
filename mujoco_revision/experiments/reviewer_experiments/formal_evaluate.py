"""Matched-seed, event-level evaluation for H1-2 recurrent PPO checkpoints.

This script deliberately writes one CSV row per episode.  It never feeds deck
truth to the actor; deck phases are logged only so that a run can be audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Iterable

import gymnasium
import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments" / "h1_deck_balance"
sys.path[:0] = [str(ROOT), str(BASE)]

from h1_2_official_deck_env import H12OfficialDeckEnv, UPSTREAM  # noqa: E402
from fair_controller_ablation import (  # noqa: E402
    FairAblationEnv,
    METHODS as FAIR_ABLATION_METHODS,
    load_predictor_augmented_common,
)
from state_enhanced_lstm import (  # noqa: E402
    StateEnhancedH12Env,
    load_state_enhanced_common,
)
from unitree_recurrent_actor_critic import UnitreeRecurrentActorCritic  # noqa: E402

POLICY = UPSTREAM / "deploy" / "pre_train" / "h1_2" / "motion.pt"
PROTOCOL = HERE / "formal_protocol.json"
CSV_FIELDS = (
    "method", "train_seed", "checkpoint", "checkpoint_sha256", "condition",
    "amplitude_deg", "eval_seed", "duration_limit_s", "elapsed_s", "success",
    "fell", "fall_reason", "return", "base_roll_rmse_rad", "base_pitch_rmse_rad",
    "base_tilt_rmse_rad", "contacting_foot_slip_rms_m_s", "action_delta_rms",
    "dual_contact_fraction", "roll_phase_rad", "pitch_phase_rad",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> tuple[str, int, Path]:
    """Parse METHOD:TRAIN_SEED:PATH without imposing a fake observation label."""
    try:
        method, seed, raw_path = value.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint must be METHOD:TRAIN_SEED:PATH") from exc
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint not found: {path}")
    return method, int(seed), path


def load_model(path: Path, method: str) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in payload:
        raise ValueError(f"checkpoint has no model_state_dict: {path}")
    if method == "Dynamic-LSTM-47":
        model = UnitreeRecurrentActorCritic(POLICY, initial_std=0.10)
    elif method == "State-Enhanced-LSTM":
        model = load_state_enhanced_common()
    elif method in FAIR_ABLATION_METHODS:
        model = load_predictor_augmented_common()
    else:
        raise ValueError(f"unsupported evaluation method: {method}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def make_env(method: str, duration: float, reward_mode: str,
             roll_deg: float, pitch_deg: float,
             sensor_kwargs: dict | None = None) -> gymnasium.Env:
    env_kwargs = {
        "episode_duration": duration,
        "randomize_phase": True,
        "reward_mode": reward_mode,
        "deck_roll_amp_deg": roll_deg,
        "deck_pitch_amp_deg": pitch_deg,
    }
    if method == "Dynamic-LSTM-47":
        return H12OfficialDeckEnv(**env_kwargs)
    if method == "State-Enhanced-LSTM":
        return StateEnhancedH12Env(**env_kwargs)
    if method in FAIR_ABLATION_METHODS:
        return FairAblationEnv(method=method, **(sensor_kwargs or {}), **env_kwargs)
    raise ValueError(f"unsupported evaluation method: {method}")


def condition_amplitudes(condition: str, amplitude: float) -> tuple[float, float]:
    if condition == "roll":
        return amplitude, 0.0
    if condition == "pitch":
        return 0.0, amplitude
    if condition == "combined":
        return amplitude, amplitude
    raise ValueError(condition)


def fall_reason(info: dict) -> str:
    if not info["fall_flag"]:
        return "administrative_censoring"
    if not all(math.isfinite(float(info[key])) for key in ("base_height", "base_roll", "base_pitch")):
        return "nonfinite_state"
    if float(info["base_height"]) < 0.6:
        return "base_height"
    if abs(float(info["base_roll"])) > math.pi / 4:
        return "base_roll"
    if abs(float(info["base_pitch"])) > math.pi / 4:
        return "base_pitch"
    return "fall_other"


def run_episode(model: torch.nn.Module, method: str, train_seed: int,
                checkpoint: Path, checkpoint_hash: str, condition: str,
                amplitude: float, seed: int, duration: float, reward_mode: str) -> dict:
    roll_deg, pitch_deg = condition_amplitudes(condition, amplitude)
    env = make_env(method, duration, reward_mode, roll_deg, pitch_deg)
    try:
        obs, _ = env.reset(seed=seed)
        base_env = env.unwrapped
        phases = {key: float(base_env.deck_cfg["deck"].get(key, 0.0)) for key in
                  ("roll_phase_rad", "pitch_phase_rad")}
        state = model.initial_state(1)
        previous = np.zeros(12, dtype=np.float64)
        total_return = 0.0
        roll_sq: list[float] = []
        pitch_sq: list[float] = []
        slip_sq: list[float] = []
        delta_sq: list[float] = []
        dual: list[float] = []
        terminated = truncated = False
        info: dict = {}
        for _ in range(base_env.max_steps):
            with torch.inference_mode():
                action_t, _, _, state = model.act_step(
                    torch.from_numpy(obs).unsqueeze(0), state, deterministic=True
                )
            action = action_t.squeeze(0).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            roll_sq.append(float(info["base_roll"]) ** 2)
            pitch_sq.append(float(info["base_pitch"]) ** 2)
            slip_sq.append(float(info["foot_slip_squared_speed"]))
            delta_sq.append(float(np.mean(np.square(action - previous))))
            dual.append(float(info["left_foot_contact"] and info["right_foot_contact"]))
            previous = action
            if terminated or truncated:
                break
        return {
            "method": method,
            "train_seed": train_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "condition": condition,
            "amplitude_deg": amplitude,
            "eval_seed": seed,
            "duration_limit_s": duration,
            "elapsed_s": float(info.get("episode_time", info.get("time", 0.0))),
            "success": int(truncated and not terminated),
            "fell": int(terminated),
            "fall_reason": fall_reason(info),
            "return": total_return,
            "base_roll_rmse_rad": math.sqrt(float(np.mean(roll_sq))),
            "base_pitch_rmse_rad": math.sqrt(float(np.mean(pitch_sq))),
            "base_tilt_rmse_rad": math.sqrt(float(np.mean(np.asarray(roll_sq) + np.asarray(pitch_sq)))),
            "contacting_foot_slip_rms_m_s": math.sqrt(float(np.mean(slip_sq))),
            "action_delta_rms": math.sqrt(float(np.mean(delta_sq))),
            "dual_contact_fraction": float(np.mean(dual)),
            **phases,
        }
    finally:
        env.close()


def mean_ci(values: Iterable[float], rng: np.random.Generator,
            replicates: int = 10000) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    samples = rng.choice(array, size=(replicates, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(array.mean()), "ci_low": float(low), "ci_high": float(high),
            "n": int(array.size)}


def summarize(rows: list[dict]) -> dict:
    rng = np.random.default_rng(20260923)
    result: dict[str, dict] = {}
    keys = sorted({(row["method"], row["train_seed"], row["condition"],
                   row["amplitude_deg"]) for row in rows})
    metrics = ("success", "elapsed_s", "base_tilt_rmse_rad",
               "contacting_foot_slip_rms_m_s", "action_delta_rms")
    for method, train_seed, condition, amplitude in keys:
        group = [row for row in rows if (row["method"], row["train_seed"], row["condition"],
                                         row["amplitude_deg"]) ==
                 (method, train_seed, condition, amplitude)]
        key = f"{method}/seed{train_seed}/{condition}/{amplitude:g}deg"
        result[key] = {metric: mean_ci((float(row[metric]) for row in group), rng)
                       for metric in metrics}
        result[key]["elapsed_s"]["name"] = "restricted_mean_survival_time_s"
    return result


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint,
                        help="Repeat METHOD:TRAIN_SEED:PATH")
    parser.add_argument("--conditions", default="combined")
    parser.add_argument("--amplitudes", default=",".join(
        map(str, protocol["evaluation"]["in_distribution_amplitudes_deg"])))
    parser.add_argument("--duration", type=float, default=protocol["evaluation"]["duration_s"])
    parser.add_argument("--reward-mode", default=protocol["training_reward"]["mode"])
    parser.add_argument("--seeds", default=",".join(map(str, protocol["evaluation"]["matched_seeds"])))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    amplitudes = tuple(float(item) for item in args.amplitudes.split(",") if item.strip())
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    rows: list[dict] = []
    csv_path = args.output_dir / "evaluation_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for method, train_seed, checkpoint in args.checkpoint:
            model = load_model(checkpoint, method)
            checkpoint_hash = sha256(checkpoint)
            for condition in conditions:
                for amplitude in amplitudes:
                    for seed in seeds:
                        row = run_episode(model, method, train_seed, checkpoint, checkpoint_hash,
                                          condition, amplitude, seed, args.duration,
                                          args.reward_mode)
                        rows.append(row)
                        writer.writerow(row)
                        stream.flush()
    manifest = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "conditions": conditions,
        "amplitudes_deg": amplitudes,
        "eval_seeds": seeds,
        "duration_s": args.duration,
        "reward_mode": args.reward_mode,
        "actor_deck_truth": False,
        "versions": {"python": sys.version, "platform": platform.platform(),
                     "numpy": np.__version__, "torch": torch.__version__,
                     "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    statistics = summarize(rows)
    (args.output_dir / "statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    (args.output_dir / "COMPLETED.json").write_text(json.dumps(
        {"episodes": len(rows), "csv": str(csv_path), "statistics": "statistics.json"}, indent=2
    ), encoding="utf-8")
    print(json.dumps({"episodes": len(rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
