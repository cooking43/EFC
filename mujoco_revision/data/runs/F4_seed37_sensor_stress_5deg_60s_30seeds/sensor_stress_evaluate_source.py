"""Frozen-policy sensitivity to causal platform-attitude sensor imperfections.

The experiment changes one sensor factor at a time.  Noise and bias are angle
errors on the roll channel supplied to the causal predictor; latency is an
integer number of 50 Hz control samples.  MuJoCo deck truth is used only for
offline error labels.  The Dynamic-LSTM reference has no platform-roll input
and is therefore evaluated once under the clean condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from formal_evaluate import (  # noqa: E402
    PROTOCOL, condition_amplitudes, fall_reason, load_model, make_env,
    parse_checkpoint, sha256,
)
from paired_statistics import exact_mcnemar, paired_continuous  # noqa: E402


SCENARIOS = (
    {"name": "clean", "noise_std_deg": 0.0, "bias_deg": 0.0, "latency_ms": 0},
    {"name": "noise_nominal", "noise_std_deg": 0.1, "bias_deg": 0.0, "latency_ms": 0},
    {"name": "noise_high", "noise_std_deg": 0.5, "bias_deg": 0.0, "latency_ms": 0},
    {"name": "bias_nominal", "noise_std_deg": 0.0, "bias_deg": 0.25, "latency_ms": 0},
    {"name": "bias_high", "noise_std_deg": 0.0, "bias_deg": 1.0, "latency_ms": 0},
    {"name": "latency_20ms", "noise_std_deg": 0.0, "bias_deg": 0.0, "latency_ms": 20},
    {"name": "latency_40ms", "noise_std_deg": 0.0, "bias_deg": 0.0, "latency_ms": 40},
    {"name": "latency_80ms", "noise_std_deg": 0.0, "bias_deg": 0.0, "latency_ms": 80},
)
METHODS = (
    "Dynamic-LSTM-47", "Prediction-to-Policy",
    "Current-State-Residual", "Predictive-Residual",
)
METRICS = (
    "success", "elapsed_s", "base_tilt_rmse_rad",
    "contacting_foot_slip_rms_m_s", "action_delta_rms",
    "roll_prediction_rmse_rad", "roll_sensor_rmse_rad",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(values: list[float], rng: np.random.Generator,
              replicates: int = 10000) -> dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "n": 0}
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    samples = array[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(array.mean()), "ci95_low": float(low),
            "ci95_high": float(high), "n": int(array.size)}


def run_episode(model: torch.nn.Module, method: str, train_seed: int,
                checkpoint: Path, checkpoint_hash: str, scenario: dict,
                amplitude_deg: float, eval_seed: int, duration_s: float,
                reward_mode: str) -> dict:
    roll_deg, pitch_deg = condition_amplitudes("combined", amplitude_deg)
    sensor_kwargs = None
    if method != "Dynamic-LSTM-47":
        sensor_kwargs = {
            "roll_sensor_noise_std_deg": scenario["noise_std_deg"],
            "roll_sensor_bias_deg": scenario["bias_deg"],
            "roll_sensor_latency_ms": scenario["latency_ms"],
        }
    env = make_env(method, duration_s, reward_mode, roll_deg, pitch_deg, sensor_kwargs)
    try:
        obs, _ = env.reset(seed=eval_seed)
        base_env = env.unwrapped
        state = model.initial_state(1)
        previous = np.zeros(12, dtype=np.float64)
        total_return = 0.0
        roll_sq: list[float] = []
        pitch_sq: list[float] = []
        slip_sq: list[float] = []
        delta_sq: list[float] = []
        prediction_sq: list[float] = []
        sensor_sq: list[float] = []
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
            previous = action
            if method != "Dynamic-LSTM-47":
                ablation = info["ablation"]
                prediction = math.radians(float(
                    ablation["predictor_state"]["filtered_prediction_roll_deg"]
                ))
                future_truth = float(env._platform_roll(0.1))
                prediction_sq.append((prediction - future_truth) ** 2)
                sensor_sq.append((float(ablation["roll_sensor_measured_rad"])
                                  - float(ablation["roll_sensor_true_rad"])) ** 2)
            if terminated or truncated:
                break
        phases = {key: float(base_env.deck_cfg["deck"].get(key, 0.0)) for key in
                  ("roll_phase_rad", "pitch_phase_rad")}
        return {
            "scenario": scenario["name"], "method": method, "train_seed": train_seed,
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "condition": "combined", "amplitude_deg": amplitude_deg,
            "eval_seed": eval_seed, "duration_limit_s": duration_s,
            "roll_sensor_noise_std_deg": scenario["noise_std_deg"],
            "roll_sensor_bias_deg": scenario["bias_deg"],
            "roll_sensor_latency_ms": scenario["latency_ms"],
            "elapsed_s": float(info.get("episode_time", info.get("time", 0.0))),
            "success": int(truncated and not terminated), "fell": int(terminated),
            "fall_reason": fall_reason(info), "return": total_return,
            "base_tilt_rmse_rad": math.sqrt(float(np.mean(
                np.asarray(roll_sq) + np.asarray(pitch_sq)
            ))),
            "contacting_foot_slip_rms_m_s": math.sqrt(float(np.mean(slip_sq))),
            "action_delta_rms": math.sqrt(float(np.mean(delta_sq))),
            "roll_prediction_rmse_rad": (math.sqrt(float(np.mean(prediction_sq)))
                                          if prediction_sq else math.nan),
            "roll_sensor_rmse_rad": (math.sqrt(float(np.mean(sensor_sq)))
                                      if sensor_sq else math.nan),
            **phases,
        }
    finally:
        env.close()


def summarize(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(20260930)
    output: list[dict] = []
    keys = sorted({(row["scenario"], row["method"]) for row in rows})
    for scenario, method in keys:
        selected = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
        for metric in METRICS:
            summary = bootstrap([float(row[metric]) for row in selected], rng)
            output.append({"scenario": scenario, "method": method, "metric": metric, **summary})
    return output


def paired_vs_clean(rows: list[dict]) -> dict:
    rng = np.random.default_rng(20260930)
    output: dict[str, dict] = {}
    continuous = METRICS[1:]
    for method in METHODS[1:]:
        clean = {int(row["eval_seed"]): row for row in rows
                 if row["method"] == method and row["scenario"] == "clean"}
        for scenario in (item["name"] for item in SCENARIOS[1:]):
            stressed = {int(row["eval_seed"]): row for row in rows
                        if row["method"] == method and row["scenario"] == scenario}
            seeds = sorted(set(clean) & set(stressed))
            key = f"{method}/{scenario}_minus_clean"
            success_a = np.asarray([int(clean[seed]["success"]) for seed in seeds])
            success_b = np.asarray([int(stressed[seed]["success"]) for seed in seeds])
            result = {"completion": exact_mcnemar(success_a, success_b)}
            for metric in continuous:
                a = np.asarray([float(clean[seed][metric]) for seed in seeds])
                b = np.asarray([float(stressed[seed][metric]) for seed in seeds])
                finite = np.isfinite(a) & np.isfinite(b)
                result[metric] = paired_continuous(a[finite], b[finite], rng)
            output[key] = result
    return output


def plot_results(statistics: list[dict], output_dir: Path) -> list[Path]:
    scenario_order = [item["name"] for item in SCENARIOS]
    labels = ["clean", "noise\n0.1°", "noise\n0.5°", "bias\n0.25°",
              "bias\n1.0°", "delay\n20 ms", "delay\n40 ms", "delay\n80 ms"]
    method_order = list(METHODS)
    colors = {method: f"C{index}" for index, method in enumerate(method_order)}
    panels = (
        ("success", "Completion rate"), ("elapsed_s", "RMST (s)"),
        ("base_tilt_rmse_rad", "Tilt RMSE (rad)"),
        ("contacting_foot_slip_rms_m_s", "Slip RMS (m/s)"),
        ("action_delta_rms", "Action-change RMS"),
        ("roll_prediction_rmse_rad", "100 ms roll forecast RMSE (rad)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    x = np.arange(len(scenario_order))
    lookup = {(row["scenario"], row["method"], row["metric"]): row for row in statistics}
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        for method in method_order:
            means, lows, highs = [], [], []
            for scenario in scenario_order:
                row = lookup.get((scenario, method, metric))
                if row is None and method == "Dynamic-LSTM-47":
                    row = lookup.get(("clean", method, metric))
                means.append(float(row["mean"]) if row else math.nan)
                lows.append(float(row["ci95_low"]) if row else math.nan)
                highs.append(float(row["ci95_high"]) if row else math.nan)
            values = np.asarray(means)
            if np.all(~np.isfinite(values)):
                continue
            low = np.asarray(lows); high = np.asarray(highs)
            axis.plot(x, values, marker="o", linewidth=1.5, color=colors[method], label=method)
            axis.fill_between(x, low, high, color=colors[method], alpha=0.15)
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Frozen-policy single-factor roll-sensor sensitivity\n(light bands: episode-bootstrap 95% CI)")
    fig.tight_layout()
    paths = [output_dir / "sensor_stress_curves.svg", output_dir / "sensor_stress_curves.png"]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    parser.add_argument("--amplitude", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seeds", default=",".join(map(str, protocol["evaluation"]["matched_seeds"])))
    parser.add_argument("--reward-mode", default=protocol["training_reward"]["mode"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = {method: (seed, path) for method, seed, path in args.checkpoint}
    if set(checkpoints) != set(METHODS):
        raise ValueError(f"expected checkpoints for {METHODS}; got {tuple(checkpoints)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    rows: list[dict] = []
    csv_path = args.output_dir / "sensor_stress_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for method in METHODS:
            train_seed, checkpoint = checkpoints[method]
            model = load_model(checkpoint, method)
            checkpoint_hash = sha256(checkpoint)
            scenarios = SCENARIOS[:1] if method == "Dynamic-LSTM-47" else SCENARIOS
            for scenario in scenarios:
                for eval_seed in seeds:
                    row = run_episode(model, method, train_seed, checkpoint, checkpoint_hash,
                                      scenario, args.amplitude, eval_seed, args.duration,
                                      args.reward_mode)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
    statistics = summarize(rows)
    stat_path = args.output_dir / "sensor_stress_statistics.csv"
    with stat_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    paired_path = args.output_dir / "paired_vs_clean.json"
    paired_path.write_text(json.dumps(paired_vs_clean(rows), indent=2), encoding="utf-8")
    plots = plot_results(statistics, args.output_dir)
    source_path = args.output_dir / "sensor_stress_evaluate_source.py"
    shutil.copy2(Path(__file__), source_path)
    env_source = args.output_dir / "fair_controller_ablation_source.py"
    shutil.copy2(HERE / "fair_controller_ablation.py", env_source)
    manifest = {
        "stage": "F4_frozen_policy_sensor_stress", "methods": METHODS,
        "dynamic_reference_evaluated_once": True, "scenarios": SCENARIOS,
        "sensor_definition": "deck-mounted roll-attitude channel at 50 Hz",
        "noise_distribution": "independent zero-mean Gaussian angle noise per control sample",
        "bias_definition": "constant roll-angle offset per episode",
        "latency_definition": "causal sample delay before noise and bias",
        "amplitude_deg": args.amplitude, "condition": "combined",
        "duration_s": args.duration, "eval_seeds": seeds,
        "actor_deck_truth": False,
        "truth_use": "offline prediction-error labels only",
        "training_seed_scope": "one frozen seed-37 checkpoint per method",
        "confidence_band_semantics": "episode-bootstrap 95% CI",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_manifest = {
        "light_band": "episode-bootstrap 95% confidence interval",
        "figure": plots[0].name, "source_csv": stat_path.name,
    }
    plot_manifest_path = args.output_dir / "plot_manifest.json"
    plot_manifest_path.write_text(json.dumps(plot_manifest, indent=2), encoding="utf-8")
    artifacts = [csv_path, stat_path, paired_path, manifest_path, plot_manifest_path,
                 source_path, env_source, *plots]
    hashes = {path.name: file_sha256(path) for path in artifacts}
    hash_path = args.output_dir / "sha256.json"
    hash_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    completed = {"status": "completed", "episode_rows": len(rows),
                 "expected_episode_rows": len(seeds) * (1 + 3 * len(SCENARIOS)),
                 "statistics_rows": len(statistics),
                 "sha256_index_sha256": file_sha256(hash_path)}
    (args.output_dir / "COMPLETED.json").write_text(
        json.dumps(completed, indent=2), encoding="utf-8"
    )
    print(json.dumps(completed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
