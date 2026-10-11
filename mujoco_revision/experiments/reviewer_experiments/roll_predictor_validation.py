"""Independent roll-prediction validation for the corrected EFC method.

The predicted variable is platform roll for every comparator.  The prescribed
MuJoCo mocap deck follows this target exactly; no robot or controller outcome
is used to tune or score the predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments" / "h1_deck_balance"
sys.path[:0] = [str(ROOT), str(BASE), str(HERE)]

MATPLOTLIB_CACHE = HERE / ".matplotlib-cache"
MATPLOTLIB_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from deck_motion import get_deck_pose


CONTROL_DT_S = 0.02
FILTER_NEW_WEIGHT = 0.3
HISTORY_WINDOW = 10
METHODS = (
    "Persistence", "First-Order", "Second-Order",
    "Filtered-Second-Order", "Causal-History", "Oracle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_difference_predictions(history: np.ndarray, horizon_s: float,
                                  dt_s: float = CONTROL_DT_S) -> dict[str, float]:
    if history.size < 3:
        raise ValueError("second-order causal prediction requires three samples")
    phi_tm2, phi_tm1, phi_t = map(float, history[-3:])
    velocity = (3.0 * phi_t - 4.0 * phi_tm1 + phi_tm2) / (2.0 * dt_s)
    acceleration = (phi_t - 2.0 * phi_tm1 + phi_tm2) / dt_s**2
    first = phi_t + velocity * horizon_s
    second = first + 0.5 * acceleration * horizon_s**2
    return {
        "Persistence": phi_t,
        "First-Order": first,
        "Second-Order": second,
    }


def causal_history_prediction(history: np.ndarray, horizon_s: float,
                              dt_s: float = CONTROL_DT_S,
                              window: int = HISTORY_WINDOW) -> float:
    values = np.asarray(history[-window:], dtype=np.float64)
    if values.size < 3:
        return float(values[-1])
    times = np.arange(-(values.size - 1), 1, dtype=np.float64) * dt_s
    coefficients = np.polyfit(times, values, deg=2)
    return float(np.polyval(coefficients, horizon_s))


def generate_roll(seed: int, amplitude_deg: float, duration_s: float,
                  max_horizon_s: float) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = yaml.safe_load((BASE / "configs" / "deck_stand.yaml").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    cfg["deck"]["roll_amp_deg"] = float(amplitude_deg)
    cfg["deck"]["pitch_amp_deg"] = 0.0
    cfg["deck"]["heave_amp_m"] = 0.0
    cfg["deck"]["roll_phase_rad"] = float(rng.uniform(0.0, 2.0 * math.pi))
    cfg["deck"]["pitch_phase_rad"] = 0.0
    count = int(round((duration_s + max_horizon_s) / CONTROL_DT_S)) + 1
    times = np.arange(count, dtype=np.float64) * CONTROL_DT_S
    roll = np.asarray([get_deck_pose(float(t), cfg)["roll"] for t in times])
    return times, roll, cfg


def evaluate_episode(seed: int, amplitude_deg: float, duration_s: float,
                     horizons_ms: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    max_horizon_s = max(horizons_ms) / 1000.0
    times, roll, cfg = generate_roll(seed, amplitude_deg, duration_s, max_horizon_s)
    frames: list[dict] = []
    metrics: list[dict] = []
    for horizon_ms in horizons_ms:
        horizon_s = horizon_ms / 1000.0
        offset = int(round(horizon_s / CONTROL_DT_S))
        filtered_second = float(roll[0])
        per_method_errors: dict[str, list[float]] = {method: [] for method in METHODS}
        last_valid_index = int(round(duration_s / CONTROL_DT_S))
        for index in range(2, last_valid_index + 1):
            truth = float(roll[index + offset])
            causal = finite_difference_predictions(roll[:index + 1], horizon_s)
            filtered_second = (FILTER_NEW_WEIGHT * causal["Second-Order"]
                               + (1.0 - FILTER_NEW_WEIGHT) * filtered_second)
            predictions = {
                **causal,
                "Filtered-Second-Order": filtered_second,
                "Causal-History": causal_history_prediction(roll[:index + 1], horizon_s),
                "Oracle": truth,
            }
            for method, prediction in predictions.items():
                error = float(prediction - truth)
                per_method_errors[method].append(error)
                frames.append({
                    "seed": seed,
                    "amplitude_deg": amplitude_deg,
                    "roll_frequency_hz": float(cfg["deck"]["roll_freq_hz"]),
                    "roll_phase_rad": float(cfg["deck"]["roll_phase_rad"]),
                    "time_s": float(times[index]),
                    "horizon_ms": horizon_ms,
                    "method": method,
                    "current_roll_rad": float(roll[index]),
                    "target_time_s": float(times[index + offset]),
                    "truth_roll_rad": truth,
                    "prediction_roll_rad": float(prediction),
                    "error_rad": error,
                    "absolute_error_rad": abs(error),
                    "squared_error_rad2": error * error,
                })
        for method, errors in per_method_errors.items():
            array = np.asarray(errors, dtype=np.float64)
            metrics.append({
                "seed": seed,
                "amplitude_deg": amplitude_deg,
                "horizon_ms": horizon_ms,
                "method": method,
                "frame_count": int(array.size),
                "mae_rad": float(np.mean(np.abs(array))),
                "rmse_rad": float(np.sqrt(np.mean(np.square(array)))),
                "bias_rad": float(np.mean(array)),
            })
    return frames, metrics


def bootstrap_ci(values: list[float], rng: np.random.Generator,
                 replicates: int = 10000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    samples = array[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return mean, float(low), float(high)


def summarize(episode_metrics: list[dict]) -> list[dict]:
    rng = np.random.default_rng(20260926)
    rows: list[dict] = []
    amplitudes = sorted({float(row["amplitude_deg"]) for row in episode_metrics})
    horizons = sorted({int(row["horizon_ms"]) for row in episode_metrics})
    for amplitude in amplitudes + [math.nan]:
        for horizon in horizons:
            for method in METHODS:
                selected = [row for row in episode_metrics
                            if int(row["horizon_ms"]) == horizon
                            and row["method"] == method
                            and (math.isnan(amplitude)
                                 or float(row["amplitude_deg"]) == amplitude)]
                for metric in ("mae_rad", "rmse_rad", "bias_rad"):
                    mean, low, high = bootstrap_ci(
                        [float(row[metric]) for row in selected], rng
                    )
                    rows.append({
                        "amplitude_deg": "pooled" if math.isnan(amplitude) else amplitude,
                        "horizon_ms": horizon,
                        "method": method,
                        "metric": metric,
                        "episode_count": len(selected),
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    })
    return rows


def plot_results(frames: list[dict], statistics: list[dict], output_dir: Path) -> list[Path]:
    display_methods = METHODS[:-1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    representative = [row for row in frames
                      if int(row["seed"]) == 303
                      and float(row["amplitude_deg"]) == 5.0
                      and int(row["horizon_ms"]) == 100
                      and 4.0 <= float(row["time_s"]) <= 7.0]
    truth_rows = [row for row in representative if row["method"] == "Persistence"]
    axes[0].plot([float(row["target_time_s"]) for row in truth_rows],
                 [math.degrees(float(row["truth_roll_rad"])) for row in truth_rows],
                 color="black", linewidth=2.0, label="truth")
    for method_index, method in enumerate(display_methods):
        selected = [row for row in representative if row["method"] == method]
        axes[0].plot([float(row["target_time_s"]) for row in selected],
                     [math.degrees(float(row["prediction_roll_rad"])) for row in selected],
                     color=f"C{method_index}", linewidth=1.2, label=method)
    axes[0].set(title="Representative 100 ms forecast", xlabel="Target time (s)",
                ylabel="Platform roll (deg)")

    horizons = sorted({int(row["horizon_ms"]) for row in statistics})
    for method_index, method in enumerate(display_methods):
        selected = [row for row in statistics
                    if row["amplitude_deg"] == "pooled"
                    and row["metric"] == "rmse_rad" and row["method"] == method]
        selected.sort(key=lambda row: int(row["horizon_ms"]))
        means = np.asarray([float(row["mean"]) for row in selected])
        lows = np.asarray([float(row["ci95_low"]) for row in selected])
        highs = np.asarray([float(row["ci95_high"]) for row in selected])
        color = f"C{method_index}"
        axes[1].plot(horizons, np.degrees(means), marker="o", color=color, label=method)
        axes[1].fill_between(horizons, np.degrees(lows), np.degrees(highs),
                             color=color, alpha=0.16)
    axes[1].set(title="Episode-macro error", xlabel="Prediction horizon (ms)",
                ylabel="Roll RMSE (deg)")

    amplitudes = [1.0, 3.0, 5.0]
    for method_index, method in enumerate(display_methods):
        selected = [row for row in statistics
                    if row["amplitude_deg"] != "pooled"
                    and int(row["horizon_ms"]) == 100
                    and row["metric"] == "rmse_rad" and row["method"] == method]
        selected.sort(key=lambda row: float(row["amplitude_deg"]))
        means = np.asarray([float(row["mean"]) for row in selected])
        lows = np.asarray([float(row["ci95_low"]) for row in selected])
        highs = np.asarray([float(row["ci95_high"]) for row in selected])
        color = f"C{method_index}"
        axes[2].plot(amplitudes, np.degrees(means), marker="o", color=color, label=method)
        axes[2].fill_between(amplitudes, np.degrees(lows), np.degrees(highs),
                             color=color, alpha=0.16)
    axes[2].set(title="100 ms amplitude sensitivity", xlabel="Roll amplitude (deg)",
                ylabel="Roll RMSE (deg)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=7)
    fig.suptitle("Corrected platform-roll predictor validation (light bands: episode-bootstrap 95% CI)")
    fig.tight_layout()
    outputs = [output_dir / "prediction_curves.svg", output_dir / "prediction_curves.png"]
    for path in outputs:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[303, 505, 1101, 1303, 1709, 1901, 2203, 2609])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    parser.add_argument("--horizons-ms", type=int, nargs="+", default=[50, 100, 200])
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frames: list[dict] = []
    episode_metrics: list[dict] = []
    for amplitude in args.amplitudes:
        for seed in args.seeds:
            episode_frames, metrics = evaluate_episode(
                seed, amplitude, args.duration_s, tuple(args.horizons_ms)
            )
            frames.extend(episode_frames)
            episode_metrics.extend(metrics)
    statistics = summarize(episode_metrics)
    write_csv(output_dir / "prediction_frames.csv", frames)
    write_csv(output_dir / "prediction_episode_metrics.csv", episode_metrics)
    write_csv(output_dir / "prediction_statistics.csv", statistics)
    plot_paths = plot_results(frames, statistics, output_dir)
    source_snapshot = output_dir / "roll_predictor_validation_source.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    manifest = {
        "stage": "P3_corrected_roll_predictor_validation",
        "predicted_variable": "platform roll",
        "control_dt_s": CONTROL_DT_S,
        "horizons_ms": args.horizons_ms,
        "methods": METHODS,
        "filtered_second_order_alpha": FILTER_NEW_WEIGHT,
        "causal_history_window_samples": HISTORY_WINDOW,
        "seeds": args.seeds,
        "amplitudes_deg": args.amplitudes,
        "duration_s": args.duration_s,
        "deck_tracking": "MuJoCo mocap deck; prescribed roll equals actual deck roll",
        "controller_performance_claim_allowed": False,
        "confidence_band_semantics": "episode-macro bootstrap 95% CI",
        "source_sha256": sha256(source_snapshot),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths = [output_dir / name for name in (
        "prediction_frames.csv", "prediction_episode_metrics.csv",
        "prediction_statistics.csv", "manifest.json"
    )] + plot_paths + [source_snapshot]
    hashes = {path.name: sha256(path) for path in paths}
    (output_dir / "sha256s.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (output_dir / "COMPLETED.json").write_text(json.dumps({
        "status": "completed", "frame_rows": len(frames),
        "episode_metric_rows": len(episode_metrics),
        "statistics_rows": len(statistics),
        "sha256s_sha256": sha256(output_dir / "sha256s.json"),
    }, indent=2), encoding="utf-8")
    primary = [row for row in statistics if row["amplitude_deg"] == "pooled"
               and int(row["horizon_ms"]) == 100 and row["metric"] == "rmse_rad"]
    print(json.dumps({row["method"]: row for row in primary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
