"""Build auditable cross-method figures from completed P4 seed-37 runs.

Individual run plots remain untouched.  This script rebuilds a canonical
comparison bundle whenever another method finishes its common exploratory
evaluation.  Evaluation bands are episode-bootstrap 95% confidence intervals;
single-seed training curves deliberately have no confidence bands.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATPLOTLIB_CACHE = HERE / ".matplotlib-cache"
MATPLOTLIB_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = (
    "Dynamic-LSTM-47",
    "State-Enhanced-LSTM",
    "Prediction-to-Policy",
    "Current-State-Residual",
    "Predictive-Residual",
    "Oracle-Predictive-Residual",
)
METHOD_COLORS = {method: f"C{index}" for index, method in enumerate(METHOD_ORDER)}
TRAIN_RUNS = {
    "Dynamic-LSTM-47": "P4_dynamic_lstm47_seed37_canonical",
    "State-Enhanced-LSTM": "P4_state_enhanced_lstm_seed37",
    "Prediction-to-Policy": "P4_prediction_to_policy_seed37",
    "Current-State-Residual": "P4_current_state_residual_seed37",
    "Predictive-Residual": "P4_predictive_residual_seed37",
    "Oracle-Predictive-Residual": "P4_oracle_predictive_residual_seed37",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_union_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rolling_mean(values: np.ndarray, window: int = 35) -> np.ndarray:
    width = min(window, values.size)
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=float) / width
    left = width // 2
    right = width - 1 - left
    return np.convolve(np.pad(values, (left, right), mode="edge"), kernel, mode="valid")


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                      replicates: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    draws = rng.choice(values, size=(replicates, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return mean, float(low), float(high)


def save_figure(fig, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    for output in outputs:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outputs


def collect_evaluations(runs_root: Path) -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    sources: list[Path] = []
    for run_dir in sorted(runs_root.glob("P4_*_seed37_exploratory_eval")):
        source = run_dir / "evaluation_episodes.csv"
        if not source.is_file() or not (run_dir / "COMPLETED.json").is_file():
            continue
        run_rows = read_csv(source)
        if not run_rows:
            continue
        methods = {row["method"] for row in run_rows}
        if len(methods) != 1 or next(iter(methods)) not in METHOD_ORDER:
            raise ValueError(f"unexpected method labels in {source}: {methods}")
        for row in run_rows:
            rows.append({"source_run_dir": run_dir.name, **row})
        sources.append(source)
    return rows, sources


def collect_training(runs_root: Path, completed_methods: set[str]) -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    sources: list[Path] = []
    for method in METHOD_ORDER:
        if method not in completed_methods:
            continue
        run_dir = runs_root / TRAIN_RUNS[method]
        source = run_dir / "train_updates.csv"
        if not source.is_file() or not (run_dir / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed evaluation has no completed training record: {source}")
        for row in read_csv(source):
            rows.append({"method": method, "source_run_dir": run_dir.name, **row})
        sources.append(source)
    return rows, sources


def plot_evaluation(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    panels = (
        ("success", "Completion rate"),
        ("elapsed_s", "RMST contribution (s)"),
        ("base_tilt_rmse_rad", "Base tilt RMSE (rad)"),
        ("contacting_foot_slip_rms_m_s", "Contact-foot slip RMS (m/s)"),
        ("action_delta_rms", "Action delta RMS"),
    )
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    axes = axes.ravel()
    rng = np.random.default_rng(20260926)
    for method in methods:
        selected_method = [row for row in rows if row["method"] == method]
        amplitudes = sorted({float(row["amplitude_deg"]) for row in selected_method})
        color = METHOD_COLORS[method]
        for axis, (key, label) in zip(axes, panels):
            means, lows, highs = [], [], []
            for amplitude in amplitudes:
                values = np.asarray([
                    float(row[key]) for row in selected_method
                    if float(row["amplitude_deg"]) == amplitude
                ])
                mean, low, high = bootstrap_mean_ci(values, rng)
                means.append(mean); lows.append(low); highs.append(high)
            axis.plot(amplitudes, means, marker="o", linewidth=1.8,
                      color=color, label=method)
            axis.fill_between(amplitudes, lows, highs, color=color, alpha=0.14)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
    axes[5].axis("off")
    for axis in axes[:5]:
        axis.set_xlabel("Combined roll/pitch amplitude (deg)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 0.05))
    fig.suptitle("Corrected seed-37 exploratory comparison\n(light bands: matched-episode bootstrap 95% CI)")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    return save_figure(fig, output_dir / "combined_evaluation_comparison")


def plot_training(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    panels = (
        ("mean_rollout_reward", "Mean per-step reward"),
        ("mean_world_upright_reward", "World-upright contribution"),
        ("mean_support_relative_height_reward", "Support-height contribution"),
        ("mean_joint_deviation_penalty", "Joint-deviation contribution"),
        ("mean_action_delta_penalty", "Action-change contribution"),
        ("mean_angular_velocity_penalty", "Angular-velocity contribution"),
        ("mean_slip_penalty", "Contact-slip contribution"),
        ("mean_fall_penalty", "Fall contribution"),
    )
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    axes = axes.ravel()
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        steps = np.asarray([float(row["global_step"]) for row in selected])
        color = METHOD_COLORS[method]
        for axis, (key, label) in zip(axes, panels):
            values = np.asarray([float(row[key]) for row in selected])
            axis.plot(steps, values, color=color, alpha=0.08, linewidth=0.6)
            axis.plot(steps, rolling_mean(values), color=color, linewidth=1.6, label=method)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            axis.set_xlabel("Global training step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(methods)))
    fig.suptitle("Corrected seed-37 training comparison\n(single training seed per method: no confidence band)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return save_figure(fig, output_dir / "combined_training_reward_comparison")


def main() -> int:
    runs_root = HERE / "runs"
    output_dir = runs_root / "P4_combined_seed37_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_rows, evaluation_sources = collect_evaluations(runs_root)
    if not evaluation_rows:
        raise RuntimeError("no completed corrected P4 exploratory evaluations found")
    completed_methods = {str(row["method"]) for row in evaluation_rows}
    training_rows, training_sources = collect_training(runs_root, completed_methods)
    evaluation_csv = output_dir / "combined_evaluation_episodes.csv"
    training_csv = output_dir / "combined_training_updates.csv"
    write_union_csv(evaluation_csv, evaluation_rows)
    write_union_csv(training_csv, training_rows)
    outputs = [evaluation_csv, training_csv]
    outputs.extend(plot_evaluation(evaluation_rows, output_dir))
    outputs.extend(plot_training(training_rows, output_dir))
    manifest = {
        "methods": [method for method in METHOD_ORDER if method in completed_methods],
        "evaluation_sources": {str(path.relative_to(runs_root)): sha256(path)
                               for path in evaluation_sources},
        "training_sources": {str(path.relative_to(runs_root)): sha256(path)
                             for path in training_sources},
        "outputs": {path.name: sha256(path) for path in outputs},
        "method_colors": METHOD_COLORS,
        "confidence_band_semantics": (
            "Evaluation light bands are matched-development-episode bootstrap 95% confidence "
            "intervals (10000 replicates, seed 20260926). Single-seed training curves have no CI."
        ),
        "evidence_level": "exploratory development comparison; not formal superiority evidence",
    }
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    integrity = {path.name: sha256(path) for path in outputs + [manifest_path]}
    (output_dir / "sha256.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
