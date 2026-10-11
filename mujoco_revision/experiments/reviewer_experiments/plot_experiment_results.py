"""Generate standardized, auditable curves for one experiment directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

MATPLOTLIB_CACHE = Path(__file__).resolve().parent / ".matplotlib-cache"
MATPLOTLIB_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.copy()
    width = min(window, values.size)
    kernel = np.ones(width, dtype=float) / width
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def save_figure(fig, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    for output in outputs:
        fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outputs


def training_curves(path: Path, run_dir: Path) -> list[Path]:
    rows = read_csv(path)
    if not rows:
        return []
    steps = np.asarray([float(row["global_step"]) for row in rows])
    reward_panels = [
        ("mean_rollout_reward", "Mean per-step rollout reward"),
        ("mean_world_upright_reward", "World-upright reward contribution"),
        ("mean_support_relative_height_reward", "Support-relative height contribution"),
        ("mean_slip_penalty", "Bounded contact-slip contribution"),
        ("mean_action_delta_penalty", "Action-change contribution"),
        ("mean_residual_abs_rad", "Mean explicit residual magnitude (rad)"),
    ]
    diagnostic_panels = [
        ("mean_roll_deg", "Mean sampled roll amplitude (deg)"),
        ("mean_pitch_deg", "Mean sampled pitch amplitude (deg)"),
        ("approx_kl", "Approximate KL"),
        ("grad_norm_before_clip", "Gradient norm before clipping"),
    ]
    def build(panels, stem: str, title: str):
        panels = [(key, label) for key, label in panels if key in rows[0]]
        columns = 2
        nrows = int(np.ceil(len(panels) / columns))
        fig, axes = plt.subplots(nrows, columns, figsize=(12, 3.2 * nrows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for axis, (key, label) in zip(axes, panels):
            values = np.asarray([float(row[key]) for row in rows])
            smooth = rolling_mean(values, 35)
            axis.plot(steps, values, color="C0", alpha=0.20, linewidth=0.8,
                      label="per update")
            axis.plot(steps, smooth, color="C0", linewidth=1.8, label="35-update mean")
            low_key, high_key = f"{key}_ci95_low", f"{key}_ci95_high"
            if low_key in rows[0] and high_key in rows[0]:
                low = np.asarray([float(row[low_key]) for row in rows])
                high = np.asarray([float(row[high_key]) for row in rows])
                axis.fill_between(steps, low, high, color="C0", alpha=0.16, label="95% CI")
            axis.set_title(label)
            axis.grid(alpha=0.2)
            axis.legend(loc="best", fontsize=8)
            axis.set_xlabel("Global training step")
        for axis in axes[len(panels):]:
            axis.set_visible(False)
        fig.suptitle(title)
        fig.tight_layout()
        return save_figure(fig, run_dir / stem)
    outputs = build(
        reward_panels, "training_curves",
        "Training reward curves (single training seed: no confidence band)",
    )
    outputs.extend(build(
        diagnostic_panels, "training_diagnostics",
        "Training optimization diagnostics (single training seed: no confidence band)",
    ))
    return outputs


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                      replicates: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    samples = rng.choice(values, size=(replicates, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return mean, float(low), float(high)


def evaluation_curves(path: Path, run_dir: Path) -> list[Path]:
    rows = read_csv(path)
    if not rows or "amplitude_deg" not in rows[0]:
        return []
    metrics = [
        ("success", "Completion rate"),
        ("elapsed_s", "Elapsed / RMST contribution (s)"),
        ("base_tilt_rmse_rad", "Base tilt RMSE (rad)"),
        ("contacting_foot_slip_rms_m_s", "Contact-foot slip RMS (m/s)"),
        ("action_delta_rms", "Action delta RMS"),
    ]
    metrics = [(key, label) for key, label in metrics if key in rows[0]]
    methods = sorted({row.get("method", "controller") for row in rows})
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.9 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(20260926)
    for method_index, method in enumerate(methods):
        method_rows = [row for row in rows if row.get("method", "controller") == method]
        amplitudes = sorted({float(row["amplitude_deg"]) for row in method_rows})
        for axis, (key, label) in zip(axes, metrics):
            means, lows, highs = [], [], []
            for amplitude in amplitudes:
                selected = [row for row in method_rows if float(row["amplitude_deg"]) == amplitude]
                values = np.asarray([float(row[key]) for row in selected])
                mean, low, high = bootstrap_mean_ci(values, rng)
                means.append(mean); lows.append(low); highs.append(high)
            color = f"C{method_index % 10}"
            axis.plot(amplitudes, means, marker="o", color=color, label=method)
            axis.fill_between(amplitudes, lows, highs, color=color, alpha=0.16)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
    for axis in axes:
        axis.legend(loc="best")
    axes[-1].set_xlabel("Roll/pitch amplitude (deg)")
    fig.suptitle("Matched-seed evaluation (light bands: episode-bootstrap 95% CI)")
    return save_figure(fig, run_dir / "evaluation_curves")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    outputs: list[Path] = []
    inputs: list[Path] = []
    train_path = run_dir / "train_updates.csv"
    eval_path = run_dir / "evaluation_episodes.csv"
    if train_path.is_file():
        inputs.append(train_path); outputs.extend(training_curves(train_path, run_dir))
    if eval_path.is_file():
        inputs.append(eval_path); outputs.extend(evaluation_curves(eval_path, run_dir))
    manifest = {
        "inputs": {path.name: sha256(path) for path in inputs},
        "outputs": {path.name: sha256(path) for path in outputs},
        "confidence_band_semantics": (
            "Light bands are 95% confidence intervals only. Training curves from one "
            "training seed have no confidence band. Evaluation bands use episode bootstrap."
        ),
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260926,
    }
    (run_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
