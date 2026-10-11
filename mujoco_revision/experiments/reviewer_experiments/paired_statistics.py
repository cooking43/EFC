"""Recompute paired tests and effect sizes from event-level evaluation CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

PAIR_KEYS = ("train_seed", "condition", "amplitude_deg", "eval_seed")
CONTINUOUS = (
    "elapsed_s", "base_tilt_rmse_rad", "contacting_foot_slip_rms_m_s",
    "action_delta_rms",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[name] for name in PAIR_KEYS)


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / 2 ** discordant
        p_value = min(1.0, 2.0 * tail)
    return {"a_only_success": a_only, "b_only_success": b_only,
            "discordant_pairs": discordant, "two_sided_exact_p": p_value,
            "paired_risk_difference_b_minus_a": float(np.mean(b - a))}


def paired_continuous(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                      replicates: int = 20000) -> dict:
    difference = b - a
    n = difference.size
    bootstrap = rng.choice(difference, size=(replicates, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(replicates, n), replace=True)
    null_means = (signs * difference).mean(axis=1)
    observed = float(difference.mean())
    p_value = float((1 + np.sum(np.abs(null_means) >= abs(observed))) / (replicates + 1))
    standard_deviation = float(difference.std(ddof=1)) if n > 1 else 0.0
    effect = observed / standard_deviation if standard_deviation > 0 else 0.0
    return {"n_pairs": int(n), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "paired_mean_difference_b_minus_a": observed, "ci_low": float(ci_low),
            "ci_high": float(ci_high), "paired_standardized_effect_dz": effect,
            "two_sided_sign_flip_p": p_value, "permutation_replicates": replicates}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.csv)
    a_rows = {key(row): row for row in rows if row["method"] == args.method_a}
    b_rows = {key(row): row for row in rows if row["method"] == args.method_b}
    if set(a_rows) != set(b_rows):
        missing_a = sorted(set(b_rows) - set(a_rows))
        missing_b = sorted(set(a_rows) - set(b_rows))
        raise RuntimeError(f"comparison is not paired; missing_a={missing_a[:5]}, missing_b={missing_b[:5]}")
    rng = np.random.default_rng(20260923)
    result: dict[str, dict] = {}
    strata = sorted({(item[1], item[2]) for item in a_rows})
    for condition, amplitude in strata:
        keys = sorted(item for item in a_rows if item[1:] and
                      item[1] == condition and item[2] == amplitude)
        success_a = np.asarray([int(a_rows[item]["success"]) for item in keys])
        success_b = np.asarray([int(b_rows[item]["success"]) for item in keys])
        name = f"{condition}/{amplitude}deg"
        result[name] = {"completion": exact_mcnemar(success_a, success_b)}
        for metric in CONTINUOUS:
            values_a = np.asarray([float(a_rows[item][metric]) for item in keys])
            values_b = np.asarray([float(b_rows[item][metric]) for item in keys])
            result[name][metric] = paired_continuous(values_a, values_b, rng)
    payload = {"method_a": args.method_a, "method_b": args.method_b,
               "pair_keys": PAIR_KEYS, "comparisons": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "strata": len(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
