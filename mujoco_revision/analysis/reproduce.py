"""Portable entry point for unchanged archived statistics and figure functions."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import shutil

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, key="root"):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), key
        for name, value in expected.items():
            compare(value, actual[name], key + "." + name)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), key
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{key}[{index}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12), (key, expected, actual)
    else:
        assert expected == actual, (key, expected, actual)


def unpack_frames():
    compressed = ROOT / "data/runs/F3_roll_predictor_formal_30seeds/prediction_frames.csv.gz"
    target = compressed.with_suffix("")
    if not target.exists():
        with gzip.open(compressed, "rb") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    expected = next(item["uncompressed_sha256"] for item in manifest["files"] if item["path"].endswith("prediction_frames.csv.gz"))
    with target.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, "Decompressed prediction CSV hash mismatch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics-only", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("efc_archived_analysis", ROOT / "analysis/statistics_and_figures_archived.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    analysis.OUT = ROOT / "analysis/recomputed"
    analysis.OUT.mkdir(exist_ok=True)
    analysis.RUNS = ROOT / "data/runs"
    analysis.analyze_current_data()
    reference = json.loads((ROOT / "analysis/paper_results/chapter_statistics_20261010.json").read_text(encoding="utf-8"))
    compare(reference, analysis.ANALYSIS)
    print("All archived numerical summaries reproduce from the released CSVs (tolerance 1e-12).", flush=True)
    if not args.statistics_only:
        unpack_frames()
        analysis.training()
        analysis.predictor()
        analysis.core()
        analysis.id_ood()
        analysis.stress()
        analysis.seed_detail()
        analysis.paired_detail()
        print("Seven quantitative PNG/SVG figure pairs generated in analysis/recomputed/.", flush=True)
    (analysis.OUT / "verification.json").write_text(json.dumps({
        "archived_numeric_summaries_match": True,
        "source_csv_values_changed": False,
        "valid_prediction_horizons_ms": [100, 200],
        "figures_regenerated": not args.statistics_only,
        "relative_tolerance": 1e-12,
        "absolute_tolerance": 1e-12,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
