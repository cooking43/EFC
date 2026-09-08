#!/usr/bin/env python3
"""Verify all experiment configs have correct observation dimensions."""

import json
from pathlib import Path

# Base dimensions from RobotRLAgent.cs
BASE_DIM = 33  # 9 + 2*ActionNum(12)
COM_DIM = 5    # CoM observation adds 5
PRED_DIM = 24  # Prediction observation adds 24

configs_dir = Path(__file__).parent / "experiments"

results = []
errors = []

for json_file in sorted(configs_dir.glob("*.json")):
    with open(json_file) as f:
        cfg = json.load(f)

    name = cfg["meta"]["experimentName"]
    notes = cfg["meta"]["notes"]

    com = cfg["observation"]["enableCoMObservation"]
    pred = cfg["observation"]["enablePredictionObservation"]
    ff = cfg["observation"]["enableFeedforwardControl"]

    # Calculate dimension
    dim = BASE_DIM
    if com:
        dim += COM_DIM
    if pred:
        dim += PRED_DIM

    # Extract claimed dimension from notes if present
    import re
    claimed_dim = None
    match = re.search(r'(\d+)-dim', notes)
    if match:
        claimed_dim = int(match.group(1))

    results.append({
        "name": name,
        "CoM": com,
        "Pred": pred,
        "FF": ff,
        "actual_dim": dim,
        "claimed_dim": claimed_dim,
        "notes": notes
    })

    # Validate
    if claimed_dim is not None and claimed_dim != dim:
        errors.append(f"❌ {name}: claims {claimed_dim}-dim but calculates to {dim}-dim")

# Print table
print(f"{'Config':<30} {'CoM':<6} {'Pred':<6} {'FF':<6} {'Actual':<8} {'Claimed':<8} {'Match'}")
print("=" * 90)

for r in results:
    match = "✓" if r["claimed_dim"] is None or r["claimed_dim"] == r["actual_dim"] else "✗"
    claimed_str = str(r["claimed_dim"]) if r["claimed_dim"] is not None else "-"
    print(f"{r['name']:<30} {str(r['CoM']):<6} {str(r['Pred']):<6} {str(r['FF']):<6} {r['actual_dim']:<8} {claimed_str:<8} {match}")

print()

# Expected configurations
expected = {
    "static_baseline": (False, False, False, 33),
    "dynamic_baseline": (False, False, False, 33),
    "dynamic_priors": (True, False, False, 38),
    "dynamic_implicit": (False, True, False, 57),
    "dynamic_lstm": (True, False, False, 38),  # +memory
    "proposed": (True, False, True, 38),
    # Ablations should match proposed
    "predictor_fdema": (True, False, True, 38),
    "predictor_first_order": (True, False, True, 38),
    "predictor_oracle": (True, False, True, 38),
    "horizon_50ms": (True, False, True, 38),
    "horizon_200ms": (True, False, True, 38),
    "channel_knee_ankle": (True, False, True, 38),
    "channel_knee_ankle_hip": (True, False, True, 38),
    "test_pure_roll": (True, False, True, 38),
}

print("Validation against expected configurations:")
print("=" * 90)

for r in results:
    name = r["name"]
    if name not in expected:
        errors.append(f"❌ {name}: not in expected config list")
        continue

    exp_com, exp_pred, exp_ff, exp_dim = expected[name]

    if (r["CoM"] != exp_com or r["Pred"] != exp_pred or
        r["FF"] != exp_ff or r["actual_dim"] != exp_dim):
        errors.append(
            f"❌ {name}: expected (CoM={exp_com}, Pred={exp_pred}, FF={exp_ff}, {exp_dim}-dim), "
            f"got (CoM={r['CoM']}, Pred={r['Pred']}, FF={r['FF']}, {r['actual_dim']}-dim)"
        )
        print(f"✗ {name}")
    else:
        print(f"✓ {name}")

print()
print(f"Total configs checked: {len(results)}")
print(f"Errors found: {len(errors)}")

if errors:
    print("\nERRORS:")
    for err in errors:
        print(err)
    exit(1)
else:
    print("\n✓ All configurations valid!")
    exit(0)
