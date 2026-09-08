# EFC experiment configurations

Each experiment is represented by an environment JSON file in `experiments/` and a matching ML-Agents trainer YAML file in `trainers/`. `manifest.json` is the machine-readable index.

## Main comparisons

| Name | Observation/control variant |
| --- | --- |
| `static_baseline` | Static platform and base proprioception |
| `dynamic_baseline` | Dynamic platform with pure learned feedback |
| `dynamic_priors` | Dynamic platform plus CoM/contact physical priors |
| `dynamic_implicit` | Prediction supplied to the policy as observations |
| `dynamic_lstm` | Recurrent implicit baseline |
| `proposed` | Physical priors plus explicit feedforward compensation |

## Ablations

- Predictor: `predictor_fdema`, `predictor_first_order`, `predictor_oracle`
- Horizon: `horizon_50ms`, `horizon_200ms`
- Compensation channels: `channel_knee_ankle`, `channel_knee_ankle_hip`

## Selecting a configuration

`ExperimentConfig.cs` reads `Assets/Playground/configs/experiment_config.json` at runtime. Copy the desired file over that path before entering Play mode or building the player. Keep the source experiment file unchanged so the run remains auditable.

Example:

```bash
cp Assets/Playground/configs/experiments/proposed.json \
   Assets/Playground/configs/experiment_config.json
```

Use the matching trainer file:

```bash
mlagents-learn Assets/Playground/configs/trainers/proposed.yaml \
  --run-id=efc_proposed_seed0 --force
```

The configuration system writes a parameter snapshot beneath `Assets/Playground/results/`. That directory is generated output and must not be committed.
