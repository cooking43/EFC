# MuJoCo H1-2 roll-pitch standing revision

This directory contains the implementation, frozen policies, configuration, episode records, prediction records, and statistical analysis used for the revised EFC standing study. The repository's original Unity release is retained outside this directory as a separate historical implementation.

## Experiment and model map

| Paper name | Implementation identifier | Observation | Explicit knee offset | Independent training runs |
|---|---|---|---|---:|
| Basic LSTM | `Dynamic-LSTM-47` | 47-dimensional proprioception | None | 1 |
| Enhanced LSTM | `State-Enhanced-LSTM` | 65-dimensional causal enhanced state | None | 1 |
| Prediction Input Only | `Prediction-to-Policy` | Same 48-dimensional predictive observation as the two core residual methods | None | 3 |
| Current-Roll Knee Residual | `Current-State-Residual` | Same predictive observation | Filtered current roll | 3 |
| EFC | `Predictive-Residual` | Same predictive observation | Filtered predicted roll | 3 |
| Oracle Knee Residual | `Oracle-Predictive-Residual` | Same causal predictive observation | True future roll | 1 |

All policies use a recurrent PPO backbone and a 12-joint action interface. The core training seeds are 37, 73, and 109, with a common initialization and 1,048,576 control steps per trained model. Roll and pitch training amplitudes are sampled independently from 0 to 5 degrees. Control evaluation uses matched disturbance seeds 5001 through 5030 and a 60-second limit. These are 3 independent training runs, not 90 independent training runs per condition.

The deck is 50 x 50 x 0.2 m, with roll and pitch excitation only. Physics runs at 0.002 s and control at 50 Hz. EFC uses a 100 ms second-order roll forecast, a new-sample filtering weight of 0.3, a differential knee mapping gain of 0.1, and an explicit offset limit of 8 degrees per knee. Oracle is an ideal-information reference, not a deployable controller or a guarantee of highest closed-loop performance.

`model_index.json` maps each archived checkpoint path in the CSV files to its portable repository path and SHA256. The 12 frozen models and their individual training logs/configurations are under `experiments/reviewer_experiments/runs/`. The common initialization and necessary Unitree model assets are included.

## Data and paper figure correspondence

All paths in this table are relative to `mujoco_revision/`.

| Result | Input records | Number of rows | Paper output |
|---|---|---:|---|
| Training rewards and components | `data/runs/P4_combined_seed37_comparison/combined_training_updates.csv` | 6,144 | Figure 5 |
| Core three-method, three-training comparison at 0/1/3/5 degrees | `data/runs/P6_three_method_three_train_seed_ID_60s_30seeds/evaluation_episodes.csv` | 1,080 | Figures 7-9; Tables 5-7 |
| Six-method single-training ID comparison | `data/runs/F1_seed37_six_method_ID_60s_30seeds/evaluation_episodes.csv` | 720 | Table 8; ID part of Figure 10 |
| Six-method single-training OOD comparison at 7/10 degrees | `data/runs/F2_seed37_six_method_OOD_7_10deg_60s_30seeds/evaluation_episodes.csv` | 360 | Figure 10; Table 9 |
| Prediction errors | `data/runs/F3_roll_predictor_formal_30seeds/prediction_episode_metrics.csv` | 1,620 archived, including excluded 50 ms records | Figure 6; Table 4 use 100/200 ms only |
| Prediction trajectories | `data/runs/F3_roll_predictor_formal_30seeds/prediction_frames.csv.gz` | Per-frame records | Figure 6 trajectory panel |
| Roll observation sensitivity | `data/runs/F4_seed37_sensor_stress_5deg_60s_30seeds/sensor_stress_episodes.csv` | 750 archived | Figure 11; Table 10 use the three core methods |

The prediction trajectory CSV is losslessly gzip-compressed because its uncompressed size exceeds GitHub's single-file limit. `verify_release.py` checks both the compressed file and the decompressed byte hash. No source CSV values are changed by packaging. The qualitative standing montage is supplied in `analysis/paper_results/fig4_1_efc_dynamic_sequence.png`; it is not an additional quantitative trial.

Archived run manifests are retained as configuration records. Some of their historical paths or checksums refer to earlier archives. Use `release_manifest.json` for the hashes of the files released here and `model_index.json` for portable checkpoint locations. Current paper statistics are in `analysis/paper_results/`; old aggregate statistical files are not used in the analysis.

### Excluded nominal 50 ms prediction comparison

The archived predictor evaluator samples at 20 ms and uses an integer future index. For a nominal horizon of 50 ms, rounding selects a 40 ms target while the predictor extrapolates 50 ms. This is not a valid 50 ms error comparison. Those historical records and the evaluator source are retained unchanged for traceability; all current paper prediction analyses exclude them. Valid reported horizons are 100 and 200 ms, corresponding to 5 and 10 control samples. Use `--horizons-ms 100 200` when running the archived evaluator. No corrected 50 ms experiment is claimed in this release.

## Verification and statistical reproduction

Use Python 3.13 and install the pinned environment in `requirements.txt`. The recorded versions are also retained in run manifests. From this directory:

```bash
python verify_release.py
python -B -m pytest -q -p no:cacheprovider experiments/reviewer_experiments/test_corrected_roll_pitch_platform.py experiments/reviewer_experiments/test_fair_controller_ablation.py experiments/reviewer_experiments/test_state_enhanced_lstm.py experiments/reviewer_experiments/test_roll_predictor_validation.py
python analysis/reproduce.py --statistics-only
python analysis/reproduce.py
```

The analysis reproduces the archived numerical summaries and compares them against `analysis/paper_results/chapter_statistics_20261010.json`. Generated statistics and plots go to `analysis/recomputed/`, without overwriting archived paper outputs. When figures are requested, the large trajectory CSV is decompressed locally into an ignored cache file; its hash must match the original before use. Figure metadata and rendering may differ between platforms, whereas numerical verification uses a tight floating-point tolerance.

The analysis uses 20,000 bootstrap replicates. Core intervals resample training clusters followed by matched disturbance trials; paired differences preserve controller matching. Exact sign-flip tests operate on training-run means. With 3 independent runs, the minimum attainable two-sided exact sign-flip p-value is 0.25. Trial-level McNemar results are descriptive complements, not substitutes for independent training replication. `d_z` standardizes paired differences and is not a percentage or a fold improvement.

Higher completion rate and 60-second RMST are better. Lower attitude RMSE, contact slip RMS, and feedback-action-change RMS are better under the defined metrics. Continuous metrics use each trial's actual survival interval. Action-change RMS measures policy outputs, not total joint energy or the entire feedforward-plus-feedback target.

## Running new evaluations

`experiments/reviewer_experiments/formal_evaluate.py` loads checkpoints supplied by repeated `--checkpoint METHOD:TRAIN_SEED:PATH` arguments. For example, one full EFC seed-37 ID evaluation is:

```bash
python experiments/reviewer_experiments/formal_evaluate.py --checkpoint Predictive-Residual:37:experiments/reviewer_experiments/runs/P4_predictive_residual_seed37/checkpoint_final.pt --conditions combined --amplitudes 0,1,3,5 --duration 60 --reward-mode corrected --output-dir new_runs/efc_seed37_id
```

The default protocol supplies the 30 matched evaluation seeds. Use `--amplitudes 7,10` with the same frozen policies for OOD evaluations. Use a new output directory for each run. Training programs are included for code inspection and reproduction, but no training is started by the verification or analysis commands.

## Licensing

EFC-specific code is distributed under the repository's Apache-2.0 license. Unitree assets and deployment files retain their BSD-3-Clause notice at `experiments/h1_deck_balance/external/unitree_rl_gym/LICENSE`. Redistribution does not imply Unitree's endorsement. Personal environments, credentials, private manuscript files, and reviewer correspondence are not part of this public package.
