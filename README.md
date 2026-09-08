# EFC: Robust Humanoid Control on Dynamic Maritime Platforms

[中文说明](README_zh-CN.md)

This repository is the lightweight code release for **“EFC: Robust Humanoid Control on Dynamic Maritime Platforms via Explicit Feedforward.”** It contains the paper-specific control, environment, configuration, evaluation, and test code. Large assets that can be obtained elsewhere are deliberately excluded.

## Requirements

- Unity `2022.3.62f3`
- Python `3.10.12`
- Unity ML-Agents / Python `mlagents==1.1.0`
- PyTorch `2.2.x` (the original experiments used `2.2.1`)

The base Gewu repository contains the Unity package layout. Python analysis dependencies are listed in `requirements.txt`.

## Installation

Clone this repository and the pinned Gewu base side by side:

```bash
git clone https://github.com/cooking43/EFC.git
git clone https://github.com/loongOpen/Unity-RL-Playground.git
cd Unity-RL-Playground
git checkout 70e063aa240d7f534b9ab8547e72866cb7765a0b
cd ../EFC
python tools/install_overlay.py ../Unity-RL-Playground
```

Open `Unity-RL-Playground/gewu` in Unity Hub. After compilation completes, choose **Tools → Create ShipDeck Scene**. The generated scene is saved as `Assets/Playground/ShipDeck.unity` and contains 14 independent training areas by default.

The installer overwrites only files that are intentionally replaced by EFC and prints every affected path. Run it on a clean checkout so that `git diff` provides an exact audit trail.

## Reproducing an experiment

Each experiment has two files:

- `configs/experiments/<name>.json`: environment, observation, predictor, and feedforward settings.
- `configs/trainers/<name>.yaml`: PPO/network settings.

Before entering Play mode, select one experiment by copying its JSON file to the runtime configuration path. For the proposed method:

```bash
cp Unity-RL-Playground/gewu/Assets/Playground/configs/experiments/proposed.json \
   Unity-RL-Playground/gewu/Assets/Playground/configs/experiment_config.json
```

On PowerShell:

```powershell
Copy-Item `
  .\Unity-RL-Playground\gewu\Assets\Playground\configs\experiments\proposed.json `
  .\Unity-RL-Playground\gewu\Assets\Playground\configs\experiment_config.json -Force
```

Start the trainer from the Gewu project directory:

```bash
mlagents-learn gewu/Assets/Playground/configs/trainers/proposed.yaml \
  --run-id=efc_proposed_seed0 --force
```

When ML-Agents reports that it is listening, open `ShipDeck.unity` and press Play. Use distinct run IDs and random seeds for repeated trials. The main comparison set is:

- `static_baseline`
- `dynamic_baseline`
- `dynamic_priors`
- `dynamic_implicit`
- `dynamic_lstm`
- `proposed`

Predictor, horizon, and joint-channel ablations are listed in `unity/Assets/Playground/configs/manifest.json`.

## Repository layout

```text
EFC/
├── unity/Assets/Playground/   # EFC controller, environment, configs, and editor tools
├── unity/Assets/Tests/        # Unity EditMode and PlayMode tests
├── scripts/                   # sweep, extraction, and analysis utilities
├── tools/install_overlay.py   # installs this lightweight overlay into Gewu
├── docs/ASSETS.md             # external/downloadable asset map
├── CITATION.cff               # GitHub citation metadata
└── LICENSE                    # Apache-2.0
```

## Citation

If you use this code, please cite both the EFC paper and the Gewu platform on which the environment is based.

```bibtex
@article{xie2026efc,
  title   = {EFC: Robust Humanoid Control on Dynamic Maritime Platforms via Explicit Feedforward},
  author  = {Xie, Daoshun and Yang, Shenhua and Wang, Weijun and Chen, Guoquan},
  year    = {2026},
  note    = {Manuscript}
}

@article{ye2026gewu,
  title   = {Gewu Playground: An Open-Source Robot Simulation Platform for Embodied Intelligence Research},
  author  = {Ye, Linqi and Xing, Boyang and Liang, Bin and Jiang, Lei and Peng, Yan},
  journal = {Science China Technological Sciences},
  year    = {2026},
  doi     = {10.1007/s11431-025-3253-2}
}

@article{ye2025unityrlplayground,
  title         = {Unity RL Playground: A Versatile Reinforcement Learning Framework for Mobile Robots},
  author        = {Ye, Linqi and Li, Rankun and Hu, Xiaowen and Li, Jiayi and Xing, Boyang and Peng, Yan and Liang, Bin},
  journal       = {arXiv preprint arXiv:2503.05146},
  year          = {2025},
  eprint        = {2503.05146},
  archivePrefix = {arXiv}
}
```

Replace the provisional EFC manuscript entry with its DOI, venue, volume, pages, or arXiv identifier once those are assigned. GitHub also exposes the same provisional EFC metadata through `CITATION.cff`.

## License and attribution

The release is provided under the Apache License 2.0. It contains a modified Gewu agent file and is intended to be used with the upstream Gewu project. See `NOTICE` and [docs/ASSETS.md](docs/ASSETS.md) for attribution and third-party boundaries. External assets remain subject to their own licenses.
