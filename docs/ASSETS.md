# External and generated assets

This code release is an overlay, not a complete Unity project. This keeps the repository small, makes the paper-specific changes reviewable, and avoids redistributing assets available from their original source.

## Required base project

- Source: [loongOpen/Unity-RL-Playground](https://github.com/loongOpen/Unity-RL-Playground)
- Development commit: `70e063aa240d7f534b9ab8547e72866cb7765a0b`
- Required content from the base project: Unity project settings and packages, the `Playground.unity` base scene, the G1 robot hierarchy, URDF-derived objects, meshes, materials, textures, and ML-Agents integration.

Use `tools/install_overlay.py` to copy this repository's `unity/` tree into the downloaded base. Do not copy the base project back into this repository.

## Generated locally

The following are generated and must stay untracked:

- `ShipDeck.unity`, created by **Tools → Create ShipDeck Scene**;
- Unity `Library`, `Temp`, `Logs`, `Obj`, and `UserSettings` directories;
- Linux/Windows/macOS player builds;
- ONNX models, checkpoints, TensorBoard events, and trainer summaries;
- result CSV files, parameter snapshots, plots, videos, and crash dumps.

## Model and data archive

No stable public URL for pretrained EFC policies or raw experiment data was present in the source workspace when this release tree was prepared. The repository therefore documents deterministic configuration files and the regeneration path, but does not claim to provide pretrained inference.

If a public archive is created, add its persistent DOI/URL, checksum, archive version, and the exact mapping from archive files to experiment IDs here. Prefer a research-data archive such as Zenodo over committing large binaries to GitHub.

## Licensing boundary

The Apache-2.0 license in this repository covers the distributed source files. It does not automatically relicense downloaded third-party assets. Users must review the license and usage restrictions attached to Gewu, Unity packages, robot descriptions, datasets, and any separately downloaded model weights.
