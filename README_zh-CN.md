# EFC：面向动态海上平台的显式前馈人形机器人鲁棒控制

[English](README.md)

本仓库是论文 **“EFC: Robust Humanoid Control on Dynamic Maritime Platforms via Explicit Feedforward”** 的精简代码发布版，仅保留论文方法相关的控制器、环境、实验配置、评估与测试代码。可从公开来源下载或可由程序生成的大体积资产不进入 Git。

## 环境

- Unity `2022.3.62f3`
- Python `3.10.12`
- `mlagents==1.1.0`
- PyTorch `2.2.x`（原实验使用 `2.2.1`）

## 安装

将 EFC 与 Gewu 克隆到同一级目录，并固定到本项目使用的 Gewu 提交：

```bash
git clone https://github.com/cooking43/EFC.git
git clone https://github.com/loongOpen/Unity-RL-Playground.git
cd Unity-RL-Playground
git checkout 70e063aa240d7f534b9ab8547e72866cb7765a0b
cd ../EFC
python tools/install_overlay.py ../Unity-RL-Playground
```

在 Unity Hub 中打开 `Unity-RL-Playground/gewu`。编译完成后，点击 **Tools → Create ShipDeck Scene**，生成的场景位于 `Assets/Playground/ShipDeck.unity`，默认包含 14 个相互独立的训练区域。

安装脚本只覆盖 EFC 明确替换的文件，并逐项输出目标路径。建议始终在干净的 Gewu checkout 上运行，以便通过 `git diff` 审核全部改动。

## 复现实验

每组实验由两个文件组成：

- `configs/experiments/<实验名>.json`：环境、观测、预测器和前馈参数；
- `configs/trainers/<实验名>.yaml`：PPO 与网络参数。

进入 Play 模式前，将要运行的 JSON 复制为运行时配置。以论文方法为例：

```powershell
Copy-Item `
  .\Unity-RL-Playground\gewu\Assets\Playground\configs\experiments\proposed.json `
  .\Unity-RL-Playground\gewu\Assets\Playground\configs\experiment_config.json -Force
```

在 Gewu 工程目录启动训练：

```bash
mlagents-learn gewu/Assets/Playground/configs/trainers/proposed.yaml \
  --run-id=efc_proposed_seed0 --force
```

出现监听提示后，打开 `ShipDeck.unity` 并点击 Play。多次重复实验时应使用不同的 run ID 和随机种子。主对比配置包括 `static_baseline`、`dynamic_baseline`、`dynamic_priors`、`dynamic_implicit`、`dynamic_lstm` 与 `proposed`；预测方法、预测时域和关节通道消融见 `unity/Assets/Playground/configs/manifest.json`。

## 引用

使用本代码时，请同时引用 EFC 论文和作为基础平台的 Gewu。完整 BibTeX 见 [英文 README](README.md#citation)，GitHub 的 **Cite this repository** 功能会读取 `CITATION.cff`。

当前 EFC 条目按 2026 年未正式出版稿填写；论文获得 DOI、期刊/会议、卷期页码或 arXiv 编号后，应更新 README 与 `CITATION.cff`。Gewu 的正式引用信息为：

```bibtex
@article{ye2026gewu,
  title   = {Gewu Playground: An Open-Source Robot Simulation Platform for Embodied Intelligence Research},
  author  = {Ye, Linqi and Xing, Boyang and Liang, Bin and Jiang, Lei and Peng, Yan},
  journal = {Science China Technological Sciences},
  year    = {2026},
  doi     = {10.1007/s11431-025-3253-2}
}
```

## 许可证与归属

代码按 Apache License 2.0 发布。仓库包含修改后的 Gewu agent 文件，并依赖上游 Gewu 工程运行。第三方归属及资产边界见 `NOTICE` 和 [docs/ASSETS.md](docs/ASSETS.md)；外部资产仍遵循各自许可证。
