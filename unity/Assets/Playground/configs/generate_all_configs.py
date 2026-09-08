#!/usr/bin/env python3
"""
生成完整的实验配置系统
- 13 个 trainer YAML (6 主实验 + 7 消融)
- 13 个 experiment JSON
- manifest.json
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any

# 基础路径
CONFIGS_DIR = Path(__file__).parent
TRAINERS_DIR = CONFIGS_DIR / "trainers"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"

# 公共超参（除 LSTM 外完全一致）
COMMON_HYPERPARAMS = {
    "trainer_type": "ppo",
    "hyperparameters": {
        "batch_size": 2048,
        "buffer_size": 20480,
        "learning_rate": 0.0003,
        "learning_rate_schedule": "linear",
        "beta": 0.005,
        "beta_schedule": "linear",
        "epsilon": 0.2,
        "epsilon_schedule": "linear",
        "lambd": 0.95,
        "num_epoch": 3,
    },
    "network_settings": {
        "normalize": True,
        "hidden_units": 512,
        "num_layers": 3,
        "vis_encode_type": "simple",
    },
    "reward_signals": {
        "extrinsic": {
            "gamma": 0.995,
            "strength": 1.0,
        }
    },
    "max_steps": 10000000,  # 1e7
    "time_horizon": 1000,
    "summary_freq": 20000,
    "checkpoint_interval": 400000,
    "keep_checkpoints": 5,
    "threaded": False,
}

# 基础实验配置模板
BASE_EXPERIMENT_CONFIG = {
    "observation": {
        "enableCoMObservation": True,
        "enablePredictionObservation": True,
        "enableFeedforwardControl": True,
    },
    "feedforward": {
        "K_roll": 0.1,
        "K_pitch": 0.1,
        "max_roll_comp": 8.0,
        "max_pitch_comp": 8.0,
        "support_threshold": 0.7,
        "filter_alpha": 0.3,
        "prediction_horizon": 0.1,
        "predictor_order": 2,  # 0=zeroth, 1=first, 2=second(FD+EMA), 3=oracle
    },
    "deckIMU": {
        "sensorMode": "Oracle",
        "angularVelocityNoise": 0.001,
        "accelerationNoise": 0.01,
        "attitudeNoise": 0.1,
        "sensorDelay": 0.01,
    },
    "waveSpectrum": {
        "waveMode": "JONSWAP",
        "significantRollAmplitudeMin": 2.0,
        "significantRollAmplitudeMax": 5.0,
        "significantHeaveAmplitudeMin": 0.1,
        "significantHeaveAmplitudeMax": 0.5,
        "peakPeriodMin": 6.0,
        "peakPeriodMax": 14.0,
        "gamma": 3.3,
        "rollNaturalPeriodMin": 8.0,
        "rollNaturalPeriodMax": 15.0,
        "rollDampingRatioMin": 0.05,
        "rollDampingRatioMax": 0.15,
        "pitchNaturalPeriodMin": 5.0,
        "pitchNaturalPeriodMax": 10.0,
        "pitchDampingRatioMin": 0.1,
        "pitchDampingRatioMax": 0.2,
        "heaveNaturalPeriodMin": 4.0,
        "heaveNaturalPeriodMax": 8.0,
        "heaveDampingRatioMin": 0.2,
        "heaveDampingRatioMax": 0.4,
        "pitchScale": 0.8,
        "numComponents": 50,
        "omegaMin": 0.2,
        "omegaMax": 3.0,
        "waveSeed": -1,
    },
}


def generate_trainer_yaml(config_name: str, is_lstm: bool = False) -> Dict:
    """生成 trainer YAML 配置"""
    import copy
    config = {"behaviors": {"gewu": copy.deepcopy(COMMON_HYPERPARAMS)}}

    if is_lstm:
        config["behaviors"]["gewu"]["network_settings"]["memory"] = {
            "sequence_length": 64,
            "memory_size": 256,
        }

    return config


def generate_experiment_json(
    experiment_name: str,
    notes: str,
    enable_com: bool,
    enable_prediction: bool,
    enable_feedforward: bool,
    predictor_order: int = 2,
    prediction_horizon: float = 0.1,
    wave_mode: str = "JONSWAP",
) -> Dict:
    """生成实验 JSON 配置"""
    config = {
        "meta": {
            "experimentName": experiment_name,
            "notes": notes,
        }
    }

    # 深拷贝基础配置
    import copy
    config.update(copy.deepcopy(BASE_EXPERIMENT_CONFIG))

    # 应用观测配置
    config["observation"]["enableCoMObservation"] = enable_com
    config["observation"]["enablePredictionObservation"] = enable_prediction
    config["observation"]["enableFeedforwardControl"] = enable_feedforward

    # 应用预测器配置
    config["feedforward"]["predictor_order"] = predictor_order
    config["feedforward"]["prediction_horizon"] = prediction_horizon

    # 应用波浪模式
    config["waveSpectrum"]["waveMode"] = wave_mode

    return config


def main():
    """生成所有配置文件"""
    TRAINERS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    # 主实验配置
    main_experiments = [
        # (config_name, notes, enable_com, enable_pred, enable_ff, is_lstm, wave_mode)
        ("static_baseline", "Static platform baseline (33-dim obs)", False, False, False, False, "RegularSine"),
        ("dynamic_baseline", "Dynamic platform, pure feedback (33-dim obs)", False, False, False, False, "JONSWAP"),
        ("dynamic_priors", "Physical priors: CoM only (38-dim obs)", True, False, False, False, "JONSWAP"),
        ("dynamic_implicit", "Implicit prediction: CoM+Pred obs, no FF (57-dim obs)", True, True, False, False, "JONSWAP"),
        ("dynamic_lstm", "Recurrent baseline: CoM+LSTM (38-dim + memory)", True, False, False, True, "JONSWAP"),
        ("proposed", "Proposed method: CoM+Pred+FF (38+FF-dim obs)", True, True, True, False, "JONSWAP"),
    ]

    manifest = {"main_experiments": [], "ablations": []}

    print("=== 生成主实验配置 ===")
    for config_name, notes, enable_com, enable_pred, enable_ff, is_lstm, wave_mode in main_experiments:
        # Trainer YAML
        trainer_config = generate_trainer_yaml(config_name, is_lstm=is_lstm)
        trainer_path = TRAINERS_DIR / f"{config_name}.yaml"
        with open(trainer_path, "w") as f:
            yaml.dump(trainer_config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ {trainer_path}")

        # Experiment JSON
        exp_config = generate_experiment_json(
            experiment_name=config_name,
            notes=notes,
            enable_com=enable_com,
            enable_prediction=enable_pred,
            enable_feedforward=enable_ff,
            wave_mode=wave_mode,
        )
        exp_path = EXPERIMENTS_DIR / f"{config_name}.json"
        with open(exp_path, "w") as f:
            json.dump(exp_config, f, indent=2)
        print(f"✓ {exp_path}")

        manifest["main_experiments"].append({
            "name": config_name,
            "trainer": f"trainers/{config_name}.yaml",
            "experiment": f"experiments/{config_name}.json",
            "notes": notes,
        })

    # 消融实验配置
    ablation_experiments = [
        # 预测器类型组
        ("predictor_fdema", "Ablation: FD+EMA predictor (2nd-order)", 2),
        ("predictor_kalman", "Ablation: Kalman predictor (1st-order)", 1),
        ("predictor_oracle", "Ablation: Oracle predictor (analytical)", 3),

        # 预测时域组
        ("horizon_50ms", "Ablation: h=50ms prediction horizon", 2, 0.05),
        ("horizon_200ms", "Ablation: h=200ms prediction horizon", 2, 0.2),
    ]

    print("\n=== 生成消融实验配置 ===")
    for item in ablation_experiments:
        config_name = item[0]
        notes = item[1]
        predictor_order = item[2]
        prediction_horizon = item[3] if len(item) > 3 else 0.1

        # Trainer YAML (与 proposed 相同)
        trainer_config = generate_trainer_yaml(config_name, is_lstm=False)
        trainer_path = TRAINERS_DIR / f"{config_name}.yaml"
        with open(trainer_path, "w") as f:
            yaml.dump(trainer_config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ {trainer_path}")

        # Experiment JSON (基于 proposed)
        exp_config = generate_experiment_json(
            experiment_name=config_name,
            notes=notes,
            enable_com=True,
            enable_prediction=True,
            enable_feedforward=True,
            predictor_order=predictor_order,
            prediction_horizon=prediction_horizon,
        )
        exp_path = EXPERIMENTS_DIR / f"{config_name}.json"
        with open(exp_path, "w") as f:
            json.dump(exp_config, f, indent=2)
        print(f"✓ {exp_path}")

        manifest["ablations"].append({
            "name": config_name,
            "trainer": f"trainers/{config_name}.yaml",
            "experiment": f"experiments/{config_name}.json",
            "notes": notes,
        })

    # 补偿通道消融（M3 待实现，先生成配置）
    channel_experiments = [
        ("channel_knee_ankle", "Ablation: Knee+Ankle roll compensation"),
        ("channel_knee_ankle_hip", "Ablation: Knee+Ankle+Hip roll compensation"),
    ]

    for config_name, notes in channel_experiments:
        trainer_config = generate_trainer_yaml(config_name, is_lstm=False)
        trainer_path = TRAINERS_DIR / f"{config_name}.yaml"
        with open(trainer_path, "w") as f:
            yaml.dump(trainer_config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ {trainer_path}")

        exp_config = generate_experiment_json(
            experiment_name=config_name,
            notes=f"{notes} (M3 channel config)",
            enable_com=True,
            enable_prediction=True,
            enable_feedforward=True,
        )
        # M3 通道配置（待 M3 实现后添加字段）
        exp_config["feedforward"]["enable_knee"] = True
        exp_config["feedforward"]["enable_ankle"] = True
        exp_config["feedforward"]["enable_hip"] = (config_name == "channel_knee_ankle_hip")

        exp_path = EXPERIMENTS_DIR / f"{config_name}.json"
        with open(exp_path, "w") as f:
            json.dump(exp_config, f, indent=2)
        print(f"✓ {exp_path}")

        manifest["ablations"].append({
            "name": config_name,
            "trainer": f"trainers/{config_name}.yaml",
            "experiment": f"experiments/{config_name}.json",
            "notes": notes,
        })

    # 生成 manifest
    manifest_path = CONFIGS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✓ {manifest_path}")

    print(f"\n=== 汇总 ===")
    print(f"主实验: {len(main_experiments)} × 3 seeds = {len(main_experiments) * 3} 次")
    print(f"消融: {len(ablation_experiments) + len(channel_experiments)} × 3 seeds = {(len(ablation_experiments) + len(channel_experiments)) * 3} 次")
    print(f"总计: {(len(main_experiments) + len(ablation_experiments) + len(channel_experiments)) * 3} 次训练")


if __name__ == "__main__":
    main()
