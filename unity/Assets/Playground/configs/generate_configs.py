#!/usr/bin/env python3
"""
P0-3 配置文件生成脚本

功能：
1. 自动生成所有实验配置文件
2. 确保参数一致性
3. 生成实验矩阵配置

使用：
    python generate_configs.py
"""

import json
import os
from pathlib import Path


def create_base_config():
    """创建基础配置模板"""
    return {
        "feedforward": {
            "K_roll": 0.1,
            "K_pitch": 0.1,
            "max_roll_comp": 8.0,
            "max_pitch_comp": 8.0,
            "support_threshold": 0.7,
            "filter_alpha": 0.3,
            "prediction_horizon": 0.1,
            "predictor_order": 2
        },
        "platform": {
            "rollAmplitude": 5.0,
            "rollFrequency": 0.2,
            "rollPhase": 0.0,
            "pitchAmplitude": 0.0,  # 纯横摇：pitch=0
            "pitchFrequency": 0.2,
            "pitchPhase": 0.0,
            "heaveAmplitude": 0.0,
            "heaveFrequency": 0.1,
            "heavePhase": 0.0,
            "enableMotion": True,
            "useCurriculumLearning": False
        },
        "domainRandomization": {
            "enableMassRandomization": False,
            "massMultiplierMin": 0.9,
            "massMultiplierMax": 1.1,
            "enableCoMRandomization": False,
            "comOffsetRange": 0.05,
            "enableFrictionRandomization": False,
            "frictionMin": 0.4,
            "frictionMax": 1.0,
            "enableExternalForce": False,
            "forceMin": 0.0,
            "forceMax": 50.0,
            "forceIntervalMin": 2.0,
            "forceIntervalMax": 5.0,
            "forceDuration": 0.1
        },
        "meta": {
            "experimentName": "",
            "configVersion": "1.0",
            "notes": "",
            "timestamp": ""
        }
    }


def save_config(config, filename, description=""):
    """保存配置文件"""
    config["meta"]["notes"] = description

    filepath = Path(__file__).parent / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✓ 已生成: {filename}")
    return filepath


def generate_main_experiments():
    """生成主实验配置（P3 表格）"""
    print("\n=== 生成主实验配置 ===\n")

    configs = []

    # 1. Static-Baseline (33 dim)
    config = create_base_config()
    config["meta"]["experimentName"] = "static_baseline_33dim"
    config["platform"]["enableMotion"] = False  # 静态平台
    save_config(config, "static_baseline_33dim.json",
                "Static-Baseline: 静态平台，基础观测（33维）")
    configs.append(("static_baseline_33dim", 33))

    # 2. Dynamic-Baseline (33 dim)
    config = create_base_config()
    config["meta"]["experimentName"] = "dynamic_baseline_33dim"
    save_config(config, "dynamic_baseline_33dim.json",
                "Dynamic-Baseline: 动态平台，基础观测（33维）")
    configs.append(("dynamic_baseline_33dim", 33))

    # 3. Dynamic-Priors (38 dim)
    config = create_base_config()
    config["meta"]["experimentName"] = "dynamic_priors_38dim"
    save_config(config, "dynamic_priors_38dim.json",
                "Dynamic-Priors: +CoM+接触观测（38维）")
    configs.append(("dynamic_priors_38dim", 38))

    # 4. Dynamic-Implicit (57 dim)
    config = create_base_config()
    config["meta"]["experimentName"] = "dynamic_implicit_57dim"
    save_config(config, "dynamic_implicit_57dim.json",
                "Dynamic-Implicit: +预测观测（57维）")
    configs.append(("dynamic_implicit_57dim", 57))

    # 5. Dynamic-Full-Implicit (62 dim)
    config = create_base_config()
    config["meta"]["experimentName"] = "dynamic_full_implicit_62dim"
    save_config(config, "dynamic_full_implicit_62dim.json",
                "Dynamic-Full-Implicit: +CoM+预测观测（62维）")
    configs.append(("dynamic_full_implicit_62dim", 62))

    # 6. Dynamic-LSTM (38 dim, memory)
    config = create_base_config()
    config["meta"]["experimentName"] = "dynamic_lstm_38dim"
    config["meta"]["notes"] = ("Dynamic-LSTM: +CoM，使用LSTM记忆（38维）\n"
                               "训练配置需额外添加：\n"
                               "  network_settings:\n"
                               "    memory:\n"
                               "      sequence_length: 64\n"
                               "      memory_size: 256")
    save_config(config, "dynamic_lstm_38dim.json", config["meta"]["notes"])
    configs.append(("dynamic_lstm_38dim", 38))

    # 7. Proposed-Explicit (38 dim) - 论文声称的配置
    config = create_base_config()
    config["meta"]["experimentName"] = "proposed_explicit_38dim"
    save_config(config, "proposed_explicit_38dim.json",
                "Proposed-Explicit: +CoM+前馈（38维），论文声称配置")
    configs.append(("proposed_explicit_38dim", 38))

    # 8. Proposed-Full (62 dim)
    config = create_base_config()
    config["meta"]["experimentName"] = "proposed_full_62dim"
    save_config(config, "proposed_full_62dim.json",
                "Proposed-Full: +CoM+预测+前馈（62维）")
    configs.append(("proposed_full_62dim", 62))

    print(f"\n✓ 主实验配置生成完成：{len(configs)} 个\n")
    return configs


def generate_sensitivity_experiments():
    """生成敏感性分析配置"""
    print("\n=== 生成敏感性分析配置 ===\n")

    base = create_base_config()
    base["meta"]["experimentName"] = "proposed"  # 基于 Proposed 方法

    # 1. 预测阶数
    print("预测阶数变化（4 × 3 seeds = 12 runs）:")
    for order, name in [(0, "zeroth"), (1, "first"), (2, "second"), (3, "oracle")]:
        config = json.loads(json.dumps(base))  # 深拷贝
        config["feedforward"]["predictor_order"] = order
        config["meta"]["experimentName"] = f"sensitivity_predictor_{name}"
        save_config(config, f"sensitivity_predictor_{name}.json",
                    f"敏感性分析：预测阶数 = {name}")

    # 2. 预测时域
    print("\n预测时域变化（2 × 3 seeds = 6 runs，0.1 复用主实验）:")
    for horizon in [0.05, 0.2]:
        config = json.loads(json.dumps(base))
        config["feedforward"]["prediction_horizon"] = horizon
        config["meta"]["experimentName"] = f"sensitivity_horizon_{int(horizon*1000)}ms"
        save_config(config, f"sensitivity_horizon_{int(horizon*1000)}ms.json",
                    f"敏感性分析：预测时域 = {horizon}s")

    # 3. K_roll
    print("\nK_roll 变化（2 × 3 seeds = 6 runs，0.1 复用主实验）:")
    for k_roll in [0.05, 0.2]:
        config = json.loads(json.dumps(base))
        config["feedforward"]["K_roll"] = k_roll
        config["meta"]["experimentName"] = f"sensitivity_kroll_{int(k_roll*100)}"
        save_config(config, f"sensitivity_kroll_{int(k_roll*100)}.json",
                    f"敏感性分析：K_roll = {k_roll}")

    # 4. 历史长度（需要修改 ExplicitVelocityPredictor）
    print("\n历史长度变化（2 × 3 seeds = 6 runs，5 复用主实验）:")
    for history_k in [3, 10]:
        config = json.loads(json.dumps(base))
        config["meta"]["experimentName"] = f"sensitivity_history_{history_k}"
        config["meta"]["notes"] = (f"敏感性分析：历史长度 = {history_k}\n"
                                   f"需要在 ExplicitVelocityPredictor.maxHistorySize 设置为 {history_k}")
        save_config(config, f"sensitivity_history_{history_k}.json",
                    config["meta"]["notes"])

    print("\n✓ 敏感性分析配置生成完成\n")


def generate_test_configs():
    """生成测试配置"""
    print("\n=== 生成测试配置 ===\n")

    # 对角轴倾斜模式（用于验证 P0 发现）
    config = create_base_config()
    config["meta"]["experimentName"] = "test_diagonal_tilt"
    config["platform"]["pitchAmplitude"] = 4.0  # roll=pitch=4°
    config["platform"]["rollAmplitude"] = 4.0
    save_config(config, "test_diagonal_tilt.json",
                "测试配置：对角轴倾斜（roll=pitch=4°），复现旧数据")

    # 纯横摇模式（论文声称）
    config = create_base_config()
    config["meta"]["experimentName"] = "test_pure_roll"
    config["platform"]["rollAmplitude"] = 5.0
    config["platform"]["pitchAmplitude"] = 0.0  # 确保 pitch=0
    save_config(config, "test_pure_roll.json",
                "测试配置：纯横摇（roll=5°, pitch=0），论文声称配置")

    print("✓ 测试配置生成完成\n")


def main():
    """主函数"""
    print("=" * 60)
    print("P0-3 参数配置化系统 - 配置文件生成器")
    print("=" * 60)

    # 生成各类配置
    main_configs = generate_main_experiments()
    generate_sensitivity_experiments()
    generate_test_configs()

    # 生成配置清单
    print("\n=== 生成配置清单 ===\n")
    manifest = {
        "generated_at": "2026-08-29",
        "config_version": "1.0",
        "main_experiments": [
            {"name": name, "observation_dim": dim} for name, dim in main_configs
        ],
        "total_main_runs": len(main_configs) * 5,  # 每配置 5 seeds
        "total_sensitivity_runs": 12 + 6 + 6 + 6,  # 30
        "total_runs": len(main_configs) * 5 + 30
    }

    manifest_path = Path(__file__).parent / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"✓ 配置清单已生成: manifest.json")
    print(f"\n总计：")
    print(f"  - 主实验: {len(main_configs)} 配置 × 5 seeds = {len(main_configs) * 5} runs")
    print(f"  - 敏感性分析: ~30 runs")
    print(f"  - 总计: ~{len(main_configs) * 5 + 30} runs\n")

    print("=" * 60)
    print("✓ 所有配置文件生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
