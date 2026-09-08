#!/usr/bin/env python3
"""
V1 Feedforward Sign A/B Test Analysis

分析 Group A 和 Group B 的配对数据，判定哪个符号更好。

使用方法:
    python3 gewu/analyze_ab_test.py

输入:
    - Assets/Playground/results/feedforward_ab_test_A.csv
    - Assets/Playground/results/feedforward_ab_test_B.csv

输出:
    - 配对差值统计
    - 决策建议
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_ab_test():
    results_dir = Path("gewu/Assets/Playground/results")

    csv_a = results_dir / "feedforward_ab_test_A.csv"
    csv_b = results_dir / "feedforward_ab_test_B.csv"

    if not csv_a.exists():
        print(f"错误: 未找到 Group A 数据文件: {csv_a}")
        print("请先运行 Group A 测试")
        return

    if not csv_b.exists():
        print(f"错误: 未找到 Group B 数据文件: {csv_b}")
        print("请先运行 Group B 测试")
        return

    # 读取数据
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    print("=" * 60)
    print("V1 FEEDFORWARD SIGN A/B TEST RESULTS")
    print("=" * 60)

    # 验证数据
    if len(df_a) != len(df_b):
        print(f"警告: Group A ({len(df_a)} episodes) 和 Group B ({len(df_b)} episodes) 数量不匹配")
        min_len = min(len(df_a), len(df_b))
        df_a = df_a.iloc[:min_len]
        df_b = df_b.iloc[:min_len]

    # 验证 seed 配对
    if not (df_a['Seed'].values == df_b['Seed'].values).all():
        print("警告: Group A 和 Group B 的 seed 不匹配!")
        print(f"A seeds: {df_a['Seed'].values}")
        print(f"B seeds: {df_b['Seed'].values}")

    # 计算配对差值 (A - B)
    diff_steps = df_a['Steps'].values - df_b['Steps'].values
    diff_rmse = df_a['PostureRMSE'].values - df_b['PostureRMSE'].values

    # 统计
    mean_diff_steps = np.mean(diff_steps)
    std_diff_steps = np.std(diff_steps, ddof=1)

    mean_diff_rmse = np.mean(diff_rmse)
    std_diff_rmse = np.std(diff_rmse, ddof=1)

    # Group A 和 Group B 的原始统计
    mean_a_steps = df_a['Steps'].mean()
    mean_b_steps = df_b['Steps'].mean()
    mean_a_rmse = df_a['PostureRMSE'].mean()
    mean_b_rmse = df_b['PostureRMSE'].mean()

    print("\n" + "=" * 60)
    print("原始数据统计")
    print("=" * 60)
    print(f"Group A (positive sign):")
    print(f"  Mean Steps: {mean_a_steps:.1f} ± {df_a['Steps'].std():.1f}")
    print(f"  Mean Posture RMSE: {mean_a_rmse:.4f}")

    print(f"\nGroup B (negative sign):")
    print(f"  Mean Steps: {mean_b_steps:.1f} ± {df_b['Steps'].std():.1f}")
    print(f"  Mean Posture RMSE: {mean_b_rmse:.4f}")

    print("\n" + "=" * 60)
    print("配对差值分析 (A - B)")
    print("=" * 60)
    print(f"存活步数差值:")
    print(f"  均值: {mean_diff_steps:.1f}")
    print(f"  标准差: {std_diff_steps:.1f}")
    print(f"  范围: [{diff_steps.min():.0f}, {diff_steps.max():.0f}]")

    print(f"\nPosture RMSE 差值:")
    print(f"  均值: {mean_diff_rmse:.4f}")
    print(f"  标准差: {std_diff_rmse:.4f}")

    # 一致性分析
    positive_count = np.sum(diff_steps > 0)
    negative_count = np.sum(diff_steps < 0)
    zero_count = np.sum(diff_steps == 0)

    print(f"\n差值方向统计:")
    print(f"  A > B: {positive_count} episodes")
    print(f"  A < B: {negative_count} episodes")
    print(f"  A = B: {zero_count} episodes")

    # 逐 episode 对照表
    print("\n" + "=" * 60)
    print("逐 Episode 配对对照")
    print("=" * 60)
    print(f"{'Seed':<6} {'A_Steps':<10} {'B_Steps':<10} {'Diff':<8} {'A_RMSE':<10} {'B_RMSE':<10}")
    print("-" * 60)
    for i in range(len(df_a)):
        seed = df_a['Seed'].iloc[i]
        a_steps = df_a['Steps'].iloc[i]
        b_steps = df_b['Steps'].iloc[i]
        diff = diff_steps[i]
        a_rmse = df_a['PostureRMSE'].iloc[i]
        b_rmse = df_b['PostureRMSE'].iloc[i]
        print(f"{seed:<6} {a_steps:<10} {b_steps:<10} {diff:<+8.0f} {a_rmse:<10.4f} {b_rmse:<10.4f}")

    # 判定
    print("\n" + "=" * 60)
    print("判定结果")
    print("=" * 60)

    # V1 判据: 配对差值均值方向一致 且 10 个中至少 7 个同向
    consistency_threshold = 7

    if positive_count >= consistency_threshold:
        winner = "A (positive sign)"
        direction = "positive"
        confidence = f"{positive_count}/10 episodes favor A"
    elif negative_count >= consistency_threshold:
        winner = "B (negative sign)"
        direction = "negative"
        confidence = f"{negative_count}/10 episodes favor B"
    else:
        winner = "INCONCLUSIVE"
        direction = "mixed"
        confidence = f"Only {max(positive_count, negative_count)}/10 consistent"

    print(f"主要指标 (存活步数):")
    print(f"  胜者: {winner}")
    print(f"  一致性: {confidence}")
    print(f"  平均差值: {mean_diff_steps:.1f} steps")

    # 次要指标
    if winner != "INCONCLUSIVE":
        if mean_diff_rmse < 0:
            print(f"\n次要指标 (Posture RMSE):")
            print(f"  Group A has lower RMSE (better posture) by {abs(mean_diff_rmse):.4f}")
        elif mean_diff_rmse > 0:
            print(f"\n次要指标 (Posture RMSE):")
            print(f"  Group B has lower RMSE (better posture) by {mean_diff_rmse:.4f}")

    print("\n" + "=" * 60)
    print("行动建议")
    print("=" * 60)

    if direction == "positive":
        print("✓ 保持当前符号 (positive)")
        print(f"  PredictiveFeedforwardController.cs:95")
        print(f"  base_comp = K_roll * filtered_roll_error;  // 保持")
        action = "KEEP_CURRENT"
    elif direction == "negative":
        print("✗ 需要修改为取负符号 (negative)")
        print(f"  PredictiveFeedforwardController.cs:95")
        print(f"  base_comp = K_roll * filtered_roll_error;  // 改为:")
        print(f"  base_comp = -K_roll * filtered_roll_error; // V1: A/B test winner")
        action = "NEGATE_SIGN"
    else:
        print("? 结果不确定，建议:")
        print("  1. 检查两组数据是否有异常")
        print("  2. 增加测试 episodes 数量 (如 20 个)")
        print("  3. 或保持当前符号作为保守选择")
        action = "INCONCLUSIVE_KEEP_CURRENT"

    # 保存决策到文件
    decision_file = results_dir / "ab_test_decision.txt"
    with open(decision_file, 'w') as f:
        f.write(f"V1 Feedforward Sign A/B Test Decision\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Winner: {winner}\n")
        f.write(f"Action: {action}\n")
        f.write(f"Mean Diff (A-B): {mean_diff_steps:.1f} ± {std_diff_steps:.1f} steps\n")
        f.write(f"Consistency: {max(positive_count, negative_count)}/10 episodes\n")
        f.write(f"\nGroup A Mean: {mean_a_steps:.1f} steps\n")
        f.write(f"Group B Mean: {mean_b_steps:.1f} steps\n")

    print(f"\n决策已保存到: {decision_file}")
    print("=" * 60)

    return action

if __name__ == "__main__":
    analyze_ab_test()
