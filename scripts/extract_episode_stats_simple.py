#!/usr/bin/env python3
"""
从 TensorBoard 日志中提取 Episode Length 统计数据（不需要 TensorFlow）

使用 tensorboard 库（更轻量）

安装：pip install tensorboard

使用方法：
    python extract_episode_stats_simple.py Assets/Playground/results/122222test
"""

import sys
import os
from pathlib import Path
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("错误: 需要安装 tensorboard")
    print("运行: pip install tensorboard")
    sys.exit(1)


def extract_episode_lengths(logdir):
    """
    从 TensorBoard 日志目录提取 Episode Length 数据
    """
    episode_lengths = []

    # 查找所有 tfevents 文件
    logdir_path = Path(logdir)
    tfevents_files = list(logdir_path.glob("**/events.out.tfevents.*"))
    tfevents_files = [f for f in tfevents_files if not str(f).endswith('.meta')]

    if not tfevents_files:
        print(f"错误: 在 {logdir} 中未找到 TensorBoard 日志文件")
        return []

    print(f"找到 {len(tfevents_files)} 个日志文件")

    for tfevents_file in tfevents_files:
        print(f"正在解析: {tfevents_file.name}")

        try:
            # 使用 EventAccumulator 读取数据
            ea = event_accumulator.EventAccumulator(str(tfevents_file))
            ea.Reload()

            # 打印所有可用的标量指标
            print(f"  可用指标: {list(ea.Tags()['scalars'])[:5]}...")  # 只显示前5个

            # 查找 Episode Length
            if 'Environment/Episode Length' in ea.Tags()['scalars']:
                events = ea.Scalars('Environment/Episode Length')
                for event in events:
                    episode_lengths.append(event.value)
                print(f"  提取了 {len(events)} 个 Episode Length 数据点")
            else:
                print(f"  警告: 未找到 'Environment/Episode Length' 指标")

        except Exception as e:
            print(f"警告: 解析文件时出错: {e}")
            continue

    return episode_lengths


def calculate_stats(data):
    """计算统计信息"""
    if not data:
        return None

    data = np.array(data)

    return {
        'count': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }


def print_stats(stats, model_name="Unknown", angle="Unknown"):
    """打印统计摘要"""
    if not stats:
        print("错误: 没有数据可统计")
        return

    print("\n" + "="*60)
    print("Episode Length 统计摘要")
    print("="*60)
    print(f"模型: {model_name}")
    print(f"角度: {angle}°")
    print(f"Episode 数量: {stats['count']}")
    print(f"平均步数: {stats['mean']:.1f} ± {stats['std']:.1f}")
    print(f"中位数: {stats['median']:.1f}")
    print(f"最小值: {stats['min']:.0f}")
    print(f"最大值: {stats['max']:.0f}")
    print(f"25%分位数: {stats['q25']:.1f}")
    print(f"75%分位数: {stats['q75']:.1f}")
    print("="*60)

    print(f"\n📝 论文表格格式: {stats['mean']:.0f} ± {stats['std']:.0f}")
    print("="*60 + "\n")


def save_to_csv(data, output_file, model_name="Unknown", angle="Unknown"):
    """保存到 CSV 文件"""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ModelName', 'RollAmplitude', 'Episode', 'Steps'])

        for i, steps in enumerate(data, 1):
            writer.writerow([model_name, angle, i, int(steps)])

    print(f"✓ 数据已保存到: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python extract_episode_stats_simple.py <logdir> [model_name] [angle]")
        print("示例: python extract_episode_stats_simple.py Assets/Playground/results/122222test gewu_full 0")
        sys.exit(1)

    logdir = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "gewu"
    angle = sys.argv[3] if len(sys.argv) > 3 else "0"

    # 提取数据
    episode_lengths = extract_episode_lengths(logdir)

    if not episode_lengths:
        print("\n错误: 未能提取到任何 Episode Length 数据")
        print("\n这可能是因为:")
        print("  1. 这是训练日志，不是测试日志（训练日志不记录单个Episode长度）")
        print("  2. 测试时间太短，Episode还没有结束")
        print("\n建议:")
        print("  - 使用 TensorBoard 查看: tensorboard --logdir=" + logdir)
        print("  - 或者使用我提供的 EpisodeStatsLogger.cs 组件重新测试")
        sys.exit(1)

    print(f"\n✓ 成功提取 {len(episode_lengths)} 个 Episode 的数据")

    # 计算统计信息
    stats = calculate_stats(episode_lengths)

    # 打印统计摘要
    print_stats(stats, model_name, angle)

    # 保存到 CSV
    output_file = f"episode_stats_{model_name}_{angle}deg.csv"
    save_to_csv(episode_lengths, output_file, model_name, angle)

    # 打印最近10个Episode
    print("\n最近10个Episode的步数:")
    recent = episode_lengths[-10:] if len(episode_lengths) >= 10 else episode_lengths
    for i, steps in enumerate(recent, len(episode_lengths) - len(recent) + 1):
        print(f"  Episode {i}: {int(steps)} 步")


if __name__ == '__main__':
    main()
