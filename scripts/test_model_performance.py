#!/usr/bin/env python3
"""
模型性能测试脚本

功能：
1. 自动修改 Unity 场景中的平台倾角
2. 运行多个测试 Episode
3. 统计每个倾角下的平均存活步数
4. 生成对比表格和可视化图表

使用方法：
1. 确保 Unity 项目已打开，机器人已加载训练好的模型
2. 修改下方的配置参数（模型名称、测试角度等）
3. 运行脚本：python test_model_performance.py
4. 查看生成的结果文件：
   - test_results_summary.csv（汇总表格）
   - test_results_plot.png（可视化图表）

注意事项：
- 需要安装：pip install pandas matplotlib numpy
- 需要从 Unity Console 或 TensorBoard 日志中手动提取 Episode Length 数据
- 或者使用 ML-Agents Python API 进行自动化测试
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# ===================== 配置参数 =====================

# 测试配置
MODELS = {
    'gewu_full': '完整版本 (Full)',
    'gewu_nocom': '去除CoM (NoCoM)',
    'gewu_nofoot': '去除Foot (NoFoot)',
    'gewu_baseline': '基线版本 (Baseline)'
}

# 测试角度列表（度）
TEST_ANGLES = [0, 1, 2, 3, 4, 5]

# 每个角度测试的 Episode 数量
EPISODES_PER_ANGLE = 20

# 输出路径
OUTPUT_DIR = Path(__file__).parent / 'test_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# ===================== 主要功能 =====================

def parse_unity_console_log(log_file):
    """
    从 Unity Console 日志中解析 Episode Length

    日志格式示例：
    [RobotRLAgent] Episode 1 完成: 步数=450, 奖励=1230.5
    """
    results = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Episode' in line and '步数=' in line:
                # 解析行
                parts = line.split('步数=')
                if len(parts) > 1:
                    steps_str = parts[1].split(',')[0].strip()
                    try:
                        steps = int(steps_str)
                        results.append(steps)
                    except ValueError:
                        continue

    return results


def load_csv_results(csv_file):
    """
    从 CSV 文件加载测试结果

    CSV 格式：
    ModelName,RollAmplitude,Episode,Steps,CumulativeReward,FallReason
    gewu_full,0.0,1,980,1850.23,Timeout
    gewu_full,0.0,2,950,1820.45,Pitch
    ...
    """
    if not Path(csv_file).exists():
        print(f"警告: 文件不存在 {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    return df


def generate_summary_table(csv_file, output_file):
    """
    生成汇总表格：每个模型在不同角度下的平均步数

    输出格式：
    | Model      | 0°   | 1°   | 2°   | 3°   | 4°   | 5°   |
    |------------|------|------|------|------|------|------|
    | Full       | 980  | 920  | 850  | 780  | 650  | 520  |
    | NoCoM      | 970  | 880  | 790  | 680  | 540  | 420  |
    | NoFoot     | 975  | 890  | 800  | 700  | 560  | 440  |
    | Baseline   | 950  | 820  | 720  | 600  | 480  | 350  |
    """
    df = load_csv_results(csv_file)
    if df is None:
        print("无法生成汇总表格：CSV 文件不存在")
        return None

    # 计算每个模型在每个角度下的平均步数和标准差
    summary = df.groupby(['ModelName', 'RollAmplitude'])['Steps'].agg(['mean', 'std', 'count'])
    summary = summary.reset_index()

    # 透视表格
    pivot_mean = summary.pivot(index='ModelName', columns='RollAmplitude', values='mean')
    pivot_std = summary.pivot(index='ModelName', columns='RollAmplitude', values='std')

    # 保存为 CSV
    pivot_mean.to_csv(output_file)
    print(f"✓ 汇总表格已保存到: {output_file}")

    # 打印到控制台
    print("\n" + "="*80)
    print("平均存活步数统计表")
    print("="*80)
    print(pivot_mean.to_string())
    print("\n标准差:")
    print(pivot_std.to_string())
    print("="*80 + "\n")

    return pivot_mean, pivot_std


def plot_performance_comparison(csv_file, output_file):
    """
    绘制性能对比图：折线图显示不同模型在不同角度下的表现
    """
    df = load_csv_results(csv_file)
    if df is None:
        print("无法生成图表：CSV 文件不存在")
        return

    # 计算均值和标准差
    summary = df.groupby(['ModelName', 'RollAmplitude'])['Steps'].agg(['mean', 'std'])
    summary = summary.reset_index()

    # 创建图表
    plt.figure(figsize=(12, 7))

    # 为每个模型绘制折线
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'D']

    for idx, (model_key, model_name) in enumerate(MODELS.items()):
        model_data = summary[summary['ModelName'] == model_key]

        if len(model_data) == 0:
            print(f"警告: 模型 {model_key} 无数据")
            continue

        x = model_data['RollAmplitude'].values
        y = model_data['mean'].values
        yerr = model_data['std'].values

        plt.plot(x, y, marker=markers[idx], color=colors[idx],
                linewidth=2, markersize=8, label=model_name)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=colors[idx])

    plt.xlabel('平台横摇幅度 (度)', fontsize=14, fontproperties='SimHei')
    plt.ylabel('平均存活步数', fontsize=14, fontproperties='SimHei')
    plt.title('不同模型在动态平台上的性能对比', fontsize=16, fontproperties='SimHei', fontweight='bold')
    plt.legend(fontsize=12, prop={'family': 'SimHei'})
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 性能对比图已保存到: {output_file}")
    plt.close()


def plot_fall_reason_distribution(csv_file, output_file):
    """
    绘制失败原因分布图：堆叠柱状图显示不同角度下的失败原因
    """
    df = load_csv_results(csv_file)
    if df is None:
        return

    # 统计失败原因
    fall_stats = df.groupby(['RollAmplitude', 'FallReason']).size().unstack(fill_value=0)

    # 绘制堆叠柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    fall_stats.plot(kind='bar', stacked=True, ax=ax,
                   color=['#E63946', '#F77F00', '#06AED5', '#52B788'])

    ax.set_xlabel('平台横摇幅度 (度)', fontsize=14, fontproperties='SimHei')
    ax.set_ylabel('Episode 数量', fontsize=14, fontproperties='SimHei')
    ax.set_title('不同角度下的失败原因分布', fontsize=16, fontproperties='SimHei', fontweight='bold')
    ax.legend(title='失败原因', prop={'family': 'SimHei'})
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 失败原因分布图已保存到: {output_file}")
    plt.close()


def generate_latex_table(csv_file, output_file):
    """
    生成 LaTeX 格式的表格，可直接用于论文
    """
    df = load_csv_results(csv_file)
    if df is None:
        return

    summary = df.groupby(['ModelName', 'RollAmplitude'])['Steps'].agg(['mean', 'std'])
    summary = summary.reset_index()
    pivot_mean = summary.pivot(index='ModelName', columns='RollAmplitude', values='mean')
    pivot_std = summary.pivot(index='ModelName', columns='RollAmplitude', values='std')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Average Survival Steps under Different Platform Roll Amplitudes}\n")
        f.write("\\label{tab:ablation_survival_steps}\n")
        f.write("\\begin{tabular}{l" + "c" * len(TEST_ANGLES) + "}\n")
        f.write("\\toprule\n")

        # 表头
        angles_str = " & ".join([f"${angle}°$" for angle in TEST_ANGLES])
        f.write(f"Model & {angles_str} \\\\\n")
        f.write("\\midrule\n")

        # 数据行
        for model_key, model_name in MODELS.items():
            if model_key not in pivot_mean.index:
                continue

            row_data = []
            for angle in TEST_ANGLES:
                if angle in pivot_mean.columns:
                    mean_val = pivot_mean.loc[model_key, angle]
                    std_val = pivot_std.loc[model_key, angle]
                    row_data.append(f"{mean_val:.0f} $\\pm$ {std_val:.0f}")
                else:
                    row_data.append("--")

            f.write(f"{model_name} & {' & '.join(row_data)} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ LaTeX 表格已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析模型测试结果')
    parser.add_argument('--csv', type=str, default='Assets/Playground/test_results.csv',
                       help='测试结果 CSV 文件路径')
    args = parser.parse_args()

    csv_file = Path(args.csv)

    if not csv_file.exists():
        print(f"错误: 找不到测试结果文件 {csv_file}")
        print("\n请先完成以下步骤：")
        print("1. 在 Unity 中给机器人添加 RobotPerformanceLogger 组件")
        print("2. 设置测试参数（测试角度、Episode 数量等）")
        print("3. 运行测试，生成 test_results.csv")
        print("4. 再运行此脚本进行分析\n")
        return

    print("="*80)
    print("模型性能分析工具")
    print("="*80)

    # 生成汇总表格
    summary_file = OUTPUT_DIR / 'test_results_summary.csv'
    generate_summary_table(csv_file, summary_file)

    # 生成性能对比图
    plot_file = OUTPUT_DIR / 'test_results_plot.png'
    plot_performance_comparison(csv_file, plot_file)

    # 生成失败原因分布图
    fall_reason_file = OUTPUT_DIR / 'fall_reason_distribution.png'
    plot_fall_reason_distribution(csv_file, fall_reason_file)

    # 生成 LaTeX 表格
    latex_file = OUTPUT_DIR / 'latex_table.tex'
    generate_latex_table(csv_file, latex_file)

    print("\n" + "="*80)
    print("分析完成！生成的文件：")
    print(f"  1. 汇总表格: {summary_file}")
    print(f"  2. 性能对比图: {plot_file}")
    print(f"  3. 失败原因分布图: {fall_reason_file}")
    print(f"  4. LaTeX 表格: {latex_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
