#!/usr/bin/env python3
"""
run_sweep.py - ML-Agents Parameter Sweep Runner

功能：
1. 分离训练器配置（YAML）和实验配置（JSON）
2. 支持 --env-args 传递 Unity 环境参数
3. 支持 --no-graphics 无头模式训练
4. 实时流式日志输出
5. 超时/重试/恢复机制

用法：
    # 单次运行
    python run_sweep.py --trainer config.yaml --experiment configs/exp001.json --run-id exp001

    # 扫描多个实验配置
    python run_sweep.py --trainer config.yaml --sweep-dir configs/sweep1/ --output results/sweep1

    # 恢复中断的扫描
    python run_sweep.py --resume results/sweep1/sweep_state.json

示例实验配置 JSON (configs/exp001.json):
{
  "meta": {
    "experimentName": "exp001_baseline",
    "notes": "Baseline without feedforward"
  },
  "observation": {
    "enableCoMObservation": true,
    "enablePredictionObservation": false,
    "enableFeedforwardControl": false
  },
  "waveSpectrum": {
    "waveMode": "JONSWAP",
    "significantRollAmplitudeMin": 2.0,
    "significantRollAmplitudeMax": 5.0
  }
}
"""

import argparse
import json
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import signal
import shutil
from datetime import datetime


class SweepRunner:
    """参数扫描运行器"""

    def __init__(
        self,
        trainer_config: str,
        unity_exe: Optional[str] = None,
        no_graphics: bool = True,
        timeout: int = 7200,  # 2 hours default
        max_retries: int = 3,
        base_port: int = 5004,
    ):
        self.trainer_config = Path(trainer_config)
        self.unity_exe = Path(unity_exe) if unity_exe else None
        self.no_graphics = no_graphics
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_port = base_port
        self.current_process: Optional[subprocess.Popen] = None

        # 验证训练器配置文件存在
        if not self.trainer_config.exists():
            raise FileNotFoundError(f"Trainer config not found: {self.trainer_config}")

    def run_single_experiment(
        self,
        experiment_config: Dict,
        run_id: str,
        output_dir: Path,
        env_args: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """
        运行单个实验

        Args:
            experiment_config: 实验配置字典
            run_id: 运行 ID
            output_dir: 输出目录
            env_args: Unity 环境参数

        Returns:
            (success, message) tuple
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存实验配置快照
        config_snapshot = output_dir / "experiment_config.json"
        with open(config_snapshot, "w") as f:
            json.dump(experiment_config, f, indent=2)

        # 构建 mlagents-learn 命令
        cmd = self._build_command(run_id, output_dir, env_args)

        # 运行训练（带重试）
        for attempt in range(self.max_retries):
            try:
                print(f"\n{'='*80}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"开始运行: {run_id} (尝试 {attempt + 1}/{self.max_retries})")
                print(f"{'='*80}")
                print(f"命令: {' '.join(cmd)}\n")

                success, msg = self._run_training(cmd, output_dir / "train.log")

                if success:
                    print(f"\n✓ 实验成功: {run_id}")
                    return True, "Training completed successfully"
                else:
                    print(f"\n✗ 实验失败 (尝试 {attempt + 1}/{self.max_retries}): {msg}")
                    if attempt < self.max_retries - 1:
                        wait_time = 30 * (attempt + 1)
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)

            except KeyboardInterrupt:
                print("\n\n⚠ 用户中断")
                if self.current_process:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=10)
                raise

        return False, f"Failed after {self.max_retries} attempts"

    def _build_command(
        self, run_id: str, output_dir: Path, env_args: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """构建 mlagents-learn 命令"""
        cmd = [
            "mlagents-learn",
            str(self.trainer_config),
            "--run-id", run_id,
            "--results-dir", str(output_dir),
            "--force",  # 覆盖已存在的运行
        ]

        # Unity 可执行文件
        if self.unity_exe:
            cmd.extend(["--env", str(self.unity_exe)])

        # 无头模式
        if self.no_graphics:
            cmd.append("--no-graphics")

        # 环境参数
        if env_args:
            for key, value in env_args.items():
                cmd.extend(["--env-args", f"--{key}={value}"])

        return cmd

    def _run_training(self, cmd: List[str], log_file: Path) -> Tuple[bool, str]:
        """
        运行训练进程（带超时和流式日志）

        Returns:
            (success, message) tuple
        """
        try:
            # 启动进程
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # 行缓冲
                universal_newlines=True,
            )

            start_time = time.time()

            # 流式输出到终端和日志文件
            with open(log_file, "w") as f:
                for line in self.current_process.stdout:
                    # 写入日志文件
                    f.write(line)
                    f.flush()

                    # 输出到终端（带时间戳）
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] {line}", end='')

                    # 检查超时
                    if time.time() - start_time > self.timeout:
                        print(f"\n⚠ 训练超时 ({self.timeout}s)，终止进程")
                        self.current_process.terminate()
                        self.current_process.wait(timeout=10)
                        return False, "Training timeout"

            # 等待进程结束
            returncode = self.current_process.wait()
            self.current_process = None

            if returncode == 0:
                return True, "Success"
            else:
                return False, f"Process exited with code {returncode}"

        except Exception as e:
            if self.current_process:
                self.current_process.terminate()
                self.current_process = None
            return False, f"Exception: {str(e)}"

    def run_sweep(
        self,
        experiment_configs: List[Tuple[str, Dict]],
        output_base: Path,
        state_file: Optional[Path] = None,
    ) -> Dict:
        """
        运行参数扫描

        Args:
            experiment_configs: List of (experiment_id, config_dict) tuples
            output_base: 扫描输出基础目录
            state_file: 状态文件路径（用于恢复）

        Returns:
            扫描结果摘要
        """
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)

        if state_file is None:
            state_file = output_base / "sweep_state.json"

        # 加载或初始化状态
        state = self._load_state(state_file)

        results = []
        total = len(experiment_configs)

        for idx, (exp_id, config) in enumerate(experiment_configs, 1):
            # 跳过已完成的实验
            if exp_id in state.get("completed", []):
                print(f"\n[{idx}/{total}] 跳过已完成: {exp_id}")
                continue

            print(f"\n{'#'*80}")
            print(f"扫描进度: [{idx}/{total}] {exp_id}")
            print(f"{'#'*80}")

            # 运行实验
            exp_output = output_base / exp_id
            success, msg = self.run_single_experiment(
                experiment_config=config,
                run_id=exp_id,
                output_dir=exp_output,
            )

            # 记录结果
            result = {
                "experiment_id": exp_id,
                "success": success,
                "message": msg,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)

            # 更新状态
            if success:
                state.setdefault("completed", []).append(exp_id)
            else:
                state.setdefault("failed", []).append(exp_id)

            self._save_state(state, state_file)

        # 保存最终结果
        summary = {
            "total": total,
            "completed": len([r for r in results if r["success"]]),
            "failed": len([r for r in results if not r["success"]]),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        summary_file = output_base / "sweep_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"扫描完成！")
        print(f"  总计: {summary['total']}")
        print(f"  成功: {summary['completed']}")
        print(f"  失败: {summary['failed']}")
        print(f"  结果: {summary_file}")
        print(f"{'='*80}\n")

        return summary

    def _load_state(self, state_file: Path) -> Dict:
        """加载扫描状态"""
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return {}

    def _save_state(self, state: Dict, state_file: Path):
        """保存扫描状态"""
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)


def load_experiment_configs(sweep_dir: Path) -> List[Tuple[str, Dict]]:
    """
    从目录加载所有实验配置

    Args:
        sweep_dir: 包含 JSON 配置文件的目录

    Returns:
        List of (experiment_id, config_dict) tuples
    """
    configs = []
    for json_file in sorted(sweep_dir.glob("*.json")):
        with open(json_file) as f:
            config = json.load(f)

        # 从文件名或 meta.experimentName 推导 experiment_id
        exp_id = config.get("meta", {}).get("experimentName", json_file.stem)
        configs.append((exp_id, config))

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="ML-Agents 参数扫描工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 基本参数
    parser.add_argument("--trainer", required=True, help="训练器配置 YAML 文件")
    parser.add_argument("--run-id", help="单次运行 ID（单实验模式）")
    parser.add_argument("--experiment", help="单个实验配置 JSON 文件")

    # 扫描模式
    parser.add_argument("--sweep-dir", help="实验配置目录（扫描模式）")
    parser.add_argument("--output", help="输出目录", default="results")

    # 恢复模式
    parser.add_argument("--resume", help="恢复扫描（状态文件路径）")

    # Unity 参数
    parser.add_argument("--unity-exe", help="Unity 可执行文件路径")
    parser.add_argument("--no-graphics", action="store_true", help="无头模式")
    parser.add_argument("--env-args", nargs="*", help="Unity 环境参数 (格式: key=value)")

    # 运行参数
    parser.add_argument("--timeout", type=int, default=7200, help="训练超时（秒），默认 2 小时")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--base-port", type=int, default=5004, help="基础端口号")

    args = parser.parse_args()

    # 解析环境参数
    env_args = {}
    if args.env_args:
        for arg in args.env_args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                env_args[key] = value

    # 创建运行器
    runner = SweepRunner(
        trainer_config=args.trainer,
        unity_exe=args.unity_exe,
        no_graphics=args.no_graphics,
        timeout=args.timeout,
        max_retries=args.max_retries,
        base_port=args.base_port,
    )

    # 模式 1: 恢复扫描
    if args.resume:
        state_file = Path(args.resume)
        if not state_file.exists():
            print(f"错误: 状态文件不存在: {state_file}")
            sys.exit(1)

        with open(state_file) as f:
            state = json.load(f)

        output_base = state_file.parent
        sweep_dir = output_base / "configs"

        configs = load_experiment_configs(sweep_dir)
        runner.run_sweep(configs, output_base, state_file)

    # 模式 2: 扫描模式
    elif args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        if not sweep_dir.exists():
            print(f"错误: 扫描目录不存在: {sweep_dir}")
            sys.exit(1)

        configs = load_experiment_configs(sweep_dir)
        if not configs:
            print(f"错误: 未找到实验配置文件在 {sweep_dir}")
            sys.exit(1)

        print(f"找到 {len(configs)} 个实验配置")
        output_base = Path(args.output)

        # 备份配置到输出目录
        config_backup = output_base / "configs"
        config_backup.mkdir(parents=True, exist_ok=True)
        for json_file in sweep_dir.glob("*.json"):
            shutil.copy(json_file, config_backup)

        runner.run_sweep(configs, output_base)

    # 模式 3: 单实验模式
    elif args.experiment and args.run_id:
        exp_file = Path(args.experiment)
        if not exp_file.exists():
            print(f"错误: 实验配置文件不存在: {exp_file}")
            sys.exit(1)

        with open(exp_file) as f:
            config = json.load(f)

        output_dir = Path(args.output) / args.run_id
        success, msg = runner.run_single_experiment(
            experiment_config=config,
            run_id=args.run_id,
            output_dir=output_dir,
            env_args=env_args,
        )

        sys.exit(0 if success else 1)

    else:
        print("错误: 必须指定以下模式之一:")
        print("  1. 单实验: --experiment + --run-id")
        print("  2. 扫描: --sweep-dir")
        print("  3. 恢复: --resume")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
