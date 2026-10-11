"""Replay a saved H1-2 PPO checkpoint with a window, video, or headless test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]

from h1_2_official_deck_env import H12OfficialDeckEnv, UPSTREAM  # noqa: E402
from unitree_recurrent_actor_critic import UnitreeRecurrentActorCritic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints/frozen_ppo_7168_seed73.pt")
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--roll-deg", type=float, default=1.0)
    parser.add_argument("--pitch-deg", type=float, default=1.0)
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo window")
    parser.add_argument("--video", type=Path, help="Write an MP4 without opening a window")
    args = parser.parse_args()
    if args.video and args.video.exists():
        raise FileExistsError(f"Refusing to overwrite {args.video}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    policy = UPSTREAM / "deploy/pre_train/h1_2/motion.pt"
    model = UnitreeRecurrentActorCritic(policy, initial_std=0.10)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    env = H12OfficialDeckEnv(episode_duration=args.duration, randomize_phase=True,
                            reward_mode="base", deck_roll_amp_deg=args.roll_deg,
                            deck_pitch_amp_deg=args.pitch_deg)
    obs, _ = env.reset(seed=args.seed)
    state = model.initial_state(1)
    viewer = renderer = writer = None
    if args.viewer:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(env.model, env.data)
    if args.video:
        import imageio.v2 as imageio
        import mujoco
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        writer = imageio.get_writer(args.video, fps=25, macro_block_size=1)

    start = time.perf_counter()
    reward_sum = 0.0
    terminated = truncated = False
    steps = 0
    try:
        for steps in range(1, env.max_steps + 1):
            if viewer is not None and not viewer.is_running():
                break
            with torch.inference_mode():
                action, _, _, state = model.act_step(
                    torch.from_numpy(obs).unsqueeze(0), state, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
            reward_sum += float(reward)
            if viewer is not None:
                viewer.sync()
                target = start + steps * env.model.opt.timestep * env.decimation
                time.sleep(max(0.0, target - time.perf_counter()))
            if renderer is not None and steps % 2 == 0:
                renderer.update_scene(env.data, camera=-1)
                writer.append_data(renderer.render())
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()
        env.close()
    if not np.isfinite(obs).all():
        raise RuntimeError("Nonfinite observation during playback")
    print(json.dumps({
        "checkpoint": str(args.checkpoint), "seed": args.seed,
        "roll_deg": args.roll_deg, "pitch_deg": args.pitch_deg,
        "steps": steps, "elapsed_sim_s": steps * env.model.opt.timestep * env.decimation,
        "return": reward_sum, "fell": bool(terminated),
        "success": bool(truncated and not terminated),
        "video": str(args.video) if args.video else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
