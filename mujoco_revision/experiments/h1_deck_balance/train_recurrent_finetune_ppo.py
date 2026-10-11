"""Safety-gated recurrent PPO fine-tuning from the common Unitree actor initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]

from dmppo.algorithms.ppo import compute_gae
from h1_2_official_deck_env import H12OfficialDeckEnv, UPSTREAM
from unitree_recurrent_actor_critic import UnitreeRecurrentActorCritic


POLICY = UPSTREAM / "deploy" / "pre_train" / "h1_2" / "motion.pt"
COMMON = HERE / "results" / "common_unitree_finetune_init" / "checkpoint_common.pt"
SELECTION = (303, 505)
HOLDOUT = (1101, 1303)


def load_common() -> UnitreeRecurrentActorCritic:
    model = UnitreeRecurrentActorCritic(POLICY, initial_std=0.10)
    model.load_state_dict(torch.load(COMMON, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
    return model


def evaluate(model: UnitreeRecurrentActorCritic, seeds: tuple[int, ...], duration: float,
             reward_mode: str, deck_roll_deg: float = 1.0,
             deck_pitch_deg: float = 1.0) -> dict:
    rows = []
    model.eval()
    for seed in seeds:
        env = H12OfficialDeckEnv(episode_duration=duration, randomize_phase=True, reward_mode=reward_mode,
                                deck_roll_amp_deg=deck_roll_deg, deck_pitch_amp_deg=deck_pitch_deg)
        state = model.initial_state(1)
        try:
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for length in range(1, env.max_steps + 1):
                with torch.inference_mode():
                    action, _, _, state = model.act_step(
                        torch.from_numpy(obs).unsqueeze(0), state, deterministic=True
                    )
                obs, reward, terminated, truncated, info = env.step(action.squeeze(0).numpy())
                total += reward
                if terminated or truncated:
                    break
            rows.append({"seed": seed, "elapsed_s": float(info["time"]), "return": total,
                         "success": bool(truncated and not terminated), "fell": bool(terminated)})
        finally:
            env.close()
    return {"success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_elapsed_s": float(np.mean([row["elapsed_s"] for row in rows])), "episodes": rows}


def recurrent_ppo_update(model, optimizer, batch: dict, epochs: int, clip_epsilon: float) -> dict:
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
    starts = torch.as_tensor(batch["episode_starts"], dtype=torch.float32)
    old_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
    rows = []
    model.train()
    for _ in range(epochs):
        initial_state = tuple(torch.as_tensor(item, dtype=torch.float32) for item in batch["initial_state"])
        logp, entropy, values = model.evaluate_sequence(obs, actions, starts, initial_state)
        ratio = torch.exp(logp - old_logp)
        policy_loss = -torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages,
        ).mean()
        value_loss = 0.5 * torch.square(returns - values).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        with torch.no_grad():
            log_ratio = logp - old_logp
            approx_kl = float(((ratio - 1.0) - log_ratio).mean())
        rows.append({"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                     "entropy": float(entropy.mean().detach()), "grad_norm_before_clip": grad_norm,
                     "approx_kl": approx_kl})
        if approx_kl > 0.01:
            break
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} | {"epochs": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--eval-duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--method", choices=("ppo", "ppo_com", "ppo_multi_stability"), default="ppo")
    parser.add_argument("--reward-mode", choices=("base", "base_plus_com", "base_plus_multi_stability"), default="base")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--deck-roll-deg", type=float, default=1.0)
    parser.add_argument("--deck-pitch-deg", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "results" / "recurrent_ppo_finetune_preflight")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = load_common()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)
    global_step_offset = 0
    if args.resume_checkpoint is not None:
        resumed = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        if resumed.get("method") != args.method or resumed.get("reward_mode", "base") != args.reward_mode:
            raise ValueError("resume checkpoint method/reward mode does not match requested run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        global_step_offset = int(resumed.get("global_step", 0))
    expected_mode = {"ppo": "base", "ppo_com": "base_plus_com",
                     "ppo_multi_stability": "base_plus_multi_stability"}[args.method]
    if args.reward_mode != expected_mode:
        raise ValueError(f"method {args.method} requires reward mode {expected_mode}")
    initial = {"selection": evaluate(model, SELECTION, args.eval_duration, args.reward_mode,
                                      args.deck_roll_deg, args.deck_pitch_deg),
               "holdout": evaluate(model, HOLDOUT, args.eval_duration, args.reward_mode,
                                    args.deck_roll_deg, args.deck_pitch_deg)}

    env = H12OfficialDeckEnv(episode_duration=args.eval_duration, randomize_phase=True,
                            reward_mode=args.reward_mode, deck_roll_amp_deg=args.deck_roll_deg,
                            deck_pitch_amp_deg=args.deck_pitch_deg)
    obs, _ = env.reset(seed=args.seed)
    state = model.initial_state(1)
    episode_start = 1.0
    used = updates = episodes = 0
    metrics = []
    while used < args.total_steps:
        count = min(args.rollout_steps, args.total_steps - used)
        rollout_initial_state = tuple(item.detach().clone().numpy() for item in state)
        buf = {key: [] for key in ("obs", "actions", "log_probs", "rewards", "dones", "values", "episode_starts")}
        model.eval()
        for _ in range(count):
            with torch.no_grad():
                action, logp, value, next_state = model.act_step(
                    torch.from_numpy(obs).unsqueeze(0), state, deterministic=False
                )
            action_np = action.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            for key, value_item in (("obs", obs), ("actions", action_np), ("log_probs", float(logp)),
                                    ("rewards", reward), ("dones", float(done)),
                                    ("values", float(value)), ("episode_starts", episode_start)):
                buf[key].append(value_item)
            obs, state, episode_start = next_obs, next_state, 0.0
            used += 1
            if done:
                episodes += 1
                obs, _ = env.reset(seed=args.seed + episodes)
                state = model.initial_state(1)
                episode_start = 1.0
        with torch.no_grad():
            last_value = float(model.critic(torch.from_numpy(obs).unsqueeze(0)).squeeze())
        advantages, returns = compute_gae(
            np.asarray(buf["rewards"], np.float32), np.asarray(buf["values"], np.float32),
            np.asarray(buf["dones"], np.float32), last_value, 0.99, 0.95,
        )
        batch = {key: np.asarray(value, np.float32) for key, value in buf.items()}
        batch["initial_state"] = rollout_initial_state
        batch["advantages"], batch["returns"] = advantages, returns
        row = recurrent_ppo_update(model, optimizer, batch, epochs=4, clip_epsilon=0.2)
        updates += 1; metrics.append({"step": global_step_offset + used, **row})
    env.close()

    final = {"selection": evaluate(model, SELECTION, args.eval_duration, args.reward_mode,
                                    args.deck_roll_deg, args.deck_pitch_deg),
             "holdout": evaluate(model, HOLDOUT, args.eval_duration, args.reward_mode,
                                  args.deck_roll_deg, args.deck_pitch_deg)}
    non_degrading = all(
        final[split][metric] + 1.0e-9 >= initial[split][metric]
        for split in ("selection", "holdout") for metric in ("success_rate", "mean_elapsed_s")
    )
    candidate = args.output_dir / "checkpoint_candidate.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step_offset + used, "source": str(COMMON), "method": args.method,
                "reward_mode": args.reward_mode}, candidate)
    result = {"method": args.method, "reward_mode": args.reward_mode,
              "common_initialization": str(COMMON), "additional_steps": used,
              "global_step": global_step_offset + used,
              "deck_roll_deg": args.deck_roll_deg, "deck_pitch_deg": args.deck_pitch_deg,
              "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
              "updates": updates, "episodes": episodes, "initial": initial, "final": final,
              "non_degrading_gate_passed": non_degrading, "candidate_promoted": False,
              "checkpoint_candidate": str(candidate), "last_update": metrics[-1]}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
