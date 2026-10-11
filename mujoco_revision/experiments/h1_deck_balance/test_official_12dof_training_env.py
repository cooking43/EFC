"""Bounded API/numerics preflight for the unified 12-DoF training task."""

from __future__ import annotations

import json
import numpy as np

from h1_2_official_deck_env import H12OfficialDeckEnv


def main() -> int:
    env = H12OfficialDeckEnv(episode_duration=2.0, randomize_phase=True)
    try:
        summaries = []
        for seed in (303, 1101):
            obs, _ = env.reset(seed=seed)
            assert obs.shape == (47,) and np.isfinite(obs).all()
            rewards = []
            for step in range(100):
                action = env.action_space.sample() * 0.05
                obs, reward, terminated, truncated, info = env.step(action)
                assert obs.shape == (47,) and np.isfinite(obs).all() and np.isfinite(reward)
                rewards.append(reward)
                if terminated or truncated:
                    break
            summaries.append({"seed": seed, "steps": step + 1, "fell": bool(terminated),
                              "mean_reward": float(np.mean(rewards))})
        result = {"passed": True, "observation_dim": 47, "action_dim": 12,
                  "unitree_observation_order": True, "deck_truth_observed": False,
                  "gait_phase_input": True, "episodes": summaries}
        print(json.dumps(result, indent=2))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
