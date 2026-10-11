"""Causal robot-only observation expansion for State-Enhanced-LSTM.

The added channels are computed exclusively from the public 47-D Unitree
proprioceptive observation and its one-control-step history.  They never read
MuJoCo contacts, centre of mass, or deck state.  Keeping the schema executable
prevents the reported dimension from becoming a hand-written label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments" / "h1_deck_balance"
sys.path[:0] = [str(ROOT), str(BASE)]

from h1_2_official_deck_env import H12OfficialDeckEnv  # noqa: E402
from train_recurrent_finetune_ppo import load_common  # noqa: E402


BASE_OBSERVATION_DIM = 47
CONTROL_PERIOD_S = 0.02


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    dimension: int
    source: str
    causal: bool = True


BASE_SCHEMA = FeatureBlock(
    "unitree_proprioception", BASE_OBSERVATION_DIM,
    "robot IMU, joint encoders, previous action, and local gait clock",
)
ENHANCED_BLOCKS = (
    FeatureBlock("imu_angular_velocity_increment", 3, "current and previous robot IMU gyroscope"),
    FeatureBlock("gravity_vector_increment", 3, "current and previous robot IMU attitude/gravity vector"),
    FeatureBlock("joint_target_tracking_residual", 12, "joint encoders and previous commanded action"),
)
OBSERVATION_DIM = BASE_SCHEMA.dimension + sum(block.dimension for block in ENHANCED_BLOCKS)


def schema_manifest() -> dict:
    blocks = (BASE_SCHEMA,) + ENHANCED_BLOCKS
    return {
        "name": "State-Enhanced-LSTM",
        "dimension": sum(block.dimension for block in blocks),
        "blocks": [
            {"name": block.name, "dimension": block.dimension,
             "source": block.source, "causal": block.causal}
            for block in blocks
        ],
        "forbidden_actor_sources": ["MuJoCo contact truth", "MuJoCo CoM truth", "deck truth"],
        "uses_forbidden_actor_sources": False,
    }


class CausalStateEnhancer:
    """Convert a 47-D observation stream into the executable enhanced schema."""

    def __init__(self, action_scale: float = 0.25) -> None:
        self.action_scale = float(action_scale)
        self.previous_base: np.ndarray | None = None

    def reset(self, base_observation: np.ndarray) -> np.ndarray:
        base = self._validate(base_observation)
        self.previous_base = base.copy()
        return self._compose(base, np.zeros(3), np.zeros(3))

    def step(self, base_observation: np.ndarray) -> np.ndarray:
        base = self._validate(base_observation)
        if self.previous_base is None:
            return self.reset(base)
        gyro_increment = base[0:3] - self.previous_base[0:3]
        gravity_increment = base[3:6] - self.previous_base[3:6]
        enhanced = self._compose(base, gyro_increment, gravity_increment)
        self.previous_base = base.copy()
        return enhanced

    @staticmethod
    def _validate(base_observation: np.ndarray) -> np.ndarray:
        base = np.asarray(base_observation, dtype=np.float32)
        if base.shape != (BASE_OBSERVATION_DIM,):
            raise ValueError(f"expected ({BASE_OBSERVATION_DIM},), got {base.shape}")
        return base

    def _compose(self, base: np.ndarray, gyro_increment: np.ndarray,
                 gravity_increment: np.ndarray) -> np.ndarray:
        joint_offset = base[9:21]
        previous_action = base[33:45]
        tracking_residual = joint_offset - self.action_scale * previous_action
        result = np.concatenate((base, gyro_increment, gravity_increment, tracking_residual))
        if result.shape != (OBSERVATION_DIM,):
            raise RuntimeError("observation does not match executable schema")
        return result.astype(np.float32, copy=False)


class StateEnhancedH12Env(gym.Wrapper):
    """Observation-only wrapper; plant dynamics, reward, and action stay unchanged."""

    def __init__(self, **env_kwargs) -> None:
        env = H12OfficialDeckEnv(**env_kwargs)
        super().__init__(env)
        self.enhancer = CausalStateEnhancer(action_scale=env.action_scale)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (OBSERVATION_DIM,), dtype=np.float32
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.enhancer.reset(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.enhancer.step(observation), reward, terminated, truncated, info


class StateEnhancedActorCritic(nn.Module):
    """65-D expansion of the common 47-D actor-critic initialization.

    All weights connected to the 18 added channels start at zero.  Therefore,
    the expanded policy is exactly equal to the 47-D common policy at creation,
    while PPO is free to learn whether the named channels are useful.
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        base_actor = base_model.actor
        self.actor = nn.Module()
        self.actor.memory = nn.LSTM(input_size=OBSERVATION_DIM, hidden_size=64, num_layers=1)
        self.actor.actor = nn.Sequential(nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12))
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1)
        )
        self.log_std = nn.Parameter(base_model.log_std.detach().clone())
        with torch.no_grad():
            self.actor.memory.weight_ih_l0.zero_()
            self.actor.memory.weight_ih_l0[:, :BASE_OBSERVATION_DIM].copy_(
                base_actor.memory.weight_ih_l0
            )
            self.actor.memory.weight_hh_l0.copy_(base_actor.memory.weight_hh_l0)
            self.actor.memory.bias_ih_l0.copy_(base_actor.memory.bias_ih_l0)
            self.actor.memory.bias_hh_l0.copy_(base_actor.memory.bias_hh_l0)
            self.actor.actor.load_state_dict(base_actor.actor.state_dict(), strict=True)
            self.critic[0].weight.zero_()
            self.critic[0].weight[:, :BASE_OBSERVATION_DIM].copy_(base_model.critic[0].weight)
            self.critic[0].bias.copy_(base_model.critic[0].bias)
            self.critic[2].load_state_dict(base_model.critic[2].state_dict(), strict=True)
            self.critic[4].load_state_dict(base_model.critic[4].state_dict(), strict=True)

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        parameter = next(self.parameters())
        shape = (1, batch_size, 64)
        return (torch.zeros(shape, device=parameter.device, dtype=parameter.dtype),
                torch.zeros(shape, device=parameter.device, dtype=parameter.dtype))

    def _actor_step(self, observation: torch.Tensor, state):
        output, next_state = self.actor.memory(observation.unsqueeze(0), state)
        return self.actor.actor(output.squeeze(0)), next_state

    def act_step(self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor],
                 deterministic: bool = False):
        mean, next_state = self._actor_step(obs, state)
        distribution = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        action = mean if deterministic else distribution.sample()
        return action, distribution.log_prob(action).sum(-1), self.critic(obs).squeeze(-1), next_state

    def evaluate_sequence(self, observations: torch.Tensor, actions: torch.Tensor,
                          episode_starts: torch.Tensor, initial_state=None):
        state = self.initial_state(1) if initial_state is None else initial_state
        log_probs, entropies, values = [], [], []
        for obs, action, episode_start in zip(observations, actions, episode_starts):
            keep = (1.0 - episode_start).reshape(1, 1, 1)
            state = (state[0] * keep, state[1] * keep)
            mean, state = self._actor_step(obs.unsqueeze(0), state)
            distribution = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
            log_probs.append(distribution.log_prob(action.unsqueeze(0)).sum(-1).squeeze(0))
            entropies.append(distribution.entropy().sum(-1).squeeze(0))
            values.append(self.critic(obs.unsqueeze(0)).squeeze())
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


def load_state_enhanced_common() -> StateEnhancedActorCritic:
    return StateEnhancedActorCritic(load_common())

