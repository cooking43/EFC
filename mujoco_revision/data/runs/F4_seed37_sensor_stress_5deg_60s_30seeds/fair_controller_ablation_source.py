"""Fair platform-roll prediction and double-support EFC ablations.

All four variants share the same 47-D Unitree proprioception, the same causal
100 ms second-order Taylor platform-roll prediction, the same recurrent PPO
backbone, and the same 12-D actuator interface.  The H1-2 task is standing, so
the explicit path is a double-support differential knee offset rather than the
paper's former swing-phase-gated stance-knee offset.
"""

from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments" / "h1_deck_balance"
sys.path[:0] = [str(ROOT), str(BASE), str(HERE)]

from deck_motion import get_deck_pose  # noqa: E402
from h1_2_official_deck_env import H12OfficialDeckEnv  # noqa: E402
from train_recurrent_finetune_ppo import load_common  # noqa: E402

METHODS = ("Prediction-to-Policy", "Current-State-Residual",
           "Predictive-Residual", "Oracle-Predictive-Residual")
BASE_OBSERVATION_DIM = 47
PREDICTION_DIM = 1
OBSERVATION_DIM = BASE_OBSERVATION_DIM + PREDICTION_DIM
PREDICTION_HORIZON_S = 0.1
CONTROL_DT_S = 0.02
PREDICTOR_ORDER = 2
FILTER_NEW_WEIGHT = 0.3
ROLL_TO_KNEE_GAIN = 0.1
KNEE_DIRECTION_SIGN = 1.0
RESIDUAL_LIMIT_RAD = np.deg2rad(8.0)
LEFT_KNEE_NAME = "left_knee_joint"
RIGHT_KNEE_NAME = "right_knee_joint"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation_schema() -> dict:
    return {
        "dimension": OBSERVATION_DIM,
        "blocks": [
            {"name": "unitree_proprioception", "dimension": 47,
             "source": "robot IMU, joint encoders, previous action, local gait clock",
             "causal": True},
            {"name": "causal_100ms_platform_roll_prediction", "dimension": 1,
             "fields": ["filtered_predicted_roll_rad"],
             "source": "platform roll sensor and second-order Taylor predictor",
             "causal": True},
        ],
        "actor_deck_truth": False,
    }


class PlatformRollTaylorPredictor:
    """Causal finite-difference roll predictor matching the revised method."""

    def __init__(self, horizon_s: float = PREDICTION_HORIZON_S,
                 control_dt_s: float = CONTROL_DT_S,
                 filter_new_weight: float = FILTER_NEW_WEIGHT) -> None:
        self.horizon_s = float(horizon_s)
        self.control_dt_s = float(control_dt_s)
        self.filter_new_weight = float(filter_new_weight)
        self.history_deg: deque[float] = deque(maxlen=3)
        self.filtered_prediction_deg = 0.0
        self.filtered_current_deg = 0.0

    def reset(self, roll_rad: float) -> dict[str, float]:
        roll_deg = float(np.rad2deg(roll_rad))
        self.history_deg.clear()
        self.history_deg.extend((roll_deg, roll_deg, roll_deg))
        self.filtered_prediction_deg = roll_deg
        self.filtered_current_deg = roll_deg
        return self.current()

    def step(self, roll_rad: float) -> dict[str, float]:
        self.history_deg.append(float(np.rad2deg(roll_rad)))
        return self.current()

    def current(self) -> dict[str, float]:
        if len(self.history_deg) != 3:
            raise RuntimeError("platform roll history is not initialized")
        phi_tm2, phi_tm1, phi_t = self.history_deg
        # Three-point backward difference is causal and second-order accurate;
        # it is exact for the velocity of a quadratic roll trajectory.
        velocity = (3.0 * phi_t - 4.0 * phi_tm1 + phi_tm2) / (2.0 * self.control_dt_s)
        acceleration = (phi_t - 2.0 * phi_tm1 + phi_tm2) / self.control_dt_s**2
        first = phi_t + velocity * self.horizon_s
        second = first + 0.5 * acceleration * self.horizon_s**2
        alpha = self.filter_new_weight
        self.filtered_prediction_deg = alpha * second + (1.0 - alpha) * self.filtered_prediction_deg
        self.filtered_current_deg = alpha * phi_t + (1.0 - alpha) * self.filtered_current_deg
        return {"current_roll_deg": phi_t, "roll_rate_deg_s": velocity,
                "roll_acceleration_deg_s2": acceleration,
                "persistence_roll_deg": phi_t, "first_order_roll_deg": first,
                "second_order_roll_deg": second,
                "filtered_current_roll_deg": self.filtered_current_deg,
                "filtered_prediction_roll_deg": self.filtered_prediction_deg}


def knee_residual_target(compensation_rad: float, left_knee_index: int = 3,
                         right_knee_index: int = 9) -> np.ndarray:
    """Apply a bounded differential offset to both knees in double support."""
    result = np.zeros(12, dtype=np.float64)
    bounded = float(np.clip(compensation_rad, -RESIDUAL_LIMIT_RAD, RESIDUAL_LIMIT_RAD))
    result[left_knee_index] = bounded
    result[right_knee_index] = -bounded
    return result


class FairAblationEnv(gym.Wrapper):
    def __init__(self, method: str, predictor_path: Path | None = None,
                 roll_sensor_noise_std_deg: float = 0.0,
                 roll_sensor_bias_deg: float = 0.0,
                 roll_sensor_latency_ms: int = 0,
                 **env_kwargs) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        super().__init__(H12OfficialDeckEnv(**env_kwargs))
        self.method = method
        if predictor_path is not None:
            raise ValueError("external learned predictor files are not part of corrected EFC")
        self.predictor = PlatformRollTaylorPredictor()
        self.roll_sensor_noise_std_deg = float(roll_sensor_noise_std_deg)
        self.roll_sensor_bias_deg = float(roll_sensor_bias_deg)
        self.roll_sensor_latency_ms = int(roll_sensor_latency_ms)
        if self.roll_sensor_noise_std_deg < 0.0:
            raise ValueError("roll sensor noise standard deviation must be non-negative")
        latency_samples = self.roll_sensor_latency_ms / (1000.0 * CONTROL_DT_S)
        if self.roll_sensor_latency_ms < 0 or not np.isclose(latency_samples, round(latency_samples)):
            raise ValueError("roll sensor latency must be a non-negative multiple of 20 ms")
        self.roll_sensor_latency_samples = int(round(latency_samples))
        self.roll_sensor_history_rad: deque[float] = deque(
            maxlen=self.roll_sensor_latency_samples + 1
        )
        self.roll_sensor_rng = np.random.default_rng(0)
        self.last_true_roll_rad = 0.0
        self.last_measured_roll_rad = 0.0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), dtype=np.float32)
        self.predictor_state: dict[str, float] = {}
        self.current_prediction = np.zeros(PREDICTION_DIM, dtype=np.float32)
        self.last_residual_rad = np.zeros(12, dtype=np.float64)
        self.last_applied_action = np.zeros(12, dtype=np.float64)
        self.left_knee_index = int(self.env.model.actuator(LEFT_KNEE_NAME).id)
        self.right_knee_index = int(self.env.model.actuator(RIGHT_KNEE_NAME).id)

    def _platform_roll(self, horizon_s: float = 0.0) -> float:
        return float(get_deck_pose(self.env.deck_time + horizon_s, self.env.deck_cfg)["roll"])

    def _measure_platform_roll(self, true_roll_rad: float) -> float:
        """Return the causal deck-attitude sensor value used by the predictor.

        Latency is applied before independent Gaussian angle noise and a fixed
        per-run bias.  The underlying MuJoCo deck truth is retained only for
        audit logging and the explicitly labelled Oracle upper bound.
        """
        self.roll_sensor_history_rad.append(float(true_roll_rad))
        delayed = self.roll_sensor_history_rad[0]
        noise_deg = float(self.roll_sensor_rng.normal(0.0, self.roll_sensor_noise_std_deg))
        measured = delayed + np.deg2rad(self.roll_sensor_bias_deg + noise_deg)
        self.last_true_roll_rad = float(true_roll_rad)
        self.last_measured_roll_rad = float(measured)
        return float(measured)

    def _actor_observation(self, base: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return np.concatenate((base, prediction)).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        base, info = self.env.reset(**kwargs)
        sensor_seed = int(kwargs.get("seed", 0) or 0) + 20260930
        self.roll_sensor_rng = np.random.default_rng(sensor_seed)
        true_roll = self._platform_roll()
        self.roll_sensor_history_rad.clear()
        self.roll_sensor_history_rad.extend(
            [true_roll] * (self.roll_sensor_latency_samples + 1)
        )
        measured_roll = self._measure_platform_roll(true_roll)
        self.predictor_state = self.predictor.reset(measured_roll)
        self.current_prediction = np.asarray([
            np.deg2rad(self.predictor_state["filtered_prediction_roll_deg"])
        ], dtype=np.float32)
        self.last_residual_rad[:] = 0.0
        self.last_applied_action[:] = 0.0
        return self._actor_observation(base, self.current_prediction), info

    def _residual(self) -> np.ndarray:
        if self.method == "Prediction-to-Policy":
            return np.zeros(12, dtype=np.float64)
        if self.method == "Current-State-Residual":
            signal_deg = self.predictor_state["filtered_current_roll_deg"]
        elif self.method == "Predictive-Residual":
            signal_deg = self.predictor_state["filtered_prediction_roll_deg"]
        else:
            signal_deg = float(np.rad2deg(self._platform_roll(PREDICTION_HORIZON_S)))
        compensation_deg = float(np.clip(
            KNEE_DIRECTION_SIGN * ROLL_TO_KNEE_GAIN * signal_deg, -8.0, 8.0
        ))
        return knee_residual_target(np.deg2rad(compensation_deg), self.left_knee_index,
                                    self.right_knee_index)

    def step(self, policy_action: np.ndarray):
        policy_action = np.asarray(policy_action, dtype=np.float64)
        residual = self._residual()
        # Preserve the original 12-action plant interface: a target offset in
        # radians is expressed in the same normalized coordinate as the policy.
        applied = policy_action + residual / self.env.action_scale
        base, reward, terminated, truncated, info = self.env.step(applied)
        self.last_residual_rad = residual.copy()
        self.last_applied_action = applied.copy()
        true_roll = self._platform_roll()
        measured_roll = self._measure_platform_roll(true_roll)
        self.predictor_state = self.predictor.step(measured_roll)
        self.current_prediction = np.asarray([
            np.deg2rad(self.predictor_state["filtered_prediction_roll_deg"])
        ], dtype=np.float32)
        info["ablation"] = {
            "method": self.method,
            "causal_prediction": self.current_prediction.tolist(),
            "predictor_state": dict(self.predictor_state),
            "residual_target_rad": residual.tolist(),
            "residual_max_abs_rad": float(np.max(np.abs(residual))),
            "knee_direction_sign": KNEE_DIRECTION_SIGN,
            "oracle_used": self.method == "Oracle-Predictive-Residual",
            "roll_sensor_true_rad": self.last_true_roll_rad,
            "roll_sensor_measured_rad": self.last_measured_roll_rad,
            "roll_sensor_noise_std_deg": self.roll_sensor_noise_std_deg,
            "roll_sensor_bias_deg": self.roll_sensor_bias_deg,
            "roll_sensor_latency_ms": self.roll_sensor_latency_ms,
        }
        return self._actor_observation(base, self.current_prediction), reward, terminated, truncated, info


class PredictorAugmentedActorCritic(nn.Module):
    """48-D zero-initialized expansion of the shared 47-D LSTM-PPO model."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.actor = nn.Module()
        self.actor.memory = nn.LSTM(input_size=OBSERVATION_DIM, hidden_size=64, num_layers=1)
        self.actor.actor = nn.Sequential(nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12))
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1)
        )
        self.log_std = nn.Parameter(base_model.log_std.detach().clone())
        with torch.no_grad():
            self.actor.memory.weight_ih_l0.zero_()
            self.actor.memory.weight_ih_l0[:, :47].copy_(base_model.actor.memory.weight_ih_l0)
            self.actor.memory.weight_hh_l0.copy_(base_model.actor.memory.weight_hh_l0)
            self.actor.memory.bias_ih_l0.copy_(base_model.actor.memory.bias_ih_l0)
            self.actor.memory.bias_hh_l0.copy_(base_model.actor.memory.bias_hh_l0)
            self.actor.actor.load_state_dict(base_model.actor.actor.state_dict(), strict=True)
            self.critic[0].weight.zero_()
            self.critic[0].weight[:, :47].copy_(base_model.critic[0].weight)
            self.critic[0].bias.copy_(base_model.critic[0].bias)
            self.critic[2].load_state_dict(base_model.critic[2].state_dict(), strict=True)
            self.critic[4].load_state_dict(base_model.critic[4].state_dict(), strict=True)

    def initial_state(self, batch_size: int = 1):
        parameter = next(self.parameters())
        shape = (1, batch_size, 64)
        return (torch.zeros(shape, device=parameter.device, dtype=parameter.dtype),
                torch.zeros(shape, device=parameter.device, dtype=parameter.dtype))

    def _actor_step(self, observation: torch.Tensor, state):
        output, next_state = self.actor.memory(observation.unsqueeze(0), state)
        return self.actor.actor(output.squeeze(0)), next_state

    def act_step(self, obs: torch.Tensor, state, deterministic: bool = False):
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


def load_predictor_augmented_common() -> PredictorAugmentedActorCritic:
    return PredictorAugmentedActorCritic(load_common())
