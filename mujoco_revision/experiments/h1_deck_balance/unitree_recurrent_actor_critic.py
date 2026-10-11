"""Recurrent actor-critic used by all pretrained-policy fine-tuning methods."""

from __future__ import annotations

import torch
from torch import nn

from unitree_trainable_policy import UnitreeH12Actor, load_unitree_actor


class UnitreeRecurrentActorCritic(nn.Module):
    """Official recurrent actor plus a newly initialized common critic."""

    def __init__(self, exported_policy, initial_std: float = 0.10) -> None:
        super().__init__()
        imported = load_unitree_actor(exported_policy)
        self.actor: UnitreeH12Actor = imported
        self.critic = nn.Sequential(
            nn.Linear(47, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1)
        )
        self.log_std = nn.Parameter(torch.full((12,), float(torch.log(torch.tensor(initial_std)))))

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor.initial_state(batch_size)

    def act_step(self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], deterministic: bool = False):
        mean, next_state = self.actor.forward_step(obs, state)
        distribution = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        action = mean if deterministic else distribution.sample()
        return action, distribution.log_prob(action).sum(-1), self.critic(obs).squeeze(-1), next_state

    def evaluate_sequence(
        self, observations: torch.Tensor, actions: torch.Tensor, episode_starts: torch.Tensor,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate one time-major rollout with recurrent state resets.

        Shapes are ``observations[T, 47]``, ``actions[T, 12]`` and
        ``episode_starts[T]``.  The loop intentionally preserves temporal order;
        recurrent PPO must not shuffle individual transitions.
        """
        state = self.initial_state(1) if initial_state is None else initial_state
        log_probs, entropies, values = [], [], []
        for obs, action, episode_start in zip(observations, actions, episode_starts):
            keep = (1.0 - episode_start).reshape(1, 1, 1)
            state = (state[0] * keep, state[1] * keep)
            mean, state = self.actor.forward_step(obs.unsqueeze(0), state)
            distribution = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
            log_probs.append(distribution.log_prob(action.unsqueeze(0)).sum(-1).squeeze(0))
            entropies.append(distribution.entropy().sum(-1).squeeze(0))
            values.append(self.critic(obs.unsqueeze(0)).squeeze())
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)
