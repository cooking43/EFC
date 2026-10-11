"""Trainable PyTorch equivalent of Unitree's exported H1-2 LSTM actor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class UnitreeH12Actor(nn.Module):
    """47 -> LSTM(64) -> 32/ELU -> 12, matching ``motion.pt``."""

    def __init__(self) -> None:
        super().__init__()
        self.memory = nn.LSTM(input_size=47, hidden_size=64, num_layers=1)
        self.actor = nn.Sequential(nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12))

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (1, batch_size, 64)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)

    def forward_step(
        self, observation: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        output, next_state = self.memory(observation.unsqueeze(0), state)
        return self.actor(output.squeeze(0)), next_state

    def forward(
        self, observations: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process a time-major sequence shaped ``[T, B, 47]``."""
        if state is None:
            state = self.initial_state(observations.shape[1], device=observations.device, dtype=observations.dtype)
        output, next_state = self.memory(observations, state)
        return self.actor(output), next_state


def load_unitree_actor(exported_policy: str | Path) -> UnitreeH12Actor:
    exported = torch.jit.load(str(exported_policy), map_location="cpu")
    model = UnitreeH12Actor()
    source = {key: value for key, value in exported.state_dict().items()
              if key not in {"hidden_state", "cell_state"}}
    expected = model.state_dict()
    if set(source) != set(expected):
        raise RuntimeError(f"exported actor keys differ: missing={set(expected)-set(source)}, extra={set(source)-set(expected)}")
    model.load_state_dict(source, strict=True)
    return model
