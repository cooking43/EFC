from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    policy_anchor_coef: float = 0.0
    auxiliary_action_coef: float = 0.0
    separate_actor_critic_grad_clip: bool = False


def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def ppo_update(model, optimizer, batch: dict[str, np.ndarray], cfg: PPOConfig, device: torch.device):
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    value_target_mean = float(batch.get("value_target_mean", 0.0))
    value_target_std = float(batch.get("value_target_std", 1.0))
    normalized_returns = (returns - value_target_mean) / value_target_std
    old_values = batch["values"]
    reference_actions = torch.as_tensor(batch["reference_actions"], dtype=torch.float32, device=device) if "reference_actions" in batch else None
    auxiliary_targets = torch.as_tensor(batch["auxiliary_action_targets"], dtype=torch.float32, device=device) if "auxiliary_action_targets" in batch else None
    auxiliary_masks = torch.as_tensor(batch["auxiliary_action_masks"], dtype=torch.float32, device=device) if "auxiliary_action_masks" in batch else None
    policy_mask = torch.as_tensor(batch["policy_masks"], dtype=torch.float32, device=device) if "policy_masks" in batch else None
    action_dimension_mask = torch.as_tensor(batch["action_dimension_mask"], dtype=torch.float32, device=device) if "action_dimension_mask" in batch else None

    advantage_sample = advantages[policy_mask > 0.5] if policy_mask is not None else advantages
    if advantage_sample.numel() == 0:
        raise ValueError("policy_masks selects no actor samples")
    advantages = (advantages - advantage_sample.mean()) / (advantage_sample.std(unbiased=False) + 1e-8)
    n = obs.shape[0]
    metrics = []
    early_stopped = False
    epochs_completed = 0
    for _ in range(cfg.update_epochs):
        indices = torch.randperm(n, device=device)
        for start in range(0, n, cfg.minibatch_size):
            mb = indices[start : start + cfg.minibatch_size]
            new_log_probs, entropy, values = model.evaluate_actions(obs[mb], actions[mb], action_dimension_mask)
            log_ratio = new_log_probs - old_log_probs[mb]
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages[mb]
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages[mb]
            actor_weight = policy_mask[mb] if policy_mask is not None else torch.ones_like(unclipped)
            actor_count = actor_weight.sum().clamp_min(1.0)
            policy_loss = -(torch.min(unclipped, clipped) * actor_weight).sum() / actor_count
            value_loss = 0.5 * torch.square(normalized_returns[mb] - values).mean()
            entropy_loss = (entropy * actor_weight).sum() / actor_count
            anchor_loss = torch.zeros((),device=device)
            auxiliary_action_loss = torch.zeros((),device=device)
            if reference_actions is not None and cfg.policy_anchor_coef > 0.0:
                deterministic_actions=torch.tanh(model.actor(obs[mb]))
                anchor_loss=torch.square(deterministic_actions-reference_actions[mb]).mean()
            if auxiliary_targets is not None and auxiliary_masks is not None and cfg.auxiliary_action_coef > 0.0:
                deterministic_actions=torch.tanh(model.actor(obs[mb]))
                mask=auxiliary_masks[mb]
                auxiliary_action_loss=(torch.square(deterministic_actions-auxiliary_targets[mb])*mask).sum()/mask.sum().clamp_min(1.)
            loss = (policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss
                    + cfg.policy_anchor_coef * anchor_loss + cfg.auxiliary_action_coef * auxiliary_action_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.separate_actor_critic_grad_clip:
                clip_grad_norm_(list(model.actor.parameters()) + [model.log_std], cfg.max_grad_norm)
                clip_grad_norm_(model.critic.parameters(), cfg.max_grad_norm)
            else:
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Non-negative second-order approximation used by PPO diagnostics.
            # The signed mean of old_logp-new_logp can cancel across samples and
            # substantially under-report a destructive policy update.
            approx_kl = ((((ratio - 1.0) - log_ratio) * actor_weight).sum() / actor_count).detach().item()
            clip_fraction = ((((ratio - 1.0).abs() > cfg.clip_epsilon).float() * actor_weight).sum() / actor_count).detach().item()
            metrics.append(
                {
                    "actor_loss": float(policy_loss.detach().item()),
                    "critic_loss": float(value_loss.detach().item()),
                    "policy_entropy": float(entropy_loss.detach().item()),
                    "approx_kl": float(approx_kl),
                    "clip_fraction": float(clip_fraction),
                    "policy_anchor_loss": float(anchor_loss.detach().item()),
                    "auxiliary_action_loss": float(auxiliary_action_loss.detach().item()),
                }
            )
        epochs_completed += 1
        if cfg.target_kl is not None and np.mean([m["approx_kl"] for m in metrics[-max(1, math.ceil(n / cfg.minibatch_size)):]]) > cfg.target_kl:
            early_stopped = True
            break
    out = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
    out["explained_variance"] = explained_variance(old_values, batch["returns"])
    out["epochs_completed"] = float(epochs_completed)
    out["kl_early_stop"] = float(early_stopped)
    return out
