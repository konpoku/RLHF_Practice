"""
rlhf_practice/rl/ppo.py 的参考答案实现。
你可以对照这个文件理解 PPO 的关键公式。
"""

from typing import Tuple

import torch


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    # 1. 从计算图中分离 value，避免梯度流入策略
    values_detached = values.detach()

    # 2. 简单的一步 advantage: A = R - V(s)
    advantages = rewards - values_detached

    # 3. 可选标准化，有助于训练稳定
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages


def ppo_clip_loss(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    # 1. 概率比 r_t
    ratio = torch.exp(new_logprobs - old_logprobs)

    # 2. 未裁剪的目标
    unclipped = ratio * advantages

    # 3. 裁剪后的目标
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped = clipped_ratio * advantages

    # 4. 对每个样本取较小值，再取负均值
    loss = -torch.mean(torch.min(unclipped, clipped))
    return loss


def ppo_step(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    clip_range: float,
    value_coef: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    advantages = compute_advantages(rewards, values, normalize=True)
    policy_loss = ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip_range)
    value_loss = torch.mean((values - rewards) ** 2)
    total_loss = policy_loss + value_coef * value_loss
    return total_loss, policy_loss.detach(), value_loss.detach()

