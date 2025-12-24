"""
rlhf_practice/rl/grpo.py 的参考答案实现。
"""

from typing import Tuple

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    normalize: bool = True,
) -> torch.Tensor:
    # 假设 rewards 形状为 [B * K]
    assert rewards.numel() % group_size == 0, "rewards 长度必须能被 group_size 整除"
    batch_size = rewards.numel() // group_size

    # 1. [B * K] -> [B, K]
    rewards_group = rewards.view(batch_size, group_size)

    # 2. 组内平均 baseline: [B, 1]
    baseline = rewards_group.mean(dim=1, keepdim=True)

    # 3. 组内相对优势
    advantages_group = rewards_group - baseline  # [B, K]

    # 4. 展平回 [B * K]
    advantages = advantages_group.view(-1)

    # 5. 可选标准化
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages


def grpo_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    # 1. 不让 advantage 反向传播
    advantages_detached = advantages.detach()

    # 2. 带权重的 REINFORCE 损失：-E[A * logpi]
    loss = -(advantages_detached * logprobs).mean()
    return loss


def grpo_step(
    logprobs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = compute_group_advantages(rewards, group_size, normalize=True)
    loss = grpo_loss(logprobs, advantages)
    return loss, advantages.detach()

