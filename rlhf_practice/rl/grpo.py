"""
GRPO (Group Relative Policy Optimization) 关键公式实现。

思想：对每个 prompt 采样一组 (group_size) 个回复，
用组内的平均奖励作为「基线」，计算相对优势，然后进行策略梯度更新。
"""

from typing import Tuple

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    计算组相对优势（每个样本一个 advantage）。

    输入:
        rewards: [B * group_size]
            假设原始 batch_size = B，
            对每个 prompt 采样 group_size 个回复，
            则 rewards 以「先 prompt 再 group」的方式展开。
        group_size: 组大小 K

    步骤:
        1. 把 rewards reshape 成 [B, K]
        2. 对每行求均值作为基线 baseline
        3. advantages = rewards - baseline
        4. 再展开回 [B * K]
        5. (可选) 对 advantages 做标准化
    """
    # TODO 1: 根据 group_size reshape 成 [B, K]
    # TODO 2: 计算组内平均 baseline
    # TODO 3: 计算 advantages = rewards_group - baseline
    # TODO 4: 展开回 [B * K]
    # TODO 5: (可选) 如果 normalize=True，对 advantages 标准化
    raise NotImplementedError("请在 compute_group_advantages 中补全代码")


def grpo_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """
    GRPO 的策略梯度损失：

        L = - E[ advantages.detach() * logpi(a|s) ]

    这里不涉及价值网络，完全是基于组内相对优势的 REINFORCE 形式。
    """
    # TODO 1: 确保 advantages 不反向传播梯度（使用 detach）
    # TODO 2: 按照上面的公式实现带权重的策略梯度损失（求均值）
    raise NotImplementedError("请在 grpo_loss 中补全代码")


def grpo_step(
    logprobs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    一个完整的 GRPO step 损失计算：
    - 根据 rewards 计算组内 advantage
    - 用 advantage 作为权重，对 logprobs 做策略梯度
    """
    advantages = compute_group_advantages(rewards, group_size, normalize=True)
    loss = grpo_loss(logprobs, advantages)
    return loss, advantages.detach()

