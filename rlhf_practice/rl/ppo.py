"""
PPO (Proximal Policy Optimization) 关键公式实现。

本文件中的函数被训练脚本调用，你需要根据注释补全部分代码。
完成后即可在 train_ppo.py 中运行 PPO 训练。
"""

from typing import Tuple

import torch


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    计算 advantage（优势函数）。

    这里我们使用最简单的一步策略梯度形式：
        A = R - V(s)

    参数:
        rewards: [B]，每个样本的标量奖励
        values:  [B]，价值网络预测的 V(s)
        normalize: 是否对 advantage 做标准化

    返回:
        advantages: [B]
    """
    # TODO 1: ��� values 从计算图中分离，避免梯度同时流向 value 和 policy
    # TODO 2: 计算 advantages = rewards - values_detached
    # TODO 3: (可选) 如果 normalize=True，则对 advantages 做标准化：
    #         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #raise NotImplementedError("请在 compute_advantages 中补全代码")
    values_detached = values.detach()
    advantages = rewards - values_detached
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    assert rewards.shape == advantages.shape
    assert values.shape == advantages.shape
    return advantages


def ppo_clip_loss(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """
    计算 PPO 的裁剪策略损失（只包含 policy 部分，不含 value loss）。

    标准的 PPO-Clip 目标函数为：
        L_clip = - E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
    其中:
        r_t = exp(new_logprob - old_logprob)
        eps = clip_range
    """
    # TODO 1: 计算 r_t = exp(new_logprobs - old_logprobs)
    # TODO 2: 计算 unclipped = r_t * advantages
    # TODO 3: 计算 clipped_ratio = r_t.clamp(1 - clip_range, 1 + clip_range)
    # TODO 4: 计算 clipped = clipped_ratio * advantages
    # TODO 5: 取 element-wise 最小值，然后取负均值作为损失
    #raise NotImplementedError("请在 ppo_clip_loss 中补全代码")
    r_t = torch.exp(new_logprobs - old_logprobs)
    unclipped = r_t * advantages
    clipped_ratio = r_t.clamp(1 - clip_range, 1 + clip_range)
    clipped = clipped_ratio * advantages
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
    """
    一个完整的 PPO step 损失计算：
    - policy_loss: 来自 ppo_clip_loss
    - value_loss:  简单的 MSE(R, V)
    - total_loss:  policy_loss + value_coef * value_loss

    训练脚本会把 total_loss 反向传播。
    """
    advantages = compute_advantages(rewards, values, normalize=True)
    policy_loss = ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip_range)
    value_loss = torch.mean((values - rewards) ** 2)
    total_loss = policy_loss + value_coef * value_loss
    return total_loss, policy_loss.detach(), value_loss.detach()

