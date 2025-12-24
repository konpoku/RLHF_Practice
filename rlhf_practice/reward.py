from typing import List

import torch


POSITIVE_WORDS = [
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "awesome",
    "fantastic",
    "love",
    "like",
]

NEGATIVE_WORDS = [
    "bad",
    "terrible",
    "awful",
    "horrible",
    "hate",
    "dislike",
    "worst",
    "boring",
]


def simple_sentiment_reward(texts: List[str]) -> torch.Tensor:
    """
    一个非常简单的基于规则的奖励函数：
    - 每出现一次正向词 +1
    - 每出现一次负向词 -1
    最终 reward = 正向词计数 - 负向词计数
    """
    rewards = []
    for t in texts:
        lower = t.lower()
        pos = sum(lower.count(w) for w in POSITIVE_WORDS)
        neg = sum(lower.count(w) for w in NEGATIVE_WORDS)
        rewards.append(float(pos - neg))
    return torch.tensor(rewards, dtype=torch.float32)

