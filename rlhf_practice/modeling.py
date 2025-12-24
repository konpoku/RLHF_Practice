from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


class PolicyValueModel(nn.Module):
    """
    一个简单的「策略 + 价值」模型封装：
    - 策略：Hugging Face 的 Causal LM（如 GPT-2）
    - 价值：在 LM 的最后一层隐藏状态上接一个线性层
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, return_values=False):
        if return_values:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]  # [B, L, H]

            if attention_mask is None:
                # 没有 mask，则认为全部为有效 token
                last_indices = torch.full(
                    (hidden_states.size(0),),
                    hidden_states.size(1) - 1,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            else:
                # 每个样本最后一个非 pad 的位置
                lengths = attention_mask.sum(dim=-1)  # [B]
                last_indices = lengths - 1

            batch_indices = torch.arange(
                hidden_states.size(0), device=hidden_states.device
            )
            last_hidden = hidden_states[batch_indices, last_indices]  # [B, H]
            values = self.value_head(last_hidden).squeeze(-1)  # [B]
            return values

        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_values(self, input_ids, attention_mask=None) -> torch.Tensor:
        """
        根据输入序列（通常是 prompt）计算状态价值 V(s)。
        这里简单地取每个样本最后一个非 padding token 的隐藏状态。
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]  # [B, L, H]

        if attention_mask is None:
            # 没有 mask，则认为全部为有效 token
            last_indices = torch.full(
                (hidden_states.size(0),),
                hidden_states.size(1) - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            # 每个样本最后一个非 pad 的位置
            lengths = attention_mask.sum(dim=-1)  # [B]
            last_indices = lengths - 1

        batch_indices = torch.arange(
            hidden_states.size(0), device=hidden_states.device
        )
        last_hidden = hidden_states[batch_indices, last_indices]  # [B, H]
        values = self.value_head(last_hidden).squeeze(-1)  # [B]
        return values


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 对于 GPT-2 这类没有 pad_token 的模型，使用 eos_token 作为 pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def evaluate_sequences(
    model: PolicyValueModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    sequences: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    给定 prompt 和生成后的完整序列（prompt + response），
    计算：
    - 每个样本的 log_prob(action)（整段 reply 的 logprob 之和）
    - 每个样本的状态价值 V(s)（基于 prompt）

    返回:
        logprobs: [B]
        values:   [B]
    """
    device = next(model.parameters()).device
    prompt_input_ids = prompt_input_ids.to(device)
    prompt_attention_mask = prompt_attention_mask.to(device)
    sequences = sequences.to(device)

    # 构建整段序列的 attention mask
    attn_mask = (sequences != tokenizer.pad_token_id).long()

    # logits: 预测当前位置 token 的分布，输入为序列前 n-1 个 token
    outputs = model(
        input_ids=sequences[:, :-1],
        attention_mask=attn_mask[:, :-1],
    )
    logits = outputs.logits  # [B, L-1, V]
    log_probs = F.log_softmax(logits, dim=-1)

    # 动作为「生成的 token 序列」，即序列从位置 1 开始的 token
    next_tokens = sequences[:, 1:]  # [B, L-1]
    action_log_probs_all = log_probs.gather(
        dim=-1, index=next_tokens.unsqueeze(-1)
    ).squeeze(-1)  # [B, L-1]

    # 使用 prompt 长度区分 prompt / response，只有 response 部分计入 logprob
    prompt_lengths = prompt_attention_mask.sum(dim=-1)  # [B]
    batch_size, seq_len_minus1 = action_log_probs_all.shape
    response_mask = torch.zeros_like(action_log_probs_all, dtype=torch.bool)
    for i in range(batch_size):
        # 位置 0 对应原始序列的 token_1，因此 prompt 的最后一个 token 的预测位置是 prompt_len-1
        start = prompt_lengths[i] - 1
        if start < seq_len_minus1:
            response_mask[i, start:] = True

    # 只对 response 部分求和，得到每个样本的总 logprob
    masked_log_probs = action_log_probs_all.masked_fill(~response_mask, 0.0)
    logprobs = masked_log_probs.sum(dim=-1)  # [B]

    # 价值函数只依赖 prompt
    # 使用 forward(return_values=True) 以兼容 FSDP
    values = model(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        return_values=True
    )
    return logprobs, values

