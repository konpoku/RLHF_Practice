import math

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import set_seed
from torch.utils.tensorboard import SummaryWriter

from rlhf_practice.config import (
    get_default_model_config,
    get_default_data_config,
    get_default_train_config,
)
from rlhf_practice.data import load_text_dataset
from rlhf_practice.modeling import PolicyValueModel, load_tokenizer, evaluate_sequences
from rlhf_practice.reward import simple_sentiment_reward
from rlhf_practice.rl.ppo import ppo_step


def main():
    accelerator = Accelerator()
    device = accelerator.device

    set_seed(42)

    model_cfg = get_default_model_config()
    data_cfg = get_default_data_config()
    train_cfg = get_default_train_config()

    # 模型与 tokenizer
    tokenizer = load_tokenizer(model_cfg.model_name)
    model = PolicyValueModel(model_cfg.model_name)

    # 数据集
    dataset = load_text_dataset(data_cfg)

    def collate_fn(batch):
        prompts = [item["prompt"] for item in batch]
        return {"prompts": prompts}

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    unwrapped_model = accelerator.unwrap_model(model)

    model.train()

    # 仅在主进程上创建 TensorBoard 日志记录器
    writer = None
    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir="runs/ppo")

    total_steps = train_cfg.num_epochs * math.ceil(
        len(dataloader.dataset) / train_cfg.train_batch_size
    )
    global_step = 0

    for epoch in range(train_cfg.num_epochs):
        progress_bar = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}",
        )

        for batch in progress_bar:
            global_step += 1

            prompts = batch["prompts"]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=data_cfg.max_prompt_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # 生成回复（不参与反向传播）
            with torch.no_grad():
                gen_sequences = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=data_cfg.max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 取出生成的回复文本，用于奖励计算
            generated_texts = tokenizer.batch_decode(
                gen_sequences[:, input_ids.size(1) :],
                skip_special_tokens=True,
            )
            rewards = simple_sentiment_reward(generated_texts).to(device)

            # 使用当前模型计算 old_logprobs 和 values（之后会作为 PPO 的基准）
            with torch.no_grad():
                old_logprobs, values = evaluate_sequences(
                    model,
                    tokenizer,
                    prompt_input_ids=input_ids,
                    prompt_attention_mask=attention_mask,
                    sequences=gen_sequences,
                )

            # PPO 内部会多次重复计算 new_logprobs, values，这里简单做若干 epoch
            for _ in range(train_cfg.ppo_epochs):
                new_logprobs, new_values = evaluate_sequences(
                    model,
                    tokenizer,
                    prompt_input_ids=input_ids,
                    prompt_attention_mask=attention_mask,
                    sequences=gen_sequences,
                )

                total_loss, policy_loss, value_loss = ppo_step(
                    old_logprobs=old_logprobs,
                    new_logprobs=new_logprobs,
                    values=new_values,
                    rewards=rewards,
                    clip_range=train_cfg.ppo_clip_range,
                    value_coef=train_cfg.value_coef,
                )

                # 近似 KL(pi_old || pi_new) = E[logpi_old - logpi_new]
                approx_kl = torch.mean(old_logprobs - new_logprobs)

                # 若设置了 kl_coef，则在总损失中加入 KL 惩罚
                loss_with_kl = total_loss + train_cfg.kl_coef * approx_kl

                accelerator.backward(loss_with_kl)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process and global_step % train_cfg.log_every == 0:
                avg_reward = rewards.mean().item()

                # 控制台 / 进度条日志
                progress_bar.set_postfix(
                    {
                        "reward": f"{avg_reward:.3f}",
                        "policy_loss": f"{policy_loss.item():.3f}",
                        "value_loss": f"{value_loss.item():.3f}",
                        "kl": f"{approx_kl.item():.4f}",
                    }
                )

                # TensorBoard 标量日志
                if writer is not None:
                    writer.add_scalar("reward/avg", avg_reward, global_step)
                    writer.add_scalar("loss/policy", policy_loss.item(), global_step)
                    writer.add_scalar("loss/value", value_loss.item(), global_step)
                    total_no_kl = (policy_loss + train_cfg.value_coef * value_loss).item()
                    writer.add_scalar("loss/total_no_kl", total_no_kl, global_step)
                    writer.add_scalar("kl/approx", approx_kl.item(), global_step)
                    writer.add_scalar(
                        "loss/total_with_kl",
                        (total_no_kl + train_cfg.kl_coef * approx_kl.item()),
                        global_step,
                    )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
