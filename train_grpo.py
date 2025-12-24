import math

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
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
from rlhf_practice.rl.grpo import grpo_step


def main():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        use_orig_params=True,
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    device = accelerator.device

    set_seed(42)

    model_cfg = get_default_model_config()
    data_cfg = get_default_data_config()
    train_cfg = get_default_train_config()

    tokenizer = load_tokenizer(model_cfg.model_name)
    model = PolicyValueModel(model_cfg.model_name)
    
    # GRPO 算法中只使用策略网络计算 logprobs，不需要 value head
    # 为了避免 DDP 报错（因为 value_head 参数没有参与 loss 计算），我们需要冻结它
    for param in model.value_head.parameters():
        param.requires_grad = False

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

    # Debug info
    print(f"Distributed Type: {accelerator.distributed_type}")
    print(f"Model Type: {type(model)}")
    
    model.train()

    writer = None
    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir="runs/grpo")

    total_steps = train_cfg.num_epochs * math.ceil(
        len(dataloader.dataset) / train_cfg.train_batch_size
    )
    global_step = 0

    group_size = train_cfg.grpo_group_size

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

            batch_size = input_ids.size(0)

            # 为 GRPO 准备：每个 prompt 复制 group_size 份
            expanded_input_ids = input_ids.repeat_interleave(group_size, dim=0)
            expanded_attention_mask = attention_mask.repeat_interleave(
                group_size, dim=0
            )

            # 对每个 prompt 生成 group_size 个回复
            with torch.no_grad():
                # 如果是 FSDP，需要 summon_full_params
                if accelerator.distributed_type == "FSDP":
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    context = FSDP.summon_full_params(model, writeback=False, rank0_only=False)
                else:
                    from contextlib import nullcontext
                    context = nullcontext()

                with context:
                    gen_sequences = unwrapped_model.generate(
                        input_ids=expanded_input_ids,
                        attention_mask=expanded_attention_mask,
                        max_new_tokens=data_cfg.max_new_tokens,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            # 生成文本和奖励（展平为 [B * K]）
            generated_texts = tokenizer.batch_decode(
                gen_sequences[:, input_ids.size(1) :],
                skip_special_tokens=True,
            )
            rewards = simple_sentiment_reward(generated_texts).to(device)

            # 计算 logprobs（不需要 value）
            logprobs, _ = evaluate_sequences(
                model,
                tokenizer,
                prompt_input_ids=expanded_input_ids,
                prompt_attention_mask=expanded_attention_mask,
                sequences=gen_sequences,
            )

            loss, advantages = grpo_step(
                logprobs=logprobs,
                rewards=rewards,
                group_size=group_size,
            )

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.is_local_main_process and global_step % train_cfg.log_every == 0:
                avg_reward = rewards.mean().item()
                adv_mean = advantages.mean().item()

                progress_bar.set_postfix(
                    {
                        "reward": f"{avg_reward:.3f}",
                        "loss": f"{loss.item():.3f}",
                        "adv_mean": f"{adv_mean:.3f}",
                    }
                )

                if writer is not None:
                    writer.add_scalar("reward/avg", avg_reward, global_step)
                    writer.add_scalar("loss/grpo", loss.item(), global_step)
                    writer.add_scalar("advantage/mean", adv_mean, global_step)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
