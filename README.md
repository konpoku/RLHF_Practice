# RLHF-Practice: 使用大语言模型的 PPO / GRPO 实践项目

本项目是一个**面向初学者**的强化学习实践项目，目标是帮助你在真实的大语言模型（LLM）和 Hugging Face 数据集上，亲手实现并运行：

- 基于策略梯度的 **PPO (Proximal Policy Optimization)** 算法
- 基于组相对优势的 **GRPO (Group Relative Policy Optimization)** 算法

你只需要补全少量关键的 RL 公式代码，就可以在 4 卡 3090 环境上运行多卡训练，观察模型在简单文本奖励上的优化过程。

---

## 1. 环境准备

### 1.1 创建虚拟环境并安装依赖

```bash
cd RLHF-Practice
python -m venv .venv
source .venv/bin/activate   # Windows 使用 .venv\\Scripts\\activate

pip install -r requirements.txt
```

> 说明：依赖主要有 `torch`, `transformers`, `datasets`, `accelerate`, `tqdm`。

### 1.2 多卡配置（4 卡 3090）

本项目使用 Hugging Face 的 `accelerate` 进行多 GPU 数据并行训练。

第一次使用前先生成配置文件：

```bash
accelerate config
```

按提示选择：
- 设备数：`4`
- 分布式后端：推荐 `nccl`
- 其他选项保持默认或根据自己习惯调整

之后即可使用：

```bash
accelerate launch --num_processes 4 train_ppo.py
# 或
accelerate launch --num_processes 4 train_grpo.py
```

也可以只用单卡运行（便于调试）：

```bash
python train_ppo.py
python train_grpo.py
```

---

## 2. 项目结构

```text
RLHF-Practice/
  README.md
  requirements.txt

  train_ppo.py          # PPO 训练入口脚本
  train_grpo.py         # GRPO 训练入口脚本

  rlhf_practice/
    __init__.py
    config.py           # 一些默认配置（模型、数据、训练参数）
    data.py             # 从 Hugging Face 加载数据集并做简单封装
    modeling.py         # 模型封装：LLM + Value Head
    reward.py           # 简单的基于规则的奖励函数
    rl/
      __init__.py
      ppo.py            # PPO 关键公式（带代码填空）
      grpo.py           # GRPO 关键公式（带代码填空）

  answer/
    ppo_answer.py       # PPO 填空参考答案
    grpo_answer.py      # GRPO 填空参考答案

  runs/
    ...                 # 训练过程中自动生成的 TensorBoard 日志目录
```

---

## 3. 使用的模型和数据集

- 语言模型：默认使用 Hugging Face 上的 `gpt2`（体积小、易于在 1~4 张 3090 上跑通）。
  - 你可以在 `rlhf_practice/config.py` 中修改为其他 Causal LM（例如 `gpt2-medium`、`Qwen/Qwen2-0.5B-Instruct` 等），但显存需求会增加。
- 数据集：默认使用 `stanfordnlp/imdb`（电影评论），通过 Hugging Face `datasets` 自动下载。
  - 我们把每条评论当作「输入文本」，让模型生成一个更正向、更积极的版本。

奖励是一个**简单的规则函数**（在 `rlhf_practice/reward.py` 中实现）：

- 输出文本中出现更多「正向词」（good, great, excellent, ...）会得到更高奖励
- 出现「负向词」（bad, terrible, awful, ...）会扣分

这不是严格的 RLHF 生产环境方案，但足以帮助你理解：**「模型输出 → 奖励评估 → RL 更新」** 这条完整链路。

---

## 4. 你需要完成的内容（代码填空）

主要需要修改的文件有两个：

1. `rlhf_practice/rl/ppo.py`
2. `rlhf_practice/rl/grpo.py`

里面用 `TODO:` 标出了需要你补全的部分，主要是：

- 如何根据 `rewards` 和 `values` 计算 **优势函数 advantage**
- PPO 中 **裁剪后的目标函数**（clip objective）
- GRPO 中 **组内归一化优势** 和对应的策略梯度损失

> 如果你不熟悉公式，可以先看同目录下的注释，或者参考 `answer/ppo_answer.py` 和 `answer/grpo_answer.py`。

填完以后，确保这两个文件中不再有 `NotImplementedError`，然后就可以运行训练脚本：

```bash
# 单卡 Debug
python train_ppo.py
python train_grpo.py

# 多卡（例如 4 卡 3090）
accelerate launch --num_processes 4 train_ppo.py
accelerate launch --num_processes 4 train_grpo.py
```

---

## 5. 训练日志与可视化（TensorBoard）

为了方便你观察训练过程，本项目已经集成了 TensorBoard 日志：

- PPO 训练日志目录：`runs/ppo`
- GRPO 训练日志目录：`runs/grpo`

启动训练（例如 PPO）后，在另一个终端中运行：

```bash
tensorboard --logdir runs
```

然后在浏览器中打开提示的地址（通常是 `http://localhost:6006`），即可看到：

- 曲线：
  - `reward/avg`：平均奖励随 step 的变化
  - PPO：`loss/policy`、`loss/value`、`loss/total_no_kl`、`loss/total_with_kl`、`kl/approx`
  - GRPO：`loss/grpo`、`advantage/mean`
- 你可以对比不同实验设置（例如不同 `clip_range`、`group_size`）的曲线差异。

---

## 6. 建议的对比实验与 KL 惩罚

### 6.1 PPO 对比实验

在 `rlhf_practice/config.py` 中有几个关键超参数：

- `ppo_clip_range`：PPO 裁剪范围（默认为 0.2）
- `kl_coef`：KL 惩罚系数（默认为 0.0，即不启用 KL 惩罚）

你可以设计如下几组实验（其他参数保持不变）：

1. **基线 PPO（无 KL）**  
   - `ppo_clip_range = 0.2`  
   - `kl_coef = 0.0`  
   观察 `reward/avg`、`loss/policy` 和 `kl/approx` 曲线。

2. **加入弱 KL 惩罚**  
   - `ppo_clip_range = 0.2`  
   - `kl_coef = 0.01`（或 0.005）  
   对比与基线实验的 `reward/avg` 上升速度和 `kl/approx` 的变化。

3. **更强的 KL 惩罚**  
   - `ppo_clip_range = 0.2`  
   - `kl_coef = 0.05`（或更大）  
   观察是否出现「更新变慢但更稳定」的现象，或者奖励难以提升。

直观理解：  
`kl_coef` 越大，策略被约束「不能偏离旧策略太远」，更新更保守；  
`kl_coef` 越小或为 0，策略可以更激进，可能更快提升奖励，也可能更不稳定。

### 6.2 PPO vs GRPO

在同样的训练步数和数据设置下：

- 分别运行 `train_ppo.py` 和 `train_grpo.py`
- 在 TensorBoard 中对比：
  - `reward/avg` 的变化趋势；
  - PPO 的 `kl/approx` 曲线（策略变化幅度）；
  - GRPO 的 `advantage/mean`（是否接近 0）。

结合 `EXPERIMENT_TASK.md` 中的对比任务，你可以给出自己对两种方法的直观评价和偏好。

---

## 7. 推荐的学习路径

1. 先通读一遍本仓库的代码结构和注释，理解大致流程：
   - 加载数据 → 文本编码 → 模型生成回复 → 奖励打分 → RL 更新
2. 从 `rlhf_practice/rl/ppo.py` 开始，尝试根据注释补全代码。
3. 能成功跑通 PPO 训练（哪怕只训练几个 step），观察 loss 和奖励的变化。
4. 再阅读并补全 `rlhf_practice/rl/grpo.py`，体会「按组归一化优势」的想法。
5. 对照 `answer/` 下的参考实现，查漏补缺。

如果你希望在理解上更进一步，可以尝试：

- 把奖励函数改成你自己设计的规则；
- 换成别的 Hugging Face Causal LM 或中文数据集；
- 在训练脚本里增加日志记录和指标可视化。

---

## 6. 常见问题

- **Q: 显存不够怎么办？**  
  A: 可以在 `rlhf_practice/config.py` 里调小：
  - `train_batch_size`
  - `max_prompt_length`
  - `max_new_tokens`
  - 或者换成更小的模型（例如 `sshleifer/tiny-gpt2`）。

- **Q: 一定要 4 卡才能跑吗？**  
  A: 不是。只用 1 张 3090 甚至更小显存的显卡也可以跑，只是 batch size 和模型规模需要更小一些。4 卡的配置只是为了展示如何用 `accelerate` 做多卡训练。

---

祝你学习愉快，真正「跑起来」是理解强化学习和 RLHF 的最好方式。如果你补完代码后希望我帮你检查或改进训练脚本，也可以随时告诉我。
