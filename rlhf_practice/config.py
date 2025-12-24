from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "gpt2"  # 可改为更大的模型，如 "gpt2-medium"


@dataclass
class DataConfig:
    dataset_name: str = "stanfordnlp/imdb"
    split: str = "train"
    text_column: str = "text"
    max_prompt_length: int = 128
    max_new_tokens: int = 50
    # 只用一小部分数据便于快速实验
    max_samples = None


@dataclass
class TrainConfig:
    train_batch_size: int = 8
    num_epochs: int = 20
    learning_rate: float = 1e-6
    # PPO 专用
    ppo_epochs: int = 2
    ppo_clip_range: float = 0.2
    value_coef: float = 0.5
    # PPO 可选 KL 惩罚系数（约束策略不要偏离旧策略太多）
    # 推荐值：0.02 ~ 0.1。如果发现模型开始输出乱码或重复单词（Reward Hacking），请调大此参数。
    kl_coef: float = 0.1
    # GRPO 专用
    grpo_group_size: int = 32
    # 日志
    log_every: int = 10


def get_default_model_config() -> ModelConfig:
    return ModelConfig()


def get_default_data_config() -> DataConfig:
    return DataConfig()


def get_default_train_config() -> TrainConfig:
    return TrainConfig()
