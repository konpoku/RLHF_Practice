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
    max_samples: int = 2000


@dataclass
class TrainConfig:
    train_batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 1e-5
    # PPO 专用
    ppo_epochs: int = 2
    ppo_clip_range: float = 0.2
    value_coef: float = 0.5
    # PPO 可选 KL 惩罚系数（约束策略不要偏离旧策略太多）
    kl_coef: float = 0.0
    # GRPO 专用
    grpo_group_size: int = 4
    # 日志
    log_every: int = 10


def get_default_model_config() -> ModelConfig:
    return ModelConfig()


def get_default_data_config() -> DataConfig:
    return DataConfig()


def get_default_train_config() -> TrainConfig:
    return TrainConfig()
