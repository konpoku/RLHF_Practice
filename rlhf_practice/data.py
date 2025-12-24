from dataclasses import dataclass
from typing import List, Dict, Any

from datasets import load_dataset
from torch.utils.data import Dataset

from .config import DataConfig


@dataclass
class TextSample:
    prompt: str


class HFDatasetWrapper(Dataset):
    """
    一个简单的 Dataset 封装，只返回 prompt 文本，
    具体的 tokenization 在训练脚本里完成。
    """

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"prompt": self.texts[idx]}


def load_text_dataset(config: DataConfig) -> HFDatasetWrapper:
    """
    从 Hugging Face 加载文本数据集，这里默认使用 stanfordnlp/imdb。
    """
    ds = load_dataset(config.dataset_name, split=config.split)
    # 部分数据，避免初次实验太慢
    if config.max_samples is not None:
        ds = ds.select(range(min(config.max_samples, len(ds))))

    texts = [str(example[config.text_column]) for example in ds]
    return HFDatasetWrapper(texts)

