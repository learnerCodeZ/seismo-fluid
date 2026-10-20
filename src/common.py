"""通用工具（迁移自 dachuang_cnn_lstm/src/common.py；路径改为本工作区本地，不依赖旧仓）。

- set_seed 与旧仓逐行一致：所有实验入口必须先调用；
- 路径全部相对本工作区根目录解析，禁止硬编码绝对路径。
"""
import json
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def set_seed(seed: int) -> None:
    """固定 random/numpy/torch 随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(name: str) -> dict:
    """读取 configs/ 下指定实验配置（JSON）。"""
    with open(CONFIG_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)
