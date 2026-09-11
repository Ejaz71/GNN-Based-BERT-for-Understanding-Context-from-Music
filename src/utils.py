from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "config.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
