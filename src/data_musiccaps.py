"""Task 1 data pipeline: MusicCaps captions -> top-K aspect-tag multi-label targets.

Each MusicCaps row has a free-text `aspect_list` (e.g. "['sad', 'ballad', 'piano melody']")
alongside its `caption`. We treat the top-K most frequent aspect phrases as a fixed tag
vocabulary (analogous to MagnaTagATune's top-50 tags) and predict them from the caption
text -- the "MusicCaps caption -> tag proxy task" named in the assignment spec.
"""
import ast
import json
import random
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.utils import ROOT, load_config


def parse_aspects(aspect_list_str: str) -> list[str]:
    return ast.literal_eval(aspect_list_str)


def build_tag_vocab_and_splits(cfg: dict) -> None:
    data_cfg = cfg["data"]
    dataset = load_dataset(
        data_cfg["dataset_name"], split="train", cache_dir=str(ROOT / data_cfg["raw_dir"])
    )

    aspects_per_row = [parse_aspects(a) for a in dataset["aspect_list"]]
    counter = Counter()
    for aspects in aspects_per_row:
        counter.update(aspects)

    top_tags = counter.most_common(data_cfg["top_k_tags"])
    tag_vocab = [tag for tag, _ in top_tags]

    processed_dir = ROOT / data_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "tag_vocab.json", "w") as f:
        json.dump([{"tag": tag, "count": count} for tag, count in top_tags], f, indent=2)

    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(data_cfg["seed"])
    rng.shuffle(indices)

    n_train = int(n * data_cfg["train_frac"])
    n_val = int(n * data_cfg["val_frac"])
    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    splits_dir = ROOT / data_cfg["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, idxs in split_indices.items():
        rows = []
        for i in idxs:
            aspects = set(aspects_per_row[i])
            label = [1 if tag in aspects else 0 for tag in tag_vocab]
            rows.append(
                {
                    "index": i,
                    "ytid": dataset[i]["ytid"],
                    "caption": dataset[i]["caption"],
                    "label": label,
                }
            )
        with open(splits_dir / f"{split_name}.json", "w") as f:
            json.dump(rows, f, indent=2)

    print(f"Tag vocab: {len(tag_vocab)} tags (top 5: {tag_vocab[:5]})")
    print(
        f"Splits: train={len(split_indices['train'])}, "
        f"val={len(split_indices['val'])}, test={len(split_indices['test'])}"
    )


class MusicCapsTagDataset(Dataset):
    """Caption -> multi-hot tag vector dataset, built from a pre-computed split file."""

    def __init__(self, cfg: dict, split: str):
        data_cfg = cfg["data"]
        self.max_length = data_cfg["max_length"]

        with open(ROOT / data_cfg["processed_dir"] / "tag_vocab.json") as f:
            self.tag_vocab = [entry["tag"] for entry in json.load(f)]

        with open(ROOT / data_cfg["splits_dir"] / f"{split}.json") as f:
            rows = json.load(f)

        self.examples = [
            {
                "caption": row["caption"],
                "label": torch.tensor(row["label"], dtype=torch.float32),
            }
            for row in rows
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        ex = self.examples[i]
        enc = self.tokenizer(
            ex["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": ex["label"],
            "caption": ex["caption"],
        }


if __name__ == "__main__":
    build_tag_vocab_and_splits(load_config())
