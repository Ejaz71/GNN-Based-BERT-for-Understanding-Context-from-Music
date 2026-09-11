# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Course project (Neural Networks CSE425/EEE474/CSE715): a hybrid BERT + GNN system for
understanding musical context. It is built as four sequential tasks — (1) BERT multi-label
tagging, (2) GNN on audio structure graphs, (3) GNN-BERT cross-attention fusion, (4)
contrastive GNN-BERT retrieval. **Only Task 1 is implemented so far**; `src/` currently
contains only the Task 1 modules. See `README.md` for the full per-task spec and results
table.

## Setup & common commands

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Task 1 pipeline — run in this order; each stage reads files the previous stage wrote:

```bash
python -m src.data_musiccaps   # downloads google/MusicCaps -> data/processed/tag_vocab.json, data/splits/{train,val,test}.json
python -m src.train             # fine-tunes the classifier -> results/checkpoints/best_model.pt, results/metrics.json
python -m src.evaluate          # test metrics/baseline/examples/plots -> results/plots/*.png, appends into results/metrics.json
```

There is no linter or test suite configured in this repo yet.

`notebooks/eda.ipynb` and `notebooks/demo_context.ipynb` assume the pipeline above has
already been run at least once (they load `data/processed/tag_vocab.json` and
`results/checkpoints/best_model.pt`) and should be run with the `.venv` kernel.

## Architecture

- **Config-driven throughout**: all hyperparameters and paths live in `config.yaml`,
  loaded via `src/utils.load_config()`. Don't hardcode paths or hparams in a script — read
  them from the loaded `cfg` dict, the way `train.py`/`evaluate.py` do.
- **Data flow (Task 1)**: `src/data_musiccaps.py` downloads MusicCaps (HF `datasets`,
  cached under `data/raw/`), builds a fixed top-K tag vocabulary from the free-text
  `aspect_list` field (exact-string matches against the K most frequent phrases — this is
  *not* a controlled tag taxonomy like MagnaTagATune's), and writes each split's JSON with
  the multi-hot label vector already baked in. Downstream code (`MusicCapsTagDataset`,
  `train.py`, `evaluate.py`) never re-touches the raw HF dataset or re-parses `aspect_list`
  — it only reads `data/processed/tag_vocab.json` and `data/splits/*.json`.
- **`MusicCapsTagDataset`** tokenizes lazily inside `__getitem__` (not pre-batched). Fine at
  this dataset's size (~5.5k rows); would need pre-tokenization/caching if scaled up.
- **Model** (`src/model.py`): `BertMultiLabelClassifier` is a thin wrapper around any
  `AutoModel`-compatible HF checkpoint — swapping `model.name` in `config.yaml` between
  `distilbert-base-uncased` and `bert-base-uncased` requires no other code changes, since
  both expose `last_hidden_state[:, 0]` as the CLS representation used for classification.
- **Device selection** is centralized in `src/utils.get_device()` (MPS → CUDA → CPU
  fallback) — reuse it rather than checking `torch.cuda.is_available()` directly; this
  project targets Apple Silicon (MPS) by default.
- **`train.py` and `evaluate.py` are separate entry points**, not one script with flags:
  `train.py` produces the checkpoint + training history (`results/metrics.json`);
  `evaluate.py` loads that checkpoint independently, runs the test split, and merges test
  results/example predictions/plots into the *same* `results/metrics.json`. Changing the
  tag vocabulary, split logic, or model architecture makes the existing checkpoint and
  `results/metrics.json` stale — rerun the full three-stage pipeline, not just one script.
- **No official MusicCaps train/val/test split exists** — `data_musiccaps.py` does a random
  80/10/10 split with a fixed seed (`data.seed` in `config.yaml`), not an artist-based split
  (MusicCaps has no artist field to split on).
- When implementing Tasks 2-4, follow the existing per-task-module pattern: one
  `src/<component>.py` per stage, config-driven, reading/writing through `data/processed/`,
  `data/splits/`, and `results/` the same way the Task 1 modules do.
