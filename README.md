# GNN-BERT Music Context Understanding

Course project for Neural Networks (CSE425 / EEE474 / CSE715): a hybrid BERT + GNN system
for understanding musical context (genre, mood, harmonic structure, lyrical semantics).

**Status:** Task 1 (BERT baseline for tag understanding) is implemented. Tasks 2-4 (GNN
structure, GNN-BERT fusion, contrastive retrieval) are not yet started.

## Task 1: BERT Baseline for Music Tag Understanding

Fine-tunes a BERT-family text encoder to predict multi-label music tags from natural
language descriptions.

**Dataset:** [`google/MusicCaps`](https://huggingface.co/datasets/google/MusicCaps)
(5,521 clips), used in text-only mode — no audio download required for this task. Each
clip has a `caption` (a natural-language description) and an `aspect_list` (short free-text
phrases like `"sad"`, `"piano melody"`, `"fast tempo"`). We build a fixed vocabulary from
the **top-50 most frequent aspect phrases** and treat them as multi-label tags, then train
a classifier to predict which of those 50 tags apply from the caption text alone. This is
the "MusicCaps caption → tag proxy task" named as an alternative to MagnaTagATune in the
assignment spec.

**Model:** `distilbert-base-uncased` (configurable to `bert-base-uncased` in `config.yaml`)
with CLS-token pooling and a linear multi-label head, fine-tuned end-to-end with
per-tag binary cross-entropy.

**Split:** MusicCaps ships only a single `train` split with no artist/author field to
de-duplicate on, so we use a random 80/10/10 train/val/test split with a fixed seed (42),
rather than the FMA/MagnaTagATune "no artist leakage" split the spec describes for those
datasets.

### Results

Test set (553 held-out clips), `distilbert-base-uncased`, 10 epochs, batch size 16, lr 2e-5:

| Model | Macro-F1 | Micro-F1 | Mean AUC-PR |
|---|---|---|---|
| Majority-tag baseline | 0.000 | 0.000 | – |
| Task 1: DistilBERT fine-tuned | **0.560** | **0.714** | **0.682** |

The majority baseline scores 0 because no single top-50 tag exceeds a 50% base rate in the
training set (the most common tag, "low quality", appears in only ~22% of clips), so
always-predict-negative is its best constant strategy — the fine-tuned model's F1 gain
comes entirely from actually reading the caption text.

Validation Macro/Micro-F1 climbed steadily and had not plateaued by epoch 10 (best
Macro-F1 0.525 at epoch 9); see `results/plots/f1_curve.png` and `loss_curve.png`. Training
loss kept falling past epoch 6 while validation loss flattened, indicating mild overfitting
in later epochs — a smaller learning rate or early stopping around epoch 8-9 would likely
match these results with less overfit.

5 example test-set predictions and a CLS-attention visualization (`results/plots/attention_example.png`)
are saved in `results/metrics.json`. Qualitatively, the model is confident and correct on
tags with strong lexical cues in the caption (e.g. predicting `low quality` 0.95, `noisy`
0.92, `mono` 0.86 for a caption literally containing those words), and less confident on
tags that require more world knowledge to infer (e.g. only 0.10 confidence on `male vocal`
for a caption describing "a male vocalist singing").

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Task 1

```bash
source .venv/bin/activate

# 1. Download MusicCaps, build the top-50 tag vocabulary and train/val/test splits
python -m src.data_musiccaps

# 2. Fine-tune the BERT classifier (writes results/checkpoints/best_model.pt + results/metrics.json)
python -m src.train

# 3. Evaluate on the test set: Macro/Micro-F1, AUC-PR, baseline comparison,
#    5 example predictions, loss/F1 curves, and an attention visualization
python -m src.evaluate
```

Then explore `notebooks/eda.ipynb` (dataset exploration) and
`notebooks/demo_context.ipynb` (interactive inference on your own caption text).

## Project structure

```
├── config.yaml              # model/data/training hyperparameters
├── data/
│   ├── raw/                 # HF datasets cache (gitignored)
│   ├── processed/           # tag_vocab.json (top-50 tags)
│   └── splits/              # train/val/test.json
├── notebooks/
│   ├── eda.ipynb
│   └── demo_context.ipynb
├── src/
│   ├── data_musiccaps.py    # dataset download, tag vocab, splits, Dataset class
│   ├── model.py              # BertMultiLabelClassifier
│   ├── train.py               # fine-tuning loop
│   ├── evaluate.py            # test metrics, baseline, examples, plots
│   └── utils.py                # config loader, device selection
└── results/
    ├── metrics.json          # training history + test report
    ├── plots/                 # loss_curve.png, f1_curve.png, attention_example.png
    └── checkpoints/           # best_model.pt (gitignored — large binary)
```

## Next steps (Tasks 2-4, not yet implemented)

- **Task 2 (Medium):** GraphSAGE/GAT on chord/segment graphs built from chroma features
  (needs audio, e.g. GTZAN or FMA-small).
- **Task 3 (Hard):** Cross-attention fusion of the Task 2 GNN encoder with this Task 1 BERT
  encoder for joint genre + mood + emotion prediction.
- **Task 4 (Advanced):** Contrastive dual-encoder (InfoNCE) between audio graphs and
  MusicCaps captions for cross-modal retrieval.
