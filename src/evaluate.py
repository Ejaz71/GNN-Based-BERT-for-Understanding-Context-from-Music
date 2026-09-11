import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.data_musiccaps import MusicCapsTagDataset
from src.model import BertMultiLabelClassifier
from src.utils import ROOT, get_device, load_config


def majority_baseline(train_labels: np.ndarray, n_test: int, threshold: float = 0.5) -> np.ndarray:
    """Predicts the training-set base rate per tag, thresholded -- a naive floor."""
    base_rates = train_labels.mean(axis=0)
    row = (base_rates >= threshold).astype(np.float32)
    return np.tile(row, (n_test, 1))


def get_probs_and_labels(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def tune_per_tag_thresholds(
    val_probs: np.ndarray, val_labels: np.ndarray, grid: np.ndarray | None = None
) -> np.ndarray:
    """Picks, per tag, the probability threshold that maximizes F1 on the validation set.

    A single global 0.5 cutoff is a poor fit for a 50-way multi-label head trained with
    plain BCE: rarer/harder tags end up systematically under-confident (their true
    positives rarely cross 0.5) while common/easy tags are well-calibrated. Tuning per tag
    on val (never on test) fixes the miscalibration without touching the model at all.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    num_labels = val_labels.shape[1]
    thresholds = np.full(num_labels, 0.5, dtype=np.float32)
    for k in range(num_labels):
        if val_labels[:, k].sum() == 0:
            continue  # no positive examples to tune against; keep the default
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            preds = (val_probs[:, k] >= t).astype(np.float32)
            f1 = f1_score(val_labels[:, k], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[k] = best_t
    return thresholds


def plot_curves(history: dict, plots_dir) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Task 1: Loss vs. Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["val_macro_f1"], label="val Macro-F1")
    plt.plot(epochs, history["val_micro_f1"], label="val Micro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Task 1: Validation F1 vs. Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "f1_curve.png", dpi=150)
    plt.close()


def plot_attention(model, dataset, device, plots_dir, example_idx: int) -> str:
    model.eval()
    example = dataset[example_idx]
    input_ids = example["input_ids"].unsqueeze(0).to(device)
    attention_mask = example["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        _, attentions = model(input_ids, attention_mask, output_attentions=True)

    last_layer_attn = attentions[-1][0]  # (num_heads, seq_len, seq_len)
    cls_attn = last_layer_attn[:, 0, :].mean(dim=0).cpu().numpy()

    tokens = dataset.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    n_real = int(example["attention_mask"].sum().item())
    tokens, cls_attn = tokens[:n_real], cls_attn[:n_real]

    plt.figure(figsize=(max(8, n_real * 0.3), 3))
    plt.bar(range(n_real), cls_attn)
    plt.xticks(range(n_real), tokens, rotation=90, fontsize=7)
    plt.ylabel("CLS attention\n(last layer, mean over heads)")
    plt.title("Task 1: Attention visualization for one test example")
    plt.tight_layout()
    plt.savefig(plots_dir / "attention_example.png", dpi=150)
    plt.close()
    return example["caption"]


def main():
    cfg = load_config()
    device = get_device()

    train_ds = MusicCapsTagDataset(cfg, "train")
    val_ds = MusicCapsTagDataset(cfg, "val")
    test_ds = MusicCapsTagDataset(cfg, "test")
    tag_vocab = test_ds.tag_vocab
    num_labels = len(tag_vocab)

    checkpoint_path = ROOT / cfg["paths"]["checkpoint_dir"] / "best_model.pt"
    model = BertMultiLabelClassifier(cfg["model"]["name"], num_labels).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    batch_size = cfg["train"]["batch_size"]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    val_probs, val_labels = get_probs_and_labels(model, val_loader, device)
    all_probs, all_labels = get_probs_and_labels(model, test_loader, device)

    ap_scores = [
        average_precision_score(all_labels[:, k], all_probs[:, k])
        for k in range(num_labels)
        if all_labels[:, k].sum() > 0
    ]
    mean_ap = float(np.mean(ap_scores)) if ap_scores else 0.0

    train_labels = np.stack([ex["label"].numpy() for ex in train_ds.examples])
    baseline_preds = majority_baseline(train_labels, n_test=len(test_ds))
    baseline_macro_f1 = f1_score(all_labels, baseline_preds, average="macro", zero_division=0)
    baseline_micro_f1 = f1_score(all_labels, baseline_preds, average="micro", zero_division=0)

    def build_report(preds: np.ndarray) -> dict:
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, preds, average=None, zero_division=0
        )
        return {
            "macro_f1": float(f1_score(all_labels, preds, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(all_labels, preds, average="micro", zero_division=0)),
            "per_tag": [
                {
                    "tag": tag_vocab[k],
                    "precision": float(precision[k]),
                    "recall": float(recall[k]),
                    "f1": float(f1[k]),
                    "support": int(support[k]),
                }
                for k in range(num_labels)
            ],
        }

    fixed_threshold = cfg["train"]["threshold"]
    fixed_preds = (all_probs >= fixed_threshold).astype(np.float32)
    fixed_report = build_report(fixed_preds)

    # Per-tag threshold tuning: calibrated on val, applied to test (see docstring for why).
    tuned_thresholds = tune_per_tag_thresholds(val_probs, val_labels)
    tuned_preds = (all_probs >= tuned_thresholds[None, :]).astype(np.float32)
    tuned_report = build_report(tuned_preds)

    test_report = {
        "mean_auc_pr": mean_ap,
        "baseline_majority_macro_f1": float(baseline_macro_f1),
        "baseline_majority_micro_f1": float(baseline_micro_f1),
        "fixed_threshold_0.5": fixed_report,
        "tuned_per_tag_threshold": {**tuned_report, "thresholds": tuned_thresholds.tolist()},
        # kept for backwards compatibility with earlier runs/notebooks
        "macro_f1": fixed_report["macro_f1"],
        "micro_f1": fixed_report["micro_f1"],
        "per_tag": fixed_report["per_tag"],
    }

    print("=== Test set results ===")
    print(f"{'':20s} {'Macro-F1':>10s} {'Micro-F1':>10s}")
    print(f"{'Majority baseline':20s} {baseline_macro_f1:10.4f} {baseline_micro_f1:10.4f}")
    print(f"{'Fixed threshold 0.5':20s} {fixed_report['macro_f1']:10.4f} {fixed_report['micro_f1']:10.4f}")
    print(f"{'Tuned per-tag thresh':20s} {tuned_report['macro_f1']:10.4f} {tuned_report['micro_f1']:10.4f}")
    print(f"Mean AUC-PR (threshold-independent): {mean_ap:.4f}")

    examples_out = []
    rng = np.random.RandomState(cfg["data"]["seed"])
    sample_idx = rng.choice(len(test_ds), size=min(5, len(test_ds)), replace=False)
    for idx in sample_idx:
        idx = int(idx)
        caption = test_ds.examples[idx]["caption"]
        probs = all_probs[idx]
        true_tags = [tag_vocab[k] for k in range(num_labels) if all_labels[idx, k] == 1]
        top_k = np.argsort(-probs)[:8]
        pred_tags = [[tag_vocab[k], float(probs[k])] for k in top_k]
        examples_out.append(
            {"caption": caption, "true_tags": true_tags, "top_predicted_tags": pred_tags}
        )
        print("\nCaption:", caption)
        print("True tags:", true_tags)
        print("Top predicted tags:", [f"{t} ({p:.2f})" for t, p in pred_tags])

    metrics_path = ROOT / cfg["paths"]["metrics_path"]
    with open(metrics_path) as f:
        train_metrics = json.load(f)

    plots_dir = ROOT / cfg["paths"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(train_metrics["history"], plots_dir)
    attn_caption = plot_attention(model, test_ds, device, plots_dir, example_idx=int(sample_idx[0]))

    train_metrics["test"] = test_report
    train_metrics["example_predictions"] = examples_out
    train_metrics["attention_example_caption"] = attn_caption
    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)

    print(f"\nSaved plots to {plots_dir}")
    print(f"Updated metrics at {metrics_path}")


if __name__ == "__main__":
    main()
