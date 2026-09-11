import json
import time

import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data_musiccaps import MusicCapsTagDataset
from src.model import BertMultiLabelClassifier
from src.utils import ROOT, get_device, load_config


def evaluate_split(model, loader, device, threshold, loss_fn):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            preds = (torch.sigmoid(logits) >= threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, macro_f1, micro_f1


def main():
    cfg = load_config()
    device = get_device()
    print(f"Using device: {device}")

    train_ds = MusicCapsTagDataset(cfg, "train")
    val_ds = MusicCapsTagDataset(cfg, "val")
    num_labels = len(train_ds.tag_vocab)
    print(f"Train examples: {len(train_ds)}, val examples: {len(val_ds)}, tags: {num_labels}")

    train_cfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    model = BertMultiLabelClassifier(cfg["model"]["name"], num_labels).to(device)
    optimizer = AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    best_macro_f1 = -1.0

    checkpoint_dir = ROOT / cfg["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * input_ids.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_macro_f1, val_micro_f1 = evaluate_split(
            model, val_loader, device, train_cfg["threshold"], loss_fn
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{train_cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} val_micro_f1={val_micro_f1:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"  -> new best val_macro_f1={best_macro_f1:.4f}, checkpoint saved")

    metrics_path = ROOT / cfg["paths"]["metrics_path"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"history": history, "best_val_macro_f1": best_macro_f1}, f, indent=2)

    print(f"Training complete. Best val Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
