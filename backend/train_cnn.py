"""Train EmotionCNN on mel-spectrograms extracted from dataset/raw/.

Usage
-----
    cd backend
    python train_cnn.py --epochs 100 --batch-size 32 --num-workers 0

    # If CUDA is not available (CPU-only, slower):
    python train_cnn.py --epochs 100 --batch-size 16 --num-workers 0

Outputs (saved to app/models/emotion_model/)
--------------------------------------------
    emotion_cnn.pt        — best model checkpoint (by val accuracy)
    training_curves.png   — loss + accuracy plots
    confusion_matrix.png  — test-set confusion matrix
    eval_report.txt       — per-class precision / recall / F1
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
RAW_DIR = BACKEND_DIR / "dataset" / "raw"
OUT_DIR = BACKEND_DIR / "app" / "models" / "emotion_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "emotion_cnn.pt"


# ---------------------------------------------------------------------------
# Lazy imports (fail early with clear messages)
# ---------------------------------------------------------------------------
def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        raise SystemExit(
            f"ERROR: '{pkg}' is not installed.\n"
            f"Run:  pip install torch torchvision torchaudio --index-url "
            f"https://download.pytorch.org/whl/cu121\n"
            f"      pip install -r requirements.txt"
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_class_weights(dataset) -> torch.Tensor:
    """Inverse-frequency class weights to handle class imbalance."""
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    n_classes = len(dataset.LABEL_TO_IDX)
    total = len(labels)
    weights = torch.zeros(n_classes, dtype=torch.float32)
    for cls_idx, cnt in counts.items():
        weights[cls_idx] = total / (n_classes * cnt)
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"    batch {batch_idx+1}/{len(loader)}  "
                f"loss={loss.item():.4f}",
                end="\r",
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def save_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs, train_losses, label="Train Loss")
        ax1.plot(epochs, val_losses, label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, [a * 100 for a in train_accs], label="Train Acc %")
        ax2.plot(epochs, [a * 100 for a in val_accs], label="Val Acc %")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("%")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved training curves → {out_path}")
    except Exception as exc:
        print(f"  [warn] Could not save training curves: {exc}")


def save_confusion_matrix(
    all_labels: list[int],
    all_preds: list[int],
    class_names: list[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(class_names)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(all_labels, all_preds):
            cm[t][p] += 1

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=range(n),
            yticks=range(n),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix (test set)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved confusion matrix → {out_path}")
    except Exception as exc:
        print(f"  [warn] Could not save confusion matrix: {exc}")


def save_eval_report(
    all_labels: list[int],
    all_preds: list[int],
    class_names: list[str],
    test_acc: float,
    out_path: Path,
) -> None:
    n = len(class_names)
    per_class: dict[str, dict] = {}
    for cls_idx, cls_name in enumerate(class_names):
        tp = sum(1 for t, p in zip(all_labels, all_preds) if t == cls_idx and p == cls_idx)
        fp = sum(1 for t, p in zip(all_labels, all_preds) if t != cls_idx and p == cls_idx)
        fn = sum(1 for t, p in zip(all_labels, all_preds) if t == cls_idx and p != cls_idx)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls_name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    lines = [
        "Emotion CNN — Evaluation Report",
        "=" * 50,
        f"Test Accuracy: {test_acc*100:.2f}%",
        "",
        f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}",
        "-" * 50,
    ]
    for cls_name, metrics in per_class.items():
        lines.append(
            f"{cls_name:<12} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>8.4f} {metrics['support']:>9}"
        )

    macro_p = np.mean([m["precision"] for m in per_class.values()])
    macro_r = np.mean([m["recall"] for m in per_class.values()])
    macro_f = np.mean([m["f1"] for m in per_class.values()])
    lines += [
        "-" * 50,
        f"{'macro avg':<12} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>8.4f}",
    ]

    report = "\n".join(lines)
    out_path.write_text(report)
    print(f"  Saved eval report → {out_path}")
    print("\n" + report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train EmotionCNN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Use 0 on Windows to avoid DataLoader deadlock.")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    args = parser.parse_args()

    # --- check deps ---
    _require("torch")
    _require("librosa")
    _require("matplotlib")

    import torch
    from app.services.emotion_detection_service.cnn_model import EmotionCNN, EMOTIONS
    from app.services.emotion_detection_service.cnn_dataset import build_splits

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("EmotionCNN Training")
    print("=" * 60)
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"Output : {OUT_DIR}")
    print()

    # --- data ---
    if not RAW_DIR.is_dir() or not any(RAW_DIR.rglob("*.wav")):
        raise SystemExit(
            f"ERROR: No WAV files found in {RAW_DIR}\n"
            "Run:  python download_datasets.py && python prepare_dataset.py"
        )

    train_ds, val_ds, test_ds = build_splits(RAW_DIR)

    if len(train_ds) == 0:
        raise SystemExit("ERROR: Training set is empty. Check dataset/raw/ structure.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # --- model / optimizer / scheduler ---
    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # --- training loop ---
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Starting training …\n")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        scheduler.step(vl_acc)

        elapsed = time.time() - t_ep
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.1f}%  "
            f"({elapsed:.1f}s)"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": len(EMOTIONS),
                    "emotions": EMOTIONS,
                    "epoch": epoch,
                    "val_acc": vl_acc,
                },
                MODEL_PATH,
            )
            print(f"  *** New best val_acc={vl_acc*100:.2f}% — checkpoint saved ***")
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs).")
                break

    total_time = time.time() - t_start
    print(f"\nTraining finished in {total_time/60:.1f} min")
    print(f"Best val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")

    # --- test evaluation ---
    print("\nEvaluating on test set …")
    # Load best checkpoint for evaluation
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    ts_loss, ts_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {ts_acc*100:.2f}%")

    # --- plots & report ---
    save_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        OUT_DIR / "training_curves.png",
    )
    save_confusion_matrix(
        all_labels, all_preds, EMOTIONS,
        OUT_DIR / "confusion_matrix.png",
    )
    save_eval_report(
        all_labels, all_preds, EMOTIONS, ts_acc,
        OUT_DIR / "eval_report.txt",
    )

    print("\n" + "=" * 60)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy : {ts_acc*100:.2f}%")
    print("=" * 60)
    print("\nNext step: start the API")
    print("  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
