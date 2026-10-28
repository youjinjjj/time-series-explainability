import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sktime.datasets import load_from_tsfile
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, heads=4, layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        position = torch.arange(1000).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        encoding = torch.zeros(1000, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position", encoding.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        embedded = self.embedding(x) + self.position[:, :x.shape[1]]
        encoded = self.transformer(embedded)
        return self.classifier(encoded.mean(dim=1))


def load_data(folder, name):
    arrays, labels = [], []
    for split in ("TRAIN", "TEST"):
        path = folder / name / f"{name}_{split}.ts"
        x, y = load_from_tsfile(str(path), return_data_type="numpy3D")
        arrays.append(x.transpose(0, 2, 1).astype(np.float32))
        labels.append(np.asarray(y).astype(str))
    label_names = sorted(np.unique(labels[0]).tolist())
    mapping = {label: index for index, label in enumerate(label_names)}
    if len(mapping) != 2 or not set(labels[1]).issubset(mapping):
        raise ValueError("Expected matching binary labels in train and test files")
    y_train, y_test = [np.array([mapping[label] for label in values]) for values in labels]
    x_train, x_test = arrays
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Input contains non-finite values")
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    scale = x_train.std(axis=(0, 1), keepdims=True).clip(1e-6)
    return (x_train - mean) / scale, y_train, (x_test - mean) / scale, y_test, label_names, mean, scale


def predict(model, x, device, batch_size=64):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], device=device)
            values.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(values)


def run(args, name):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    x_train, y_train, x_test, y_test, labels, mean, scale = load_data(args.data_dir, name)
    model = TimeSeriesTransformer(x_train.shape[2]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(x_train)})
        print(f"{name} epoch {epoch + 1}: loss={history[-1]['train_loss']:.4f}", flush=True)

    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    predicted = predict(model, x_test, device).argmax(axis=1)
    report = classification_report(y_test, predicted, target_names=labels, output_dict=True, zero_division=0)
    metrics = {
        "dataset": name, "train_shape": list(x_train.shape), "test_shape": list(x_test.shape),
        "accuracy": report["accuracy"], "macro_f1": f1_score(y_test, predicted, average="macro"),
        "labels": labels, "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "epochs": args.epochs, "seed": args.seed,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history)
    torch.save({"model": model.state_dict(), "input_dim": x_train.shape[2],
                "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                "labels": labels}, output / "model.pt")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history])
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    matrix = np.asarray(metrics["confusion_matrix"])
    axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            axes[1].text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
                xlabel="Predicted label", ylabel="Actual label")
    figure.suptitle(name)
    figure.tight_layout()
    figure.savefig(output / "training_and_test.png", dpi=150)
    plt.close(figure)
    print(f"{name}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}", flush=True)
    if args.explanations != "none":
        from explainability import explain
        explain(model, x_train, x_test, device, output, args.explain_samples, args.seed)
    return metrics


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "current")
    parser.add_argument("--dataset", choices=["Wafer", "FingerMovements", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--explanations", choices=["none", "all"], default="all")
    parser.add_argument("--explain-samples", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.explain_samples) < 1:
        parser.error("epochs, batch-size, threads and explain-samples must be positive")
    return args


if __name__ == "__main__":
    options = parse_args()
    torch.set_num_threads(options.threads)
    names = ["Wafer", "FingerMovements"] if options.dataset == "both" else [options.dataset]
    for dataset in names:
        run(options, dataset)
