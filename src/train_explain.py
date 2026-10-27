"""Coursework Transformer experiment, packaged for repository use.

The original array layout and model logic are retained. Read
docs/experiment-notes.md before interpreting the historical results.
"""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.datasets import load_from_tsfile_to_dataframe
import matplotlib.pyplot as plt
import seaborn as sns

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Transformer for time series
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=128, d_model=64, nhead=4, num_layers=1, dim_feedforward=128, num_classes=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x[:, -1, :])

# Load and preprocess time series data
def load_ts_data(train_path, test_path):
    X_train_df, y_train = load_from_tsfile_to_dataframe(train_path)
    X_test_df, y_test = load_from_tsfile_to_dataframe(test_path)
    X_train = from_nested_to_3d_numpy(X_train_df).astype(np.float32)
    X_test = from_nested_to_3d_numpy(X_test_df).astype(np.float32)
    unique_labels = sorted(set(y_train))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_train = np.array([label_map[y] for y in y_train])
    y_test = np.array([label_map[y] for y in y_test])
    return X_train, y_train, X_test, y_test

# Training and evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test, device, name, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).long()  # <-- 여기서 long() 추가
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{name} Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).to(device)
        y_test_tensor = torch.tensor(y_test).to(device).long()  # <-- 여기서도 long() 추가
        preds = model(X_test_tensor).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()


from sklearn.metrics import classification_report
import pandas as pd

def detailed_metrics_report(model, X_test, y_test, device, name="Model"):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).to(device)
        y_test_tensor = torch.tensor(y_test).to(device).long()
        preds = model(X_test_tensor).argmax(dim=1).cpu().numpy()

    # 👉 classification report (per class)
    report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()

    # 👉 표시
    print(f"\n[Metrics] Detailed Report for {name}")
    print(df_report[['precision', 'recall', 'f1-score']])



def parse_args():
    parser = argparse.ArgumentParser(description="Train and explain the archived time-series classifier")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--dataset", choices=["Wafer", "FingerMovements", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explanations", choices=["none", "all"], default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["Wafer", "FingerMovements"] if args.dataset == "both" else [args.dataset]
    for name in datasets:
        folder = args.data_dir / name
        train_path = folder / f"{name}_TRAIN.ts"
        test_path = folder / f"{name}_TEST.ts"
        for path in (train_path, test_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing dataset file: {path}. See data/README.md.")
        X_train, y_train, X_test, y_test = load_ts_data(str(train_path), str(test_path))
        print(f"{name}: loaded shape {X_train.shape}; original coursework layout retained")
        model = TimeSeriesTransformer(input_dim=X_train.shape[2])
        train_and_evaluate(model, X_train, y_train, X_test, y_test, device, name, args.epochs)
        detailed_metrics_report(model, X_test, y_test, device, name)
        if args.explanations == "all":
            from explainability import (
                analyze_lime_patterns, explain_with_shap, explain_with_integrated_gradients,
            )
            analyze_lime_patterns(model, X_test, device, name=name, num_samples=100)
            explain_with_shap(model, X_train, X_test, device, name)
            explain_with_integrated_gradients(model, X_test, device, name)


if __name__ == "__main__":
    main()
