"""LIME, SHAP and Integrated Gradients helpers from the team experiment.

Original explanation calculations are retained for provenance. Known array-axis,
LIME feature-label parsing and SHAP plotting limitations are documented.
"""
import torch
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns

from lime.lime_tabular import LimeTabularExplainer
from collections import defaultdict, Counter
import numpy as np
import torch

import re
from collections import defaultdict
import numpy as np

def analyze_lime_patterns(model, X_test, device, name="Model", num_samples=100):
    model.eval()
    model.to(device)
    X_test_np = X_test[:, -1, :]  # (N, Features)
    explainer = LimeTabularExplainer(X_test_np, mode="classification")

    def predict_fn(x):
        x_tensor = torch.tensor(x).float().unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

    feature_impact = defaultdict(list)
    condition_counter = defaultdict(int)

    for _ in range(num_samples):
        i = np.random.randint(0, len(X_test_np))
        exp = explainer.explain_instance(X_test_np[i], predict_fn, num_features=10)
        for feature_str, weight in exp.as_list():
            condition_counter[feature_str] += 1
            feature_nums = [int(s) for s in re.findall(r'\d+', feature_str)]
            if not feature_nums:
                continue
            feature_idx = max(feature_nums)  # 가장 큰 숫자를 feature 인덱스로 간주
            feature_impact[feature_idx].append(weight)

    print(f"\n--- {name}: Top LIME Features from {num_samples} Samples ---")
    sorted_features = sorted(feature_impact.items(), key=lambda x: -np.abs(np.mean(x[1])))
    for feat, weights in sorted_features:
        print(f"Feature {feat}: Mean Weight = {np.mean(weights):.4f}, Count = {len(weights)}")

    print("\nTop Conditions (LIME rules):")
    for cond, count in sorted(condition_counter.items(), key=lambda x: -x[1])[:10]:
        print(f"{cond} - {count} times")

    # --- 시각화 추가 ---
    top_k = 20
    top_features = sorted_features[:top_k]
    features = [f"Feature {feat}" for feat, _ in top_features]
    weights = [np.mean(weights) for _, weights in top_features]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(features[::-1], weights[::-1])  # 역순으로 뒤집기 (가장 큰 게 위로)
    plt.xlabel("Mean LIME Weight")
    plt.title(f"Top {top_k} Influential Features ({name})")
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()


def explain_with_shap(model, X_train, X_test, device, name="Model"):
    model.eval()
    model.to(device)
    X_train_tensor = torch.tensor(X_train[:100]).to(device)
    X_test_tensor = torch.tensor(X_test[:10]).to(device)

    explainer = shap.GradientExplainer(model, X_train_tensor)
    shap_values = explainer.shap_values(X_test_tensor)

    # SHAP 값이 리스트인지 확인하고 image_plot에 맞게 포맷 변경
    if isinstance(shap_values, list):
        shap.image_plot(shap_values, X_test_tensor.cpu().numpy(), show=False)
    else:
        shap.image_plot([shap_values], X_test_tensor.cpu().numpy(), show=False)

    plt.suptitle(f"SHAP Explanation - {name}")
    plt.show()

def explain_with_integrated_gradients(model, X_test, device, name="Model"):
    model.eval()
    model.to(device)
    X_sample = torch.tensor(X_test[0:1]).to(device).requires_grad_(True)

    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(X_sample, target=1, return_convergence_delta=True)

    attr = attr.squeeze().detach().cpu().numpy()  # (sequence_len, feature_dim) or (sequence_len,)

    # 1차원일 경우 2D로 reshape
    if attr.ndim == 1:
        attr = attr.reshape(-1, 1)

    plt.figure(figsize=(10, 4))
    plt.imshow(attr.T, aspect='auto', cmap='coolwarm')
    plt.colorbar()
    plt.title(f"Integrated Gradients Attribution - {name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Features")
    plt.show()