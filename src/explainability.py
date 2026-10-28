import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from captum.attr import IntegratedGradients
from lime.lime_tabular import LimeTabularExplainer
from torch import nn


def plot_attribution(values, title, path):
    limit = max(float(np.abs(values).max()), 1e-8)
    figure, axis = plt.subplots(figsize=(10, 3.5))
    image = axis.imshow(values.T, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set(xlabel="Time step", ylabel="Channel", title=title)
    axis.set_yticks(np.arange(0, values.shape[1], max(1, values.shape[1] // 6)))
    figure.colorbar(image, ax=axis, label="Attribution")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def explain(model, x_train, x_test, device, output, sample_count=2, seed=42):
    model.eval()
    rng = np.random.default_rng(seed)
    background_index = rng.choice(len(x_train), min(32, len(x_train)), replace=False)
    background = x_train[background_index]
    time_steps, channels = x_train.shape[1:]
    selected = rng.choice(len(x_test), min(sample_count, len(x_test)), replace=False)
    probability_model = nn.Sequential(model, nn.Softmax(dim=1))

    def probabilities(flattened):
        values = flattened.reshape(-1, time_steps, channels).astype(np.float32)
        result = []
        with torch.no_grad():
            for start in range(0, len(values), 64):
                batch = torch.as_tensor(values[start:start + 64], device=device)
                result.append(probability_model(batch).cpu().numpy())
        return np.concatenate(result)

    lime = LimeTabularExplainer(
        background.reshape(len(background), -1), mode="classification", random_state=seed,
        feature_names=[f"t{time}_c{channel}" for time in range(time_steps) for channel in range(channels)],
        discretize_continuous=False,
    )
    shap_explainer = shap.GradientExplainer(probability_model, torch.as_tensor(background, device=device))
    integrated_gradients = IntegratedGradients(probability_model)
    baseline = torch.as_tensor(background.mean(axis=0, keepdims=True), device=device)
    summary = []
    for index in selected:
        sample = x_test[index:index + 1]
        tensor = torch.as_tensor(sample, device=device)
        target = int(probabilities(sample.reshape(1, -1)).argmax())
        explanation = lime.explain_instance(
            sample.reshape(-1), probabilities, labels=(target,), num_features=min(20, time_steps * channels),
            num_samples=400,
        )
        lime_values = np.zeros(time_steps * channels)
        for feature, weight in explanation.as_map()[target]:
            lime_values[feature] = weight
        lime_values = lime_values.reshape(time_steps, channels)
        raw_shap = shap_explainer.shap_values(tensor, nsamples=64, rseed=seed)
        shap_values = raw_shap[target][0] if isinstance(raw_shap, list) else np.asarray(raw_shap)[0, ..., target]
        ig, delta = integrated_gradients.attribute(
            tensor, baselines=baseline, target=target, n_steps=64,
            internal_batch_size=8, return_convergence_delta=True,
        )
        ig_values = ig.detach().cpu().numpy()[0]
        for name, values in (("lime", lime_values), ("shap", shap_values), ("ig", ig_values)):
            if values.shape != (time_steps, channels) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} attribution shape or values: {values.shape}")
            np.save(output / f"sample_{index}_{name}.npy", values)
            plot_attribution(values, f"{name.upper()} / sample {index} / class {target}",
                             output / f"sample_{index}_{name}.png")
        summary.append({"test_index": int(index), "target_class": target,
                        "ig_convergence_delta": float(delta.item()), "lime_local_score": float(explanation.score)})
        print(f"Explanations saved: sample {index}", flush=True)
    (output / "explanations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
