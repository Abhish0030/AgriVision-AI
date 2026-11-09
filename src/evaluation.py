"""Real held-out metrics, hardware-aware latency and persisted model comparisons."""
import time
import platform
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from .dataset import load_manifest
from .artifacts import model_artifacts, validate_evaluation_record, validate_model_contract
from .preprocessing import make_dataset, resize_rgb
from .utils import atomic_copy, file_hash, read_json, validate_probabilities, write_json
from .metrics import classification_metrics, rejection_summary
from .validation import load_rejection_profile


def evaluate_architecture(architecture, paths, config):
    frame, mapping, dataset_report = load_manifest(paths, verify_files=True, config=config)
    model_path, trained_config, trained_mapping, state, digest = model_artifacts(paths, architecture)
    if config != trained_config:
        raise ValueError("Evaluate with the same configuration that was used for training.")
    if trained_mapping != mapping:
        raise ValueError("Model label mapping differs from dataset mapping.")
    if not state.get("complete"):
        raise ValueError("Finish both training stages before final held-out evaluation.")
    if state["signature"]["manifest_sha256"] != dataset_report["manifest_sha256"]:
        raise ValueError("Model was trained with a different dataset manifest.")
    model = tf.keras.models.load_model(model_path, compile=False)
    validate_model_contract(model, architecture, mapping, config)
    test = frame[frame.split == "test"].copy()
    probabilities = validate_probabilities(model.predict(make_dataset(test, paths.dataset, config), verbose=0), len(mapping), len(test))
    labels = test.label.to_numpy()
    predicted = np.argmax(probabilities, axis=1)
    ids = list(range(len(mapping)))
    names = [name for name, _ in sorted(mapping.items(), key=lambda x: x[1])]
    measured = classification_metrics(labels, probabilities, mapping)
    report = measured["classification_report"]
    batch = resize_rgb(paths.dataset / test.iloc[0].path, config.image_size)[None]
    for _ in range(5):
        model(batch, training=False).numpy()
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        model(batch, training=False).numpy()  # Synchronize before stopping timer.
        timings.append((time.perf_counter() - start) * 1000)
    result = {**measured, "model": architecture, "training_seconds": state["training_seconds"],
              "inference_ms_median": float(np.median(timings)), "inference_ms_p95": float(np.percentile(timings, 95)),
              "latency_protocol": "batch=1, 5 warmups, 30 synchronized calls; excludes image I/O and Grad-CAM",
              "tensorflow": tf.__version__, "devices": [d.name for d in tf.config.list_logical_devices()],
              "execution_context": {"platform": platform.platform(), "tensorflow": tf.__version__,
                                    "cpu_processor": platform.processor() or "unknown", "cpu_count": os.cpu_count(),
                                    "keras": getattr(tf.keras, "__version__", "unknown"),
                                    "gpu_details": [tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices("GPU")],
                                    "precision_policy": model.dtype_policy.name},
              "test_images": len(test), "parameters": model.count_params(), "model_sha256": file_hash(model_path),
              "manifest_sha256": dataset_report["manifest_sha256"], "classification_report": report}
    result["split"] = "internal_test"
    rejection = load_rejection_profile(model_path.parent, digest, dataset_report["manifest_sha256"])
    if rejection:
        result["rejection"] = rejection_summary(labels, probabilities, rejection)
    write_json(paths.output / "evaluation" / f"{architecture}.json", result)
    pd.DataFrame(report).T.to_csv(paths.output / "evaluation" / f"{architecture}_classes.csv")
    predictions = test[["path", "class_name", "label"]].copy()
    predictions["predicted_label"] = predicted
    predictions["confidence"] = probabilities.max(axis=1)
    predictions.to_csv(paths.output / "predictions" / f"{architecture}_test.csv", index=False)
    atomic_copy(paths.output / "predictions" / f"{architecture}_test.csv", paths.output / "evaluation" / f"{architecture}_internal_predictions.csv")
    matrix = confusion_matrix(labels, predicted, labels=ids)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(paths.output / "confusion_matrices" / f"{architecture}.csv")
    plot_confusion(matrix, names, paths.output / "confusion_matrices" / f"{architecture}.png")
    for suffix in ("csv", "png"):
        atomic_copy(paths.output / "confusion_matrices" / f"{architecture}.{suffix}", paths.output / "evaluation" / f"{architecture}_confusion.{suffix}")
    plot_per_class(report, names, paths.output / "evaluation" / f"{architecture}_per_class.png")
    return result


def plot_confusion(matrix, names, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(max(9, len(names) * .55), max(8, len(names) * .5)))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=names, yticklabels=names, cmap="Greens", ax=ax)
    ax.set(xlabel="Predicted label", ylabel="Actual label")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def compare_models(paths, architectures, allow_config_differences=False):
    rows, signatures, contexts = [], [], []
    if len(set(architectures)) < 2:
        raise ValueError("Architecture comparison requires at least two distinct models.")
    for architecture in architectures:
        path = paths.output / "evaluation" / f"{architecture}.json"
        if not path.exists():
            raise FileNotFoundError(f"Evaluation not yet available for {architecture}; evaluate every requested model.")
        _, _, _, state, digest = model_artifacts(paths, architecture)
        value = read_json(path)
        validate_evaluation_record(value, architecture, digest, state["signature"]["manifest_sha256"])
        signatures.append({key: state["signature"].get(key) for key in ("config", "class_indices", "manifest_sha256", "implementation_sha256")})
        contexts.append(value.get("execution_context"))
        rows.append({k: v for k, v in value.items() if not isinstance(v, (dict, list))})
    if len({row["manifest_sha256"] for row in rows}) != 1:
        raise ValueError("Cannot compare evaluations from different manifests.")
    same_configuration = all(signature["config"] == signatures[0]["config"] for signature in signatures)
    comparable_data = all(all(signature[key] == signatures[0][key] for key in ("class_indices", "manifest_sha256", "implementation_sha256")) for signature in signatures)
    if not comparable_data or (not same_configuration and not allow_config_differences):
        raise ValueError("Comparison requires the same configuration, labels, source and frozen manifest for both models.")
    known_hardware = all(context is not None and (context.get("gpu_details") or context.get("cpu_processor", "unknown") != "unknown") for context in contexts)
    comparable = known_hardware and len({json.dumps(c, sort_keys=True) for c in contexts}) == 1
    for row in rows:
        row["latency_comparable"] = comparable
        row["configuration_comparable"] = same_configuration
    if not comparable:
        print("Timing contexts differ or are missing. Do not interpret the latency columns as a controlled speed comparison.")
    table = pd.DataFrame(rows)
    table.to_csv(paths.output / "evaluation" / "model_comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = table.set_index("model")[["accuracy", "macro_f1", "balanced_accuracy"]].plot.bar(ylim=(0, 1), figsize=(8, 4))
    ax.figure.tight_layout()
    ax.figure.savefig(paths.output / "evaluation" / "model_comparison.png", dpi=150)
    plt.close(ax.figure)
    return table


def plot_per_class(report, names, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = pd.DataFrame({name: report[name] for name in names}).T[["precision", "recall", "f1-score"]]
    ax = values.plot.barh(figsize=(10, max(5, len(names) * .4)), xlim=(0, 1))
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=150)
    plt.close(ax.figure)
