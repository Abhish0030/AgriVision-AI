"""Shared measured metrics, confidence diagnostics and uncertainty intervals."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from .utils import validate_probabilities


def wilson_interval(correct, count, z=1.95996398454):
    if count < 1:
        return [None, None]
    fraction = correct / count
    denominator = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / denominator
    margin = z * np.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [float(max(0, center - margin)), float(min(1, center + margin))]


def classification_metrics(labels, probabilities, mapping, evaluated_ids=None):
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or not len(labels) or np.any(labels < 0) or np.any(labels >= len(mapping)):
        raise ValueError("Expected nonempty integer class labels within the mapping.")
    probabilities = validate_probabilities(probabilities, len(mapping), len(labels))
    predicted = probabilities.argmax(axis=1)
    ids = list(range(len(mapping))) if evaluated_ids is None else sorted(set(evaluated_ids))
    if not set(np.unique(labels)).issubset(ids):
        raise ValueError("Metric scope must include every observed truth class.")
    precision, recall, macro_f1, _ = precision_recall_fscore_support(labels, predicted, labels=ids, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(labels, predicted, labels=ids, average="weighted", zero_division=0)
    true_ids = np.unique(labels).tolist()
    _, balanced, _, _ = precision_recall_fscore_support(labels, predicted, labels=true_ids, average="macro", zero_division=0)
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    report = classification_report(labels, predicted, labels=list(range(len(mapping))), target_names=names, output_dict=True, zero_division=0)
    confidence = probabilities.max(axis=1)
    correct = labels == predicted
    ece = 0.
    for index in range(15):
        mask = (confidence >= index / 15) & ((confidence < (index + 1) / 15) if index < 14 else (confidence <= 1))
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    one_hot = np.eye(len(mapping))[labels]
    return {"accuracy": float(accuracy_score(labels, predicted)), "error_rate": float(1 - correct.mean()), "precision_macro": float(precision),
            "recall_macro": float(recall), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
            "balanced_accuracy": float(balanced), "samples": len(labels), "macro_class_ids": ids,
            "classification_report": report, "accuracy_wilson_95": wilson_interval(int(correct.sum()), len(labels)),
            "ece_15_bins": float(ece), "brier_score": float(np.mean(np.sum((probabilities - one_hot)**2, axis=1))),
            "negative_log_likelihood": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)))),
            "confidence_note": "Softmax scores are uncalibrated; these diagnostics do not prove field reliability."}


def rejection_summary(labels, probabilities, profile):
    probabilities = np.asarray(probabilities)
    mask = probabilities.max(axis=1) >= profile["threshold"]
    if profile.get("reject_all"):
        mask[:] = False
    count = int(mask.sum())
    return {"threshold": profile["threshold"], "accepted_images": count,
            "coverage": float(mask.mean()), "rejected_images": int(len(mask) - count),
            "accepted_accuracy": float(np.mean(probabilities.argmax(axis=1)[mask] == np.asarray(labels)[mask])) if count else None,
            "note": "Conditional accuracy among accepted images is separate from unconditional all-image accuracy."}
