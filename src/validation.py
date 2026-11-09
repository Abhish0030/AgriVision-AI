"""Validation-only trial scoring and threshold fitting; never reads test images."""
import numpy as np
import pandas as pd
import tensorflow as tf
from .artifacts import model_artifacts, validate_model_contract
from .dataset import load_manifest
from .metrics import classification_metrics, wilson_interval
from .preprocessing import make_dataset
from .utils import read_json, validate_probabilities, write_json


def fit_rejection_threshold(labels, probabilities, target_precision=.9, minimum_samples=30):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(labels)
    curve = []
    for threshold in np.unique(np.r_[np.linspace(0, 1, 101), confidence]):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        hits = int(correct[accepted].sum())
        bounds = wilson_interval(hits, count)
        curve.append({"threshold": float(threshold), "accepted_images": count, "coverage": float(accepted.mean()),
                      "accepted_accuracy": hits / count if count else None, "wilson_lower_95": bounds[0]})
    eligible = [row for row in curve if row["accepted_images"] >= minimum_samples and row["wilson_lower_95"] >= target_precision]
    selected = min(eligible, key=lambda row: row["threshold"]) if eligible else None
    return {"threshold": selected["threshold"] if selected else 1., "reject_all": selected is None,
            "status": "validation_operating_point" if selected else "no_qualifying_threshold",
            "target_precision": target_precision, "minimum_samples": minimum_samples,
            "coverage_on_validation": selected["coverage"] if selected else 0.,
            "note": "Validation-selected heuristic; multiple-threshold screening and model selection make this no guarantee of future precision."}, curve


def score_validation(architecture, paths, config):
    frame, mapping, report = load_manifest(paths, verify_files=True, config=config)
    model_path, saved_config, saved_mapping, state, digest = model_artifacts(paths, architecture)
    if config != saved_config or mapping != saved_mapping:
        raise ValueError("Validation model configuration/labels differ from frozen data.")
    model = tf.keras.models.load_model(model_path, compile=False)
    validate_model_contract(model, architecture, mapping, config)
    validation = frame[frame.split == "validation"]
    probabilities = validate_probabilities(model.predict(make_dataset(validation, paths.dataset, config), verbose=0), len(mapping), len(validation))
    result = classification_metrics(validation.label.to_numpy(), probabilities, mapping)
    result.update(model=architecture, model_sha256=digest, manifest_sha256=report["manifest_sha256"], split="validation", training_seconds=state["training_seconds"])
    folder = paths.output / "models" / architecture
    profile, curve = fit_rejection_threshold(validation.label.to_numpy(), probabilities,
                                             config.rejection_target_precision, config.rejection_min_samples)
    profile.update(model_sha256=digest, manifest_sha256=report["manifest_sha256"], split="validation")
    write_json(folder / "rejection.json", profile)
    pd.DataFrame(curve).to_csv(paths.output / "evaluation" / f"{architecture}_rejection_curve.csv", index=False)
    write_json(paths.output / "evaluation" / f"{architecture}_validation.json", result)
    return result


def load_rejection_profile(folder, digest, manifest_digest):
    path = folder / "rejection.json"
    if not path.exists():
        return None
    profile = read_json(path)
    if profile.get("model_sha256") != digest or profile.get("manifest_sha256") != manifest_digest or profile.get("split") != "validation":
        raise ValueError("Rejection profile does not belong to this model's validation split.")
    if not 0 <= profile.get("threshold", -1) <= 1 or type(profile.get("reject_all")) is not bool:
        raise ValueError("Invalid saved rejection profile.")
    return profile
