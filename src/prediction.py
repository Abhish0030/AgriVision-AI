"""Inference only from project checkpoints, with explicit artifact provenance."""
import numpy as np
import tensorflow as tf
from .artifacts import model_artifacts, validate_evaluation_record, validate_model_contract
from .disease_information import class_display, get_disease_information
from .gradcam import gradcam
from .preprocessing import read_rgb, resize_rgb
from .utils import read_json, validate_probabilities
from .validation import load_rejection_profile


class PlantDiseasePredictor:
    def __init__(self, paths, architecture):
        self.model_path, self.config, mapping, state, self.model_hash = model_artifacts(paths, architecture)
        self.classes = [name for name, _ in sorted(mapping.items(), key=lambda x: x[1])]
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Class indices must be contiguous.")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        validate_model_contract(self.model, architecture, mapping, self.config)
        self.metrics_path = paths.output / "evaluation" / f"{architecture}.json"
        self.external_metrics_path = paths.output / "evaluation" / f"{architecture}_external.json"
        self.external_report_path = paths.output / "external" / "report.json"
        self.manifest_hash = state["signature"]["manifest_sha256"]
        self.version = f"{state.get('exported_from_trial', architecture)}:{self.model_hash[:12]}"
        self.rejection_profile = load_rejection_profile(self.model_path.parent, self.model_hash, self.manifest_hash)
        self.info_path = paths.repository / "disease_info.json"

    def evaluation_metrics(self):
        if not self.metrics_path.exists():
            return "Evaluation not yet available"
        try:
            metrics = read_json(self.metrics_path)
            if not isinstance(metrics, dict):
                raise ValueError("Expected an evaluation object.")
        except (OSError, ValueError):
            return "Evaluation not yet available (saved evaluation is unreadable or invalid)"
        try:
            validate_evaluation_record(metrics, self.model.name, self.model_hash, self.manifest_hash)
        except ValueError:
            return "Evaluation not yet available (saved evaluation is invalid or belongs to different artifacts)"
        if self.external_metrics_path.exists():
            try:
                external = read_json(self.external_metrics_path)
                external_report = read_json(self.external_report_path)
                if not isinstance(external, dict) or not isinstance(external_report, dict):
                    raise ValueError("Invalid external evaluation object.")
                if external.get("model_sha256") != self.model_hash or external.get("external_manifest_sha256") != external_report["manifest_sha256"]:
                    raise ValueError("Stale external metrics.")
                metrics["external_evaluation"] = external
            except (OSError, ValueError, KeyError, TypeError):
                metrics["external_evaluation"] = {"status": "External evaluation not yet available or artifacts do not match"}
        return metrics

    def predict_plant_disease(self, image, confidence_threshold=None, *, explain=True):
        original = read_rgb(image)
        use_validation = confidence_threshold is None and self.rejection_profile is not None
        threshold = (self.rejection_profile["threshold"] if use_validation else
                     self.config.confidence_threshold if confidence_threshold is None else float(confidence_threshold))
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be in [0,1].")
        probabilities = validate_probabilities(self.model(resize_rgb(original, self.config.image_size)[None], training=False).numpy(),
                                               len(self.classes), 1)[0]
        indices = np.argsort(probabilities)[::-1][:3]
        label = self.classes[int(indices[0])]
        crop, disease = class_display(label)
        confidence = float(probabilities[indices[0]])
        warning = ("Uncertain prediction. Please upload a clearer image or consult an agricultural expert."
                   if confidence < threshold or (use_validation and self.rejection_profile["reject_all"]) else "")
        visualization = gradcam(self.model, original, int(indices[0])) if explain else None
        return {"class_name": label, "crop": crop, "disease": disease, "confidence_percent": confidence * 100,
                "model_version": self.version, "rejection_threshold": threshold,
                "threshold_source": "validation" if use_validation else "manual override" if confidence_threshold is not None else "configured heuristic",
                "top_three": [{"class": self.classes[int(i)], "probability": float(probabilities[i])} for i in indices],
                "health_status": "Predicted healthy" if label.split("___")[-1].lower() == "healthy" else "Predicted affected",
                "uncertainty_warning": warning, "gradcam": visualization,
                "information": get_disease_information(label, self.info_path),
                "limitation": "A high softmax score does not prove reliability. Non-leaf and unfamiliar images may still receive high scores."}


def predict_plant_disease(image, predictor, confidence_threshold=None):
    return predictor.predict_plant_disease(image, confidence_threshold)
