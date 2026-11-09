"""Inference only from project checkpoints, with explicit artifact provenance."""
import numpy as np
import tensorflow as tf
from .artifacts import model_artifacts, validate_evaluation_record, validate_model_contract
from .disease_information import class_display, get_disease_information
from .gradcam import gradcam
from .preprocessing import read_rgb, resize_rgb
from .utils import read_json, validate_probabilities


class PlantDiseasePredictor:
    def __init__(self, paths, architecture):
        self.model_path, self.config, mapping, state, self.model_hash = model_artifacts(paths, architecture)
        self.classes = [name for name, _ in sorted(mapping.items(), key=lambda x: x[1])]
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Class indices must be contiguous.")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        validate_model_contract(self.model, architecture, mapping, self.config)
        self.metrics_path = paths.output / "evaluation" / f"{architecture}.json"
        self.manifest_hash = state["signature"]["manifest_sha256"]
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
        return metrics

    def predict_plant_disease(self, image, confidence_threshold=None):
        original = read_rgb(image)
        threshold = self.config.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        if not 0 < threshold <= 1:
            raise ValueError("Confidence threshold must be in (0,1].")
        probabilities = validate_probabilities(self.model(resize_rgb(original, self.config.image_size)[None], training=False).numpy(),
                                               len(self.classes), 1)[0]
        indices = np.argsort(probabilities)[::-1][:3]
        label = self.classes[int(indices[0])]
        crop, disease = class_display(label)
        confidence = float(probabilities[indices[0]])
        warning = ("Uncertain prediction. Please upload a clearer image or consult an agricultural expert."
                   if confidence < threshold else "")
        visualization = gradcam(self.model, original, int(indices[0]))
        return {"class_name": label, "crop": crop, "disease": disease, "confidence_percent": confidence * 100,
                "top_three": [{"class": self.classes[int(i)], "probability": float(probabilities[i])} for i in indices],
                "health_status": "Predicted healthy" if label.split("___")[-1].lower() == "healthy" else "Predicted affected",
                "uncertainty_warning": warning, "gradcam": visualization,
                "information": get_disease_information(label, self.info_path),
                "limitation": "A high softmax score does not prove reliability. Non-leaf and unfamiliar images may still receive high scores."}


def predict_plant_disease(image, predictor, confidence_threshold=None):
    return predictor.predict_plant_disease(image, confidence_threshold)
