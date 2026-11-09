"""Validate model, configuration and label provenance before using predictions."""
from dataclasses import asdict
import json
import math
from .config import ProjectConfig, SUPPORTED_ARCHITECTURES
from .utils import file_hash, read_json, training_source_hash, validate_mapping


def model_artifacts(paths, architecture):
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Unknown model architecture.")
    folder = paths.output / "models" / architecture
    model_path = folder / "best.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"No trained project checkpoint: {model_path}. Complete training first.")
    try:
        state = read_json(folder / "state.json")
        mapping = read_json(folder / "class_indices.json")
        config = ProjectConfig.load(folder / "config.json")
        signature = state["signature"]
        validate_mapping(mapping)
        if not state.get("complete") or set(state.get("finished_stages", [])) != {"head", "fine_tune"}:
            raise ValueError("Project training has not completed both stages.")
        if signature["architecture"] != architecture or mapping != signature["class_indices"]:
            raise ValueError("Model architecture/class mapping differs from its training record.")
        if json.loads(json.dumps(asdict(config))) != signature["config"]:
            raise ValueError("Saved configuration differs from its training record.")
        if signature.get("implementation_sha256") != training_source_hash():
            raise ValueError("Training source differs from the saved experiment. Use its recorded source version.")
        digest = file_hash(model_path)
        if digest != state["best_model_sha256"]:
            raise ValueError("Checkpoint differs from its completed training record.")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid model metadata. Restore the completed experiment files.") from exc
    return model_path, config, mapping, state, digest


def validate_model_contract(model, architecture, mapping, config):
    if model.name != architecture or tuple(model.input_shape[1:]) != (*config.image_size, 3):
        raise ValueError("Loaded model architecture or RGB input shape differs from metadata.")
    if model.output_shape[-1] != len(mapping) or str(model.output.dtype) != "float32":
        raise ValueError("Loaded model output dimension/dtype differs from the class contract.")


def validate_evaluation_record(record, architecture, model_digest, manifest_digest):
    if not isinstance(record, dict) or record.get("model") != architecture:
        raise ValueError("Evaluation model identity is invalid.")
    if record.get("model_sha256") != model_digest or record.get("manifest_sha256") != manifest_digest:
        raise ValueError("Evaluation belongs to different model/dataset artifacts.")
    for key in ("accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid saved metric: {key}.")
    for key in ("training_seconds", "inference_ms_median", "inference_ms_p95"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid saved timing: {key}.")
    if type(record.get("test_images")) is not int or record["test_images"] < 1:
        raise ValueError("Evaluation must record a positive test sample count.")
