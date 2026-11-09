"""Optional network check: actual ImageNet weights, forward pass and Grad-CAM.

No dataset is downloaded and no disease-training accuracy is measured. Cached
application weights and results are kept under ignored outputs/audit-cache.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["KERAS_HOME"] = str(ROOT / "outputs" / "audit-cache")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import tensorflow as tf
    from src.config import ProjectConfig
    from src.gradcam import gradcam
    from src.models import build_model, enable_fine_tuning, get_backbone
    from src.utils import validate_probabilities, write_json
    config = ProjectConfig()
    image = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    records = []
    for architecture in config.architectures:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(config.seed)
        try:
            model = build_model(architecture, 3, config)  # Production default: ImageNet.
            validate_probabilities(model(image[None].astype(np.float32), training=False).numpy(), 3, 1)
            cam = gradcam(model, image, class_index=1)
            assert cam["overlay"].shape == image.shape and np.isfinite(cam["values"]).all()
            enable_fine_tuning(model, config.fine_tune_layers)
            backbone = get_backbone(model)
            assert any(layer.trainable and layer.weights for layer in backbone.layers)
            assert all(not layer.trainable for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization))
            records.append({"architecture": architecture, "status": "PASS", "weights": "imagenet",
                            "forward_shape": list(model.output_shape), "output_dtype": str(model.output.dtype),
                            "gradcam_layer": cam["layer"], "heatmap_shape": list(cam["values"].shape),
                            "zero_positive_map": cam["zero_map"], "batchnorm_frozen": True})
            print(architecture, "ImageNet loading / forward / Grad-CAM / BatchNorm checks: PASS")
        except Exception as exc:
            records.append({"architecture": architecture, "status": "FAIL", "error": str(exc)})
            print(architecture, "FAIL:", exc)
    write_json(ROOT / "outputs" / "audit-pretrained.json", {"tensorflow": tf.__version__,
               "keras": tf.keras.__version__, "device": "CPU", "disease_training_performed": False, "checks": records})
    if any(record["status"] != "PASS" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
