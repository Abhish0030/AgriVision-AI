"""Synthetic temporary fixtures test software only, never research performance."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["MPLBACKEND"] = "Agg"
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
from PIL import Image
import tensorflow as tf
from src.acquisition import extract_dataset_zip
from src.config import ProjectConfig, ProjectPaths
from src.dataset import prepare_dataset, load_manifest, difference_hash
from src.gradcam import gradcam
from src.models import build_model, compile_model, enable_fine_tuning
from src.prediction import PlantDiseasePredictor
from src.preprocessing import resize_rgb
from src.training import train_architecture, PersistentEpoch
from src.evaluation import evaluate_architecture, compare_models
from src.utils import read_json


def tiny_model(architecture, classes, config):
    """Cheap integration fixture. Real backbones are independently tested below."""
    inputs = tf.keras.Input((*config.image_size, 3))
    x = tf.keras.layers.Rescaling(1 / 255)(inputs)
    x = tf.keras.layers.Conv2D(4, 3, activation="relu")(x)
    backbone = tf.keras.Model(inputs, x, name="tiny_backbone")
    backbone.trainable = False
    outer = tf.keras.Input((*config.image_size, 3))
    x = backbone(outer, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="head_dropout")(x)
    x = tf.keras.layers.Dense(classes, dtype="float32", name="logits")(x)
    outputs = tf.keras.layers.Activation("softmax", dtype="float32", name="probabilities")(x)
    return tf.keras.Model(outer, outputs, name=architecture)


class CPUChecks(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(42)

    def test_real_backbones_gradcam_training_and_reload(self):
        config = ProjectConfig(image_size=(64, 64), batch_size=1)
        image = np.random.default_rng(42).integers(0, 256, (77, 61, 3), dtype=np.uint8)
        for architecture in config.architectures:
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as folder:
                model = build_model(architecture, 3, config, weights=None)  # No network download.
                compile_model(model, .001)
                batch = resize_rgb(image, config.image_size)[None]
                model.train_on_batch(batch, np.array([1]))
                before = model(batch, training=False).numpy()
                self.assertEqual(before.dtype, np.float32)
                np.testing.assert_allclose(before.sum(1), 1, atol=1e-5)
                cam = gradcam(model, image, 1)
                self.assertEqual(cam["overlay"].shape, image.shape)
                self.assertTrue(np.isfinite(cam["values"]).all())
                model.save(Path(folder) / "model.keras")
                restored = tf.keras.models.load_model(Path(folder) / "model.keras")
                self.assertEqual(int(restored.optimizer.iterations), 1)
                np.testing.assert_allclose(restored(batch, training=False), before, rtol=1e-5, atol=1e-6)
                gradcam(restored, image, 1)
                enable_fine_tuning(restored, 5)
                compile_model(restored, 1e-5)
                result = restored.train_on_batch(batch, np.array([1]))
                self.assertTrue(np.isfinite(result).all())
            tf.keras.backend.clear_session()

    def test_dataset_pipeline_and_saved_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ProjectPaths.create(root / "data", root / "outputs").initialize()
            config = ProjectConfig(image_size=(32, 32), batch_size=4, head_epochs=1, fine_tune_epochs=1)
            rng = np.random.default_rng(1)
            for crop in config.crops:
                folder = paths.dataset / f"{crop}___healthy"
                folder.mkdir(parents=True)
                for i in range(8):
                    rgb = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
                    Image.fromarray(rgb).save(folder / f"{i}.png")
                (folder / "exact.png").write_bytes((folder / "0.png").read_bytes())
                first = np.array(Image.open(folder / "0.png"))
                near = np.clip(first.astype(np.int16) + 1, 0, 255).astype(np.uint8)
                self.assertLessEqual((difference_hash(first) ^ difference_hash(near)).bit_count(), config.near_duplicate_distance)
                Image.fromarray(near).save(folder / "near.png")
                (folder / "broken.png").write_bytes(b"invalid fixture")
            frame, mapping, report = prepare_dataset(paths, config)
            self.assertEqual(report["corrupted"], 4)
            self.assertEqual(report["exact_duplicates_removed"], 4)
            self.assertGreaterEqual(report["near_duplicate_pairs"], 4)
            self.assertTrue((frame.groupby("group").split.nunique() == 1).all())
            self.assertEqual(set(frame.split), {"train", "validation", "test"})
            with patch("src.training.build_model", side_effect=tiny_model):
                for architecture in config.architectures:
                    if architecture == config.architectures[0]:
                        original_callback = PersistentEpoch.on_epoch_end
                        def disconnect(callback, epoch, logs=None):
                            original_callback(callback, epoch, logs)
                            raise RuntimeError("Simulated disconnection after committed epoch")
                        with patch.object(PersistentEpoch, "on_epoch_end", disconnect):
                            with self.assertRaisesRegex(RuntimeError, "Simulated disconnection"):
                                train_architecture(architecture, paths, config)
                    train_architecture(architecture, paths, config)
                    state = read_json(paths.output / "models" / architecture / "state.json")
                    self.assertEqual(state["finished_stages"], ["head", "fine_tune"])
                    self.assertEqual(len(state["history"]), 2)
                    predictor = PlantDiseasePredictor(paths, architecture)
                    self.assertEqual(predictor.evaluation_metrics(), "Evaluation not yet available")
                    prediction = predictor.predict_plant_disease(paths.dataset / frame.iloc[0].path, 1)
                    self.assertEqual(len(prediction["top_three"]), 3)
                    self.assertTrue(prediction["uncertainty_warning"])
                    evaluate_architecture(architecture, paths, config)
                    self.assertIsInstance(predictor.evaluation_metrics(), dict)
                    train_architecture(architecture, paths, config)  # Completed run is idempotent.
            self.assertEqual(len(compare_models(paths, config.architectures)), 2)
            from src.app import create_app
            demo = create_app(paths, config)
            self.assertTrue(demo.config)
            analyze = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "analyze")
            outputs = analyze(str(paths.dataset / frame.iloc[0].path), config.architectures[0], .7)
            self.assertEqual(len(outputs), 10)
            self.assertIsInstance(outputs[8], dict)
            # Frozen manifest checks detect changed files rather than silently reusing splits.
            (paths.dataset / frame.iloc[0].path).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_manifest(paths, verify_files=True)

    def test_archive_and_configuration_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", "unsafe")
            with self.assertRaises(ValueError):
                extract_dataset_zip(archive, root / "extracted")
            config_path = root / "config.json"
            ProjectConfig().save(config_path)
            self.assertEqual(ProjectConfig.load(config_path), ProjectConfig())
            with self.assertRaises(ValueError):
                ProjectConfig(train_fraction=.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
