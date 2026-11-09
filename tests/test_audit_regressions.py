"""Focused regressions for audit findings; every dataset/model is temporary."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["MPLBACKEND"] = "Agg"
import importlib
from pathlib import Path
import socket
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
import tensorflow as tf
from PIL import Image
from gradio_client import Client, handle_file
from src.acquisition import extract_dataset_zip
from src.app import create_app
from src.bootstrap import write_colab_constraints
from src.bootstrap import validate_checkout
from src.bootstrap import record_experiment
from src.config import ProjectConfig, ProjectPaths
from src.dataset import prepare_dataset, load_manifest
from src.evaluation import compare_models, evaluate_architecture
from src.gradcam import gradcam
from src.prediction import PlantDiseasePredictor
from src.preprocessing import make_dataset, resize_rgb
from src.training import PersistentEpoch, restore_best_alias, train_architecture
from src.utils import atomic_save_model, file_hash, read_json, validate_probabilities, write_json
from test_cpu_smoke import tiny_model


def fixture(paths, config):
    rng = np.random.default_rng(50)
    for crop in config.crops:
        folder = paths.dataset / f"{crop}___healthy"
        folder.mkdir(parents=True)
        for index in range(6):
            Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(folder / f"{index}.png")
    return prepare_dataset(paths, config)


class AuditRegressions(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_import_every_module_and_preserve_stack_constraints(self):
        for path in Path("src").glob("*.py"):
            importlib.import_module("src." + path.stem)
        with tempfile.TemporaryDirectory() as folder:
            constraints = Path(folder) / "constraints.txt"
            pinned = write_colab_constraints(constraints)
            self.assertEqual(pinned["tensorflow"], tf.__version__)
            self.assertIn("numpy==", constraints.read_text())
            paths = ProjectPaths.create(Path(folder) / "data", Path(folder) / "out").initialize()
            original = record_experiment(paths, ProjectConfig(), "https://github.com/example/fixture.git", "synthetic-commit")
            self.assertEqual(record_experiment(paths, ProjectConfig(), "https://github.com/example/fixture.git", "synthetic-commit"), original)
            with self.assertRaisesRegex(ValueError, "provenance"):
                record_experiment(paths, ProjectConfig(), "https://github.com/example/fixture.git", "changed-commit")

    def test_invalid_config_and_probability_contract(self):
        for values in ({"train_fraction": float("nan")}, {"head_learning_rate": float("inf")},
                       {"fine_tune_learning_rate": .01}, {"dropout": 1}, {"seed": True},
                       {"batch_size": 1.5}, {"image_size": (31, 224)}, {"image_size": (224,)},
                       {"image_size": None}, {"crops": ("Rice",)}, {"architectures": ("unknown",)}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ProjectConfig(**values)
        for values in ([[np.nan, 1]], [[.9, .9]], [[1.1, -.1]], [[[.5, .5]]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_probabilities(values, 2, 1)
        np.testing.assert_equal(validate_probabilities([[.3, .7]], 2, 1), [[.3, .7]])

    def test_gradcam_matches_an_analytic_positive_and_negative_class(self):
        inputs = tf.keras.Input((32, 32, 3))
        x = tf.keras.layers.Rescaling(1 / 255)(inputs)
        conv = tf.keras.layers.Conv2D(1, 1, use_bias=False, name="known_conv")
        features = conv(x)
        backbone = tf.keras.Model(inputs, features)
        outer = tf.keras.Input((32, 32, 3))
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(backbone(outer))
        x = tf.keras.layers.Dropout(0, name="head_dropout")(x)
        head = tf.keras.layers.Dense(2, name="logits", dtype="float32")
        x = head(x)
        model = tf.keras.Model(outer, tf.keras.layers.Activation("softmax", dtype="float32")(x))
        conv.set_weights([np.array([[[[1.], [0.], [0.]]]], np.float32)])
        head.set_weights([np.array([[1., -1.]], np.float32), np.zeros(2, np.float32)])
        image = np.zeros((32, 32, 3), np.uint8)
        image[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] + 1
        positive = gradcam(model, image, 0)
        np.testing.assert_allclose(positive["values"], image[:, :, 0] / 32, atol=1e-6)
        self.assertTrue(gradcam(model, image, 1)["zero_map"])
        self.assertEqual(gradcam(model, image)["class_index"], 0)

    def test_checkout_validation_with_temporary_git_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT, text=True)
            git("init", "-b", "main")
            git("config", "user.name", "Audit Fixture")
            git("config", "user.email", "audit@example.invalid")
            (root / "src").mkdir()
            (root / "src" / "__init__.py").write_text("# synthetic checkout fixture\n")
            git("add", ".")
            git("commit", "-m", "Synthetic fixture")
            url = "https://github.com/example/agrivision-fixture.git"
            git("remote", "add", "origin", url)
            with patch("src.__file__", str(root / "src" / "__init__.py")):
                self.assertEqual(validate_checkout(root, url), git("rev-parse", "HEAD").strip())
                with self.assertRaisesRegex(ValueError, "another remote"):
                    validate_checkout(root, "https://github.com/example/other.git")
                (root / "uncommitted.txt").write_text("synthetic edit")
                with self.assertRaisesRegex(ValueError, "local changes"):
                    validate_checkout(root, url)

    def test_manifest_integrity_labels_paths_and_split_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ProjectConfig(image_size=(32, 32))
            paths = ProjectPaths.create(root / "data", root / "outputs").initialize()
            frame, mapping, _ = fixture(paths, config)
            metadata = paths.output / "dataset"
            original_manifest = (metadata / "manifest.csv").read_bytes()
            original_report = (metadata / "report.json").read_bytes()
            original_mapping = (metadata / "class_indices.json").read_bytes()
            changed = dict(mapping)
            keys = list(changed)
            changed[keys[0]], changed[keys[1]] = changed[keys[1]], changed[keys[0]]
            write_json(metadata / "class_indices.json", changed)
            report = read_json(metadata / "report.json")
            report["class_indices_sha256"] = file_hash(metadata / "class_indices.json")
            write_json(metadata / "report.json", report)
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                load_manifest(paths)
            (metadata / "class_indices.json").write_bytes(original_mapping)
            (metadata / "report.json").write_bytes(original_report)
            for key, value in (("path", "../outside.png"), ("path", "C:/outside.png"), ("split", "external"), ("label", 999)):
                damaged = frame.copy()
                damaged.loc[0, key] = value
                damaged.to_csv(metadata / "manifest.csv", index=False)
                report = read_json(metadata / "report.json")
                report["manifest_sha256"] = file_hash(metadata / "manifest.csv")
                write_json(metadata / "report.json", report)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_manifest(paths)
                (metadata / "manifest.csv").write_bytes(original_manifest)
                (metadata / "report.json").write_bytes(original_report)
            with self.assertRaisesRegex(ValueError, "split settings"):
                load_manifest(paths, config=ProjectConfig(image_size=(32, 32), seed=43))

    def test_validation_preprocessing_is_deterministic_and_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ProjectConfig(image_size=(32, 32), batch_size=4)
            paths = ProjectPaths.create(root / "data", root / "outputs").initialize()
            frame, _, _ = fixture(paths, config)
            subset = frame[frame.split == "validation"]
            first = list(make_dataset(subset, paths.dataset, config).as_numpy_iterator())
            second = list(make_dataset(subset, paths.dataset, config).as_numpy_iterator())
            for (a, y), (b, z) in zip(first, second):
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(y, z)
            np.testing.assert_array_equal(first[0][0][0], resize_rgb(paths.dataset / subset.iloc[0].path, config.image_size))
            self.assertGreater(first[0][0].max(), 1)

    def test_atomic_checkpoint_and_failed_state_commit_preserve_best(self):
        class BrokenSave:
            def save(self, destination):
                Path(destination).write_bytes(b"partial")
                raise OSError("simulated interrupted write")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            best = folder / "best.keras"
            best.write_bytes(b"old committed model")
            with self.assertRaises(OSError):
                atomic_save_model(BrokenSave(), best)
            self.assertEqual(best.read_bytes(), b"old committed model")
            model = tiny_model("EfficientNetB0", 4, ProjectConfig(image_size=(32, 32)))
            from src.models import compile_model
            compile_model(model, .001)
            model.train_on_batch(np.zeros((1, 32, 32, 3)), np.array([0]))
            old = folder / "resume_head_0001_original.keras"
            model.save(old)
            state = {"stage": "head", "next_epoch": 1, "best_checkpoint": old.name,
                     "latest_checkpoint": old.name, "best_val_loss": 2., "history": []}
            write_json(folder / "state.json", state)
            restore_best_alias(folder, state)
            digest = file_hash(best)
            callback = PersistentEpoch(folder, state, "head")
            callback.set_model(model)
            callback.on_epoch_begin(1)
            with patch("src.training.write_json", side_effect=OSError("state commit interrupted")):
                with self.assertRaises(OSError):
                    callback.on_epoch_end(1, {"loss": 1., "val_loss": 1.})
            self.assertEqual(read_json(folder / "state.json")["next_epoch"], 1)
            self.assertEqual(state["next_epoch"], 1)
            self.assertEqual(file_hash(best), digest)
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                callback.on_epoch_end(1, {"loss": np.nan, "val_loss": 1.})
            self.assertEqual(file_hash(best), digest)
            best.write_bytes(b"broken alias")
            restore_best_alias(folder, state)
            self.assertEqual(file_hash(best), file_hash(old))

    def test_archive_interruption_and_cross_platform_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(("../escape", "C:/escape", "..\\escape")):
                archive = root / f"bad{index}.zip"
                with ZipFile(archive, "w") as zipped:
                    zipped.writestr(name, "synthetic")
                with self.assertRaises(ValueError):
                    extract_dataset_zip(archive, root / f"bad{index}")
            archive = root / "good.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("folder/fixture.txt", "synthetic")
            destination = root / "extracted"
            with patch.object(ZipFile, "extractall", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    extract_dataset_zip(archive, destination)
            self.assertFalse(destination.exists())
            extract_dataset_zip(archive, destination)
            self.assertEqual((destination / "folder" / "fixture.txt").read_text(), "synthetic")

    def test_artifact_tampering_comparison_and_http_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ProjectConfig(image_size=(32, 32), batch_size=4, head_epochs=1, fine_tune_epochs=1)
            paths = ProjectPaths.create(root / "data", root / "outputs").initialize()
            frame, _, _ = fixture(paths, config)
            with self.assertRaises(FileNotFoundError):
                PlantDiseasePredictor(paths, config.architectures[0])
            with self.assertRaises(FileNotFoundError):
                compare_models(paths, config.architectures)
            with patch("src.training.build_model", side_effect=tiny_model):
                for architecture in config.architectures:
                    train_architecture(architecture, paths, config)
                    evaluate_architecture(architecture, paths, config)
                    if architecture == config.architectures[0]:
                        with self.assertRaisesRegex(FileNotFoundError, "every requested model"):
                            compare_models(paths, config.architectures)
            self.assertEqual(len(compare_models(paths, config.architectures)), 2)
            with patch("src.training.training_source_hash", return_value="changed source"):
                with self.assertRaisesRegex(ValueError, "source or manifest changed"):
                    train_architecture(config.architectures[0], paths, config)
            model_folder = paths.output / "models" / config.architectures[0]
            config_bytes = (model_folder / "config.json").read_bytes()
            changed_config = read_json(model_folder / "config.json")
            changed_config["confidence_threshold"] = .5
            write_json(model_folder / "config.json", changed_config)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                PlantDiseasePredictor(paths, config.architectures[0])
            (model_folder / "config.json").write_bytes(config_bytes)
            predictor = PlantDiseasePredictor(paths, config.architectures[0])
            metrics_path = predictor.metrics_path
            metrics_bytes = metrics_path.read_bytes()
            metrics_path.write_text("invalid JSON")
            self.assertIn("Evaluation not yet available", predictor.evaluation_metrics())
            metrics_path.write_bytes(metrics_bytes)
            stale = read_json(metrics_path)
            stale["model_sha256"] = "stale fixture"
            write_json(metrics_path, stale)
            with self.assertRaisesRegex(ValueError, "different model/dataset"):
                compare_models(paths, config.architectures)
            metrics_path.write_bytes(metrics_bytes)
            invalid_image = root / "invalid.png"
            invalid_image.write_bytes(b"not an image")
            with self.assertRaises(OSError):
                predictor.predict_plant_disease(invalid_image)
            demo = create_app(paths, config)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            try:
                demo.launch(server_name="127.0.0.1", server_port=port, share=False,
                            inbrowser=False, prevent_thread_lock=True, quiet=True)
                client = Client(f"http://127.0.0.1:{port}", verbose=False)
                outputs = client.predict(handle_file(str(paths.dataset / frame.iloc[0].path)), config.architectures[0], .7, api_name="/analyze")
                self.assertEqual(len(outputs), 10)
                self.assertIn("Crop:", outputs[0])
                for image_output in outputs[3:6]:
                    self.assertTrue(Path(image_output).is_file())
            finally:
                demo.close()
