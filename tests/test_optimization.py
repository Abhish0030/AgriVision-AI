"""Synthetic software checks; no PlantVillage/PlantDoc performance is measured."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from src.augmentation import build_augmentation
from src.config import ProjectConfig, ProjectPaths
from src.experiments import build_trial_plan, run_trials, select_best_trials, export_selected_models, evaluate_selected_models, generate_final_comparison
from src.external import prepare_external_dataset, load_external_manifest, evaluate_external
from src.error_analysis import analyze_external_errors
from src.gradcam import save_error_examples
from src.metrics import classification_metrics, wilson_interval
from src.models import MacroF1, SparseSmoothedCrossentropy, compile_model
from src.prediction import PlantDiseasePredictor
from src.utils import file_hash
from src.validation import fit_rejection_threshold
from test_audit_regressions import fixture
from test_cpu_smoke import tiny_model


class OptimizationChecks(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.keras.utils.set_random_seed(42)

    def test_metrics_and_rejection_do_not_inflate_all_image_accuracy(self):
        labels = np.array([0, 1])
        probabilities = np.array([[.9, .1], [.6, .4]])
        result = classification_metrics(labels, probabilities, {"Apple___healthy": 0, "Tomato___healthy": 1})
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["balanced_accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertAlmostEqual(result["ece_15_bins"], .35)
        lower, upper = wilson_interval(1, 2)
        self.assertLess(lower, .5)
        self.assertGreater(upper, .5)
        profile, _ = fit_rejection_threshold(labels, probabilities, .9, 30)
        self.assertTrue(profile["reject_all"])
        high_quality = np.tile([.99, .01], (100, 1))
        profile, _ = fit_rejection_threshold(np.zeros(100, int), high_quality, .9, 30)
        self.assertFalse(profile["reject_all"])

    def test_profiles_adamw_smoothed_loss_and_metric_reload(self):
        batch = np.full((2, 32, 32, 3), 120., np.float32)
        for profile in ("none", "mild", "robust"):
            augment = build_augmentation(42, profile)
            transformed = augment(batch, training=True).numpy()
            self.assertTrue(np.isfinite(transformed).all())
            self.assertGreaterEqual(transformed.min(), 0)
            self.assertLessEqual(transformed.max(), 255)
            if profile == "none":
                np.testing.assert_array_equal(transformed, batch)
        metric = MacroF1(2)
        metric.update_state([0, 1], [[.9, .1], [.6, .4]])
        self.assertAlmostEqual(float(metric.result()), 1 / 3, places=6)
        loss = SparseSmoothedCrossentropy(.1)
        self.assertAlmostEqual(float(loss([0], [[.9, .1]])), -.95 * np.log(.9) - .05 * np.log(.1), places=6)
        config = ProjectConfig(image_size=(32, 32), optimizer="AdamW", label_smoothing=.05)
        model = tiny_model("EfficientNetB0", 4, config)
        compile_model(model, .001, config)
        self.assertTrue(np.isfinite(model.train_on_batch(batch, np.array([0, 1]))).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "smoothed.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            self.assertEqual(type(restored.optimizer).__name__, "AdamW")
            self.assertTrue(np.isfinite(restored.train_on_batch(batch, np.array([0, 1]))).all())

    def test_validation_selection_exports_and_independent_external_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            master = ProjectPaths.create(root / "plantvillage_fixture", root / "outputs").initialize()
            config = ProjectConfig(image_size=(32, 32), batch_size=8, head_epochs=1, fine_tune_epochs=1,
                                   rejection_min_samples=2, rejection_target_precision=.5)
            frame, _, report = fixture(master, config)
            original_manifest = file_hash(master.output / "dataset" / "manifest.csv")
            trials = build_trial_plan(config)
            self.assertEqual(len(trials), 6)
            self.assertEqual(len(build_trial_plan(config, True, True, True)), 12)
            with patch("src.training.build_model", side_effect=tiny_model):
                with patch("src.experiments.evaluate_architecture", side_effect=AssertionError("Test inference before selection")):
                    table = run_trials(master, config, trials)
            self.assertEqual(len(table), 6)
            self.assertFalse((master.output / "evaluation" / "EfficientNetB0.json").exists())
            selection = select_best_trials(master)
            self.assertTrue(selection["validation_only"])
            for architecture, winner in selection["winners"].items():
                matching = table[table.model == architecture]
                self.assertAlmostEqual(winner["validation_macro_f1"], matching.macro_f1.max())
            with self.assertRaisesRegex(ValueError, "Selection is frozen"):
                run_trials(master, config, trials)
            export_selected_models(master)
            self.assertTrue((master.output / "models" / "efficientnet_best.keras").is_file())
            self.assertTrue((master.output / "models" / "mobilenet_best.keras").is_file())
            result = evaluate_selected_models(master)
            self.assertIn("Not evaluated", result.to_numpy())
            self.assertTrue((master.output / "evaluation" / "EfficientNetB0_training.png").exists())
            self.assertTrue((master.output / "evaluation" / "EfficientNetB0_confusion.csv").exists())
            external_root = root / "external_fixture"
            rng = np.random.default_rng(123)
            for label in ("field apple", "field tomato", "unsupported"):
                destination = external_root / label
                destination.mkdir(parents=True)
                for index in range(3):
                    Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(destination / f"{index}.png")
            (external_root / "field apple" / "duplicate.png").write_bytes((external_root / "field apple" / "0.png").read_bytes())
            pv_apple = frame[frame.class_name == "Apple___healthy"].iloc[0]
            (external_root / "field apple" / "pv_overlap.png").write_bytes((master.dataset / pv_apple.path).read_bytes())
            spec = {"dataset_name": "synthetic fixture", "source_url": "https://example.invalid/synthetic", "reviewed": False,
                    "labels": {"field apple": "Apple___healthy", "field tomato": "Tomato___healthy", "unsupported": None}}
            with self.assertRaisesRegex(ValueError, "Review the explicit"):
                prepare_external_dataset(master, external_root, spec)
            spec["reviewed"] = True
            external, audit = prepare_external_dataset(master, external_root, spec)
            self.assertEqual(len(external), 6)
            self.assertEqual(audit["unsupported_classes"]["unsupported"], 3)
            self.assertEqual(audit["excluded_reasons"]["plantvillage_exact_overlap"], 1)
            self.assertEqual(audit["excluded_reasons"]["external_exact_or_near_duplicate"], 1)
            load_external_manifest(master, external_root)
            for architecture in config.architectures:
                result = evaluate_external(architecture, master, external_root)
                self.assertEqual(result["samples"], 6)
                self.assertEqual(len(result["macro_class_ids"]), 2)
                self.assertFalse(result["per_class_support_sufficient"])
                self.assertIn("Not established", result["target_status"])
                predictor = PlantDiseasePredictor(master, architecture)
                prediction = predictor.predict_plant_disease(external_root / external.iloc[0].path)
                self.assertEqual(prediction["threshold_source"], "validation")
                self.assertIn(":", prediction["model_version"])
                self.assertIn("external_evaluation", predictor.evaluation_metrics())
                self.assertIn("pending", analyze_external_errors(master, architecture)["status"])
                annotations = external[["path"]].copy()
                annotations["lighting"] = "synthetic_test_group"
                annotation_path = root / "annotations.csv"
                annotations.to_csv(annotation_path, index=False)
                self.assertEqual(analyze_external_errors(master, architecture, annotation_path)["groups"][0]["images"], 6)
                predictions = pd.read_csv(master.output / "evaluation" / f"{architecture}_external_predictions.csv")
                examples = save_error_examples(predictor.model, predictions, external_root, root / "cams" / architecture, per_outcome=1)
                self.assertTrue(examples)
                self.assertTrue(all(Path(record["overlay"]).exists() for record in examples))
            self.assertEqual(file_hash(master.output / "dataset" / "manifest.csv"), original_manifest)
            self.assertEqual(report["manifest_sha256"], original_manifest)
            self.assertNotIn("Not evaluated", generate_final_comparison(master).loc["External Test Accuracy"].to_numpy())
            (external_root / external.iloc[0].path).write_bytes(b"changed external image")
            with self.assertRaises(ValueError):
                load_external_manifest(master, external_root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
