# Development validation

Validated locally on **10 November 2026** using Windows CPU execution in an isolated
`.venv`: Python 3.13.7, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.2.6 and Gradio
5.50.0. This is one tested environment, not a guarantee for every version allowed
by dependency ranges.

## Executed checks

- `python scripts/check_static.py`: passed. Parses all project Python files and every
  notebook code cell; validated notebook schema and 21 disease-information records.
  The notebook has no stored executions or outputs.
- `python -m ruff check .`: passed.
- `python -m pip check`: passed, no broken installed requirements.
- `python -m src.cli environment --output outputs/development-check`: passed;
  executed a synchronized CPU operation, selected float32 and saved actual
  environment/dependency records. TensorFlow reported no usable GPU in this
  native Windows environment.
- `python -m unittest discover -s tests -v`: **12 tests passed** in 44.009 seconds
  in the final audit run. This expands the original three-test development suite.
- `python scripts/audit_pretrained.py`: both real architectures loaded ImageNet
  weights and passed CPU forward/Grad-CAM/BatchNorm checks at 224×224. The head
  remains untrained. EfficientNet selected top_conv; MobileNetV3Large selected conv_1.
- Constrained dependency installer: executed successfully against the local CPU
  environment, preserving its TensorFlow stack and passing pip check. This is not
  execution inside a real Colab runtime.
- Git ignore checks confirmed `.venv`, dataset examples, output logs, `.env` and
  `kaggle.json` are excluded. The source-file inventory contains no dataset/model
  artifacts. Only placeholder `.gitkeep` files under data/outputs are eligible
  for source control; local validation logs/records are ignored.

## What the CPU suite actually exercises

1. Configuration save/load and validation; malicious ZIP traversal rejection.
2. Synthetic RGB files in temporary folders for all four crops; corruption
   detection, exact-copy removal, deliberate near-match grouping and frozen-split
   integrity. A tiny CNN fixture exercises both training stages, full-model
   checkpoint commit, simulated interruption/resumption, saved evaluation,
   comparison, confidence warnings and class mapping. The Gradio Blocks
   configuration and prediction callback are exercised. Changed input files
   are rejected by the frozen-manifest verifier.
3. The **real EfficientNetB0 and MobileNetV3Large architectures**, initialized
   with `weights=None`, each run a CPU forward pass, a head-training update,
   float32/probability checks, Grad-CAM, full-model save/load with optimizer
   iteration recovery, post-load Grad-CAM and a fine-tuning update.
4. Audit regressions cover NaN/range/type config errors; invalid probability
   tensors; inconsistent mappings/labels; path traversal; changed split settings;
   deterministic identical resize for validation/inference; interrupted model and
   state writes; best-alias recovery; nonfinite-epoch rejection; incomplete ZIP
   extraction; changed model config/source and stale/corrupt evaluation.
5. An analytic convolution fixture verifies Grad-CAM against a known expected
   positive heatmap and negative-class zero map. Temporary Git fixtures exercise
   correct/wrong/dirty checkout checks. Original commit/config provenance cannot
   be silently overwritten.
6. A real localhost Gradio server was launched without a browser or public share
   link. A Gradio client uploaded a temporary image and retrieved a successful
   prediction plus original, heatmap and overlay files. The server was then closed.

Synthetic fixtures are software tests. Their scores are not AgriVision research
results and are never published into project evaluation artifacts. Temporary
model/data files are removed when tests finish. The local test log is ignored
under `outputs/cpu-smoke.log`.

The first run caught an invalid Gradio JSON status value and a Windows plotting
backend issue. These were fixed: status uses a structured JSON object and figures
render with the Agg backend. Later runs passed. Third-party TensorFlow/Keras and
Gradio deprecation/resource warnings remain; Gradio is bounded below version 6.

## Not verified during development

- Google Colab session execution, Drive authorization/storage, NVIDIA T4/CUDA
  execution, mixed-float16 GPU training or disconnection behavior on Drive itself.
- Real PlantVillage acquisition, dataset counts, model convergence or accuracy.
- Interactive browser visual layout or a Colab-hosted/public Gradio share link.
- PlantDoc evaluation, calibration or universal agricultural-field generalization.

Follow COLAB_GUIDE.md and collect actual execution evidence for these steps.
Do not report a passing CPU software check as verified GPU training or plant
disease performance.

See AUDIT_INITIAL.md for pre-fix findings, AUDIT_FINAL.md for requirement statuses,
and SOURCE_VERIFICATION.md for the online disease-reference review. No credentials,
datasets, cached application weights, test logs or evaluation fixtures are staged
for Git publication.
