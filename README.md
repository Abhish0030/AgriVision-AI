# AgriVision AI

**An Explainable Deep Learning-Based Multi-Crop Plant Disease Detection and Classification System**

TensorFlow/Keras research project for tomato, potato, maize and apple leaf images.
Develop in VS Code, upload source to GitHub, then run the main notebook in Colab.
Datasets, checkpoints and secrets remain outside Git.

Repository: [Abhish0030/AgriVision-AI](https://github.com/Abhish0030/AgriVision-AI).
Open the [main notebook in Google Colab](https://colab.research.google.com/github/Abhish0030/AgriVision-AI/blob/main/notebooks/AgriVision_AI_Complete.ipynb).
The repository's original README is preserved verbatim in
[ORIGINAL_GITHUB_README.md](docs/ORIGINAL_GITHUB_README.md), alongside its original
Git commit history.

## Problem and objectives

Given one supported plant-leaf image, predict its joint crop/disease class and
expose the evidence and uncertainty behind that prediction. Compare two transfer
learning architectures on the same audited split, evaluate every class, and
provide a simple demonstration whose metrics come from actual saved experiments.
The application supports inspection and learning; it does not establish a field
diagnosis from a photograph alone.

## Technology stack

| Area | Tools |
|---|---|
| Language and development | Python, VS Code, Git/GitHub |
| Training | TensorFlow/Keras, EfficientNetB0, MobileNetV3Large |
| Intended execution | Google Colab T4; CPU fallback for development |
| Images and data | Pillow, OpenCV, NumPy, Pandas |
| Evaluation and plots | Scikit-learn, Matplotlib, Seaborn |
| Explanation and demo | GradientTape Grad-CAM, Gradio Blocks |
| Persistent storage | Configurable filesystem paths / Google Drive |

## Dataset and architecture

Supply original PlantVillage color images for Tomato, Potato, Corn (maize) and
Apple. Classes are discovered rather than invented; no rice labels are assumed.
See [dataset acquisition and provenance](docs/COLAB_GUIDE.md#dataset-upload).

```text
PlantVillage → RGB audit + duplicate groups → frozen train/validation/test
             → shared resize + training-only augmentation
             → EfficientNetB0 / MobileNetV3Large
             → global average pooling → dropout → float32 logits/softmax
             → saved checkpoint + evaluation → Grad-CAM/prediction → Gradio
```

## Implemented features

- PlantVillage audit, corrupted-file detection, class/dimension reports, exact
  pixel deduplication and practical dHash near-duplicate grouping.
- Frozen class-stratified group splits, approximately 70/15/15. Duplicate groups
  stay in one split; conflicting near-match labels are quarantined.
- EfficientNetB0 and MobileNetV3Large: ImageNet transfer learning, head training,
  selected-tail fine-tuning, fixed BatchNorm and float32 classification output.
- Training-only augmentation, class weighting, early stopping, learning-rate
  reduction and resumable full-model epoch checkpoints on persistent storage.
- Actual held-out metrics, per-class reports, confusion matrices, training time
  and synchronized single-image inference latency with architecture comparison.
- Controlled validation-only trials, macro-F1 checkpointing, optional AdamW/
  label smoothing, balanced accuracy, confidence diagnostics and rejection profiles.
- Explicitly mapped external evaluation with overlap screening, frozen selection,
  internal matched-class comparisons and optional human-annotated error analysis.
- Grad-CAM for both nested backbones, top-three predictions, confidence warnings,
  source-backed static disease records and a green Gradio Blocks interface.

**Research status:** no PlantVillage training, GPU experiment or Colab execution
has been performed during development. No real accuracy, external generalization
result or screenshot is supplied. Synthetic CPU tests verify software only.
See [validation notes](docs/VALIDATION.md).
The [optimization plan](docs/OPTIMIZATION_PLAN.md) separates the 90% independent
real-world accuracy target from achieved software checks and pending experiments.
No independent real-world target has been verified.
Current executed checks and pending outcomes are recorded in
[OPTIMIZATION_VALIDATION.md](docs/OPTIMIZATION_VALIDATION.md).
The [initial audit](docs/AUDIT_INITIAL.md) records findings before fixes; the
[final audit](docs/AUDIT_FINAL.md) lists evidence, fixes and publication gates.

## Layout

```text
notebooks/AgriVision_AI_Complete.ipynb   Main Colab entry point
src/config.py                         Settings and portable paths
src/environment.py                    TensorFlow/GPU inspection and policy
src/bootstrap.py                      Checked checkout and constrained Colab install
src/artifacts.py                      Model/config/mapping integrity checks
src/acquisition.py                    Safe upload-ZIP extraction
src/dataset.py                        Audit, hashes, reports and frozen splits
src/preprocessing.py                  Shared RGB decode and resize
src/augmentation.py                   Training-only Keras transforms
src/models.py                         Both architectures
src/metrics.py                        Balanced/class metrics and confidence diagnostics
src/validation.py                     Validation-only scoring and rejection threshold
src/experiments.py                    Controlled trials, frozen selection and exports
src/external.py                       Reviewed external mapping/audit/evaluation
src/error_analysis.py                 Saved errors and human-annotated subgroups
src/training.py                       Two-stage training and resumption
src/evaluation.py                     Metrics and comparison
src/gradcam.py                        Nested-backbone explanations
src/prediction.py                     Checkpoint-backed inference
src/disease_information.py            Static information lookup
src/app.py                            Gradio Blocks
src/cli.py                            Terminal entry point
app.py                                Local demo launcher
disease_info.json                     Reviewed sources and explicit fallbacks
configs/default.json                  Editable experiment defaults
scripts/                              Notebook builder and static checks
tests/test_cpu_smoke.py                Temporary synthetic software fixtures
docs/                                 Guides, methodology and viva
data/                                 Ignored dataset storage
outputs/                              Ignored experiment artifacts
```

## Colab workflow

1. Follow [GitHub upload instructions](docs/GITHUB_GUIDE.md).
2. Open the main notebook from GitHub in Colab and select a T4 GPU if available.
3. Inspect TensorFlow/GPU first, set the repository URL, clone code and install
   `requirements-colab.txt`, which preserves Colab's TensorFlow installation.
4. Mount Drive, configure paths, upload the original color PlantVillage ZIP.
5. Extract and inspect the audit; train controlled trials on the same frozen split.
6. Freeze validation-selected winners, evaluate internal/optional external data,
   check both Grad-CAM implementations and launch the versioned Gradio demo.

Full instructions: [Colab execution guide](docs/COLAB_GUIDE.md).
For the current optimization study use the detailed
[training and external evaluation guide](docs/TRAINING_GUIDE.md).

## Local development

Use Python 3.11–3.13. Native Windows CPU execution supports development; Colab is
the intended T4 training environment. ML code uses TensorFlow and `tf.keras` only.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/check_static.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_repository.py
```

Linux/macOS: use `.venv/bin/python`. Select that interpreter in VS Code. The smoke
suite builds both real architectures with `weights=None` to avoid downloads; its
tiny integration model is a temporary test fixture. Production training uses
ImageNet weights and does not fall back to random initialization.
An optional network check loads actual ImageNet backbones and checks forward
passes/Grad-CAM at 224×224: `python scripts/audit_pretrained.py`. Weights remain
in ignored outputs/audit-cache. This measures no plant-disease accuracy.

Example commands from the repository root:
For this study, run real disease-model training in Colab. These local terminal
examples remain available for development; CPU tests use synthetic fixtures only.

```powershell
python -m src.cli environment --output "D:/AgriVision/experiment01"
python -m src.cli prepare --dataset "D:/PlantVillage/color" --output "D:/AgriVision/experiment01"
python -m src.cli train --architecture EfficientNetB0 --dataset "D:/PlantVillage/color" --output "D:/AgriVision/experiment01" --config configs/default.json
python -m src.cli train --architecture MobileNetV3Large --dataset "D:/PlantVillage/color" --output "D:/AgriVision/experiment01" --config configs/default.json
python -m src.cli evaluate --architecture EfficientNetB0 --dataset "D:/PlantVillage/color" --output "D:/AgriVision/experiment01"
python -m src.cli evaluate --architecture MobileNetV3Large --dataset "D:/PlantVillage/color" --output "D:/AgriVision/experiment01"
python -m src.cli compare --output "D:/AgriVision/experiment01"
python app.py --output "D:/AgriVision/experiment01"
```

Explicit paths work across OSs. Defaults resolve from the source repository,
not the current terminal directory. `AGRIVISION_DATA` and `AGRIVISION_OUTPUT`
are optional environment alternatives. Pass matching `--config` for non-default
image/batch settings during evaluation. The notebook imports these same modules.

## Scientific limits

PlantVillage backgrounds can cause shortcut learning. Grouped random splits are
not field/site-disjoint experiments. dHash misses some similar images and can
over-group others; review its reports. Softmax scores are uncalibrated and do not
reject every unfamiliar/non-leaf upload. Grad-CAM does not prove a diagnosis.

Disease records provide concise nonchemical prevention information from university
extension sources. Unknown/unverified records display “Information not verified yet”.
Healthy is a predicted class, not proof of absence of every pest/disease. Alternate
label spellings still train through the discovered mapping but may receive an
information fallback. No professional treatment or chemical dosage is generated.

PlantDoc evaluation is implemented but pending actual reviewed data. ConvNeXt
and temperature scaling remain future work. No external generalization result
is claimed.

## Training, evaluation and Grad-CAM

The CLI examples above and the notebook train a frozen head, then a selected
backbone tail at a lower learning rate. BatchNorm stays fixed. Atomic epoch
checkpoints preserve optimizer state; a committed best-epoch pointer allows
repair of best.keras after interruption. Only finite epoch metrics are committed.

Evaluation uses the held-out test manifest after validation-based selection.
Accuracy, macro precision/recall/F1, weighted F1, per-class reports and confusion
matrices are saved. Comparison requires current model/evaluation files and matching
source/mapping/splits. Baseline comparison requires equal configs; selected-pipeline
comparison discloses differing winning configs. `latency_comparable` flags whether recorded timing
contexts match; timings from different contexts are not a controlled speed comparison.

Grad-CAM weights convolutional maps using spatially averaged gradients of the
selected logit, applies ReLU, normalizes and overlays the map. Both nested
architectures are handled independently. Zero maps are reported honestly.
See [methodology](docs/METHODOLOGY.md) for the full protocol.

## Gradio demo

After training, run `python app.py --output PATH_TO_EXPERIMENT` or the notebook's
demo cell. Upload JPG/JPEG/PNG, choose a model and analyze. The interface displays
top-three scores, uncertainty, original/heatmap/overlay, static information and
matching saved metrics. Missing, corrupt or stale evaluation shows an explicit
unavailable status. A Colab share link is temporary and depends on the live session.

## Results — pending real experiments

| Architecture | PlantVillage accuracy / macro F1 | External accuracy / macro F1 | Training status |
|---|---|---|---|
| EfficientNetB0 | Pending real evaluation | Not evaluated | No disease training performed |
| MobileNetV3Large | Pending real evaluation | Not evaluated | No disease training performed |

CPU software checks and ImageNet-backbone checks are not disease-classification
experiments. Do not insert numerical performance or screenshots until collected
from the actual dataset workflow.

## Future scope and references

Future experiments can execute the implemented compatible-class PlantDoc workflow,
add calibration, multiple seeds and more diverse independently collected field imagery.
Primary papers, framework documentation and disease references are listed in
[REFERENCES.md](docs/REFERENCES.md); each verified disease record also links to
its source directly.

See [methodology](docs/METHODOLOGY.md), [viva](docs/VIVA.md),
[report/presentation outline](docs/REPORT_AND_PRESENTATION.md), and
[references](docs/REFERENCES.md).
