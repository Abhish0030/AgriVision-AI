# Colab training optimization and independent evaluation

## What is achieved, pending and targeted

Implemented: controlled trials, validation-only selection, resumable training,
balanced accuracy, confidence diagnostics, validation-based rejection, screened
external evaluation and versioned predictions. Local synthetic tests exercise
software; they do not establish plant-disease performance.

Pending: actual PlantVillage/PlantDoc files, image counts, GPU training, measured
internal/external accuracy/F1, the best architecture and training duration.
No connected Colab runtime or real dataset was available during development.

Research target: at least 90% unconditional accuracy on a genuinely independent
real-world test set, if supported by sufficient relevant data. It is not guaranteed.
PlantVillage test accuracy is controlled-dataset performance, not field accuracy.

## 1. Prepare the real datasets

Obtain original-color PlantVillage for Tomato, Potato, Corn/maize and Apple.
Follow [the acquisition guide](COLAB_GUIDE.md#dataset-upload), record source/version/
license, and upload `PlantVillage.zip` to:

```text
MyDrive/AgriVision-AI/datasets/PlantVillage.zip
```

Do not substitute an augmented derivative or mix segmented/grayscale copies.
Class names and image counts are discovered from actual directories. No class
count, training count or accuracy is invented before acquisition.

For optional external evaluation obtain the classification version of PlantDoc:
https://github.com/pratikkayal/PlantDoc-Dataset. Upload a separate `PlantDoc.zip`.
Its official repository contains train/test directories; start with its test tree.
The names are that dataset's original split, not permission to use its train tree
for our model fitting. Neither tree is used in our training. A larger external pool
can include previously unused images only if decided before evaluation and screened
consistently. Never use PlantDoc images for training while calling it independent.

The [PlantDoc paper](https://arxiv.org/abs/1911.10317) describes internet-scraped
images. These are useful for domain-shift assessment, but not proof of representative
agricultural field sampling. Prefer newly collected, independently labeled field
photos when available; use a separate root and explicit reviewed mapping for those too.

## 2. Open and configure the main notebook

[Open AgriVision AI in Colab](https://colab.research.google.com/github/Abhish0030/AgriVision-AI/blob/main/notebooks/AgriVision_AI_Complete.ipynb).
Save your own copy to Drive. Select **Runtime → Change runtime type → T4 GPU**
when available. The notebook inspects TensorFlow/CUDA/GPU before installation,
clones the repository and preserves the installed TensorFlow stack with constraints.
It never blindly upgrades TensorFlow. Fresh Colab execution still needs verification.

Run cells 1–7 in order. Authorize Drive and use an unused persistent output:

```python
EXPERIMENT_DIR = DRIVE_ROOT / 'experiments' / 'optimization_01'
config = ProjectConfig(checkpoint_monitor='val_macro_f1')
```

Initial values are 224×224, batch 32, head epochs 5, fine-tuning epochs 10, head LR
0.001, fine-tuning LR 0.00001, dropout 0.3, tail depth 30, Adam and optional training
class weights. These are configurable starting values, not empirically optimized
hyperparameters. If memory is insufficient, use batch 16 or 8 in a NEW experiment
for every trial. Do not edit a saved plan/config in place after partial results.

The dataset report includes corruption, exact/near matches, quarantine, dimensions,
imbalance and approximate 70/15/15 group-stratified splits. Inspect it before
training. Existing frozen splits are integrity-checked; trial directories receive
identical manifest/mapping/report copies. Validation and test are never augmented.
Raw RGB pixels remain [0,255] with the Keras backbones' preprocessing retained.

## 3. Choose a controlled plan and train in Colab

Cell 8 defaults to six candidates:

| Trial family | EfficientNetB0 | MobileNetV3Large | Change from mild baseline |
|---|---|---|---|
| no_augmentation | Yes | Yes | Disable training augmentation |
| mild_baseline | Yes | Yes | Original mild pose/lighting variation |
| camera_robustness | Yes | Yes | Add 5% saturation variation, 1.5-pixel noise and occasional mild Gaussian blur |

Optional flags add one-factor trials relative to mild, for each architecture:

- AdamW with configured weight decay, replacing Adam.
- A deeper tail: 20 more selected backbone layers.
- Label smoothing 0.05 with integer labels handled by a registered sparse-label
  loss. This is a candidate, not a justified improvement until validation supports it.

Do not enable all flags blindly. Defaults keep these off. Set the plan once,
review the printed configurations and keep a free-Colab time budget in mind.
Six two-stage trials can require multiple sessions; no runtime promise is made.

Each trial uses ImageNet weights, a frozen head-training stage and lower-LR tail
fine-tuning. BatchNorm remains frozen. The optimizer is recompiled after changing
trainability. Macro F1 is accumulated through a confusion-matrix Keras metric;
checkpointing, LR reduction and early stopping monitor validation macro F1 in the
optimization plan. Legacy direct baseline training can still monitor validation loss.

`RUN_TRIAL_IDS = None` runs the entire plan. To run a session-sized subset, set it
to a list of IDs printed by the notebook. All saved-plan trials must finish before
selection. Reconnect, rerun setup/extraction with unchanged code/config/output,
then run remaining trials. Full-model epoch checkpoints restore weights, optimizer,
stage and next epoch. Partial epochs repeat; augmentation order and callback
patience restart, so continuation is not bit-for-bit identical.

Trial outputs live in `EXPERIMENT_DIR/trials/<trial_id>/`. The master
`evaluation/validation_experiments.csv/json/png` records actual configs, seeds,
validation accuracy/macro F1/balanced accuracy/NLL, duration and observed F1 change
from mild baseline. An improvement there is an observed single-seed validation
result, not evidence of field generalization or statistical significance.

## 4. Select once, then open the held-out test

Cell 9 requires every planned trial's matching validation report. It ranks by:
validation macro F1, balanced accuracy, lower NLL, then stable trial ID. It freezes
`evaluation/selection.json` and chooses one winner per architecture plus an overall
validation-selected default. No internal/external test inference selects a winner.
After freezing, the trial runner refuses additional tuning in that experiment.

Exports include full project metadata under `models/<architecture>/` plus:

```text
models/efficientnet_best.keras
models/mobilenet_best.keras
models/model_versions.json
```

The aliases are convenience copies; use the project loader so mappings/configuration,
hashes, source version and rejection profile stay attached. Do not publish weights
or datasets to GitHub. Checkpoints and outputs persist in Drive.

Cell 10 evaluates the selected pipelines on the untouched PlantVillage test split.
Actual JSON/CSV/figures go under `outputs/evaluation/` relative to EXPERIMENT_DIR.
Reports include accuracy, macro precision/recall/F1, weighted F1, balanced accuracy,
all per-class support, confusion matrices, synchronized latency, ECE, Brier score,
NLL and an accuracy Wilson interval. Timing excludes preprocessing/Grad-CAM.

When selected configs differ, `configuration_comparable=False` discloses this as
a comparison of selected pipelines rather than architecture alone. Compare latency
only when its recorded contexts are comparable. Reconnecting reuses matching
saved results; do not revise a plan after viewing held-out scores.

## 5. Interpret confidence and explanations

Validation predictions fit a rejection threshold with a minimum accepted count
and an observed Wilson lower-bound screen. If no threshold qualifies, the saved
profile rejects every automatic prediction and warns rather than inventing a
reliable operating point. The UI can explicitly override the threshold.

This does not calibrate softmax. Early stopping/trial selection and threshold
screening reuse validation data and can be optimistic. A binomial interval assumes
independent observations; hash grouping cannot establish that assumption. Report
all-image accuracy separately from coverage/conditional accepted accuracy.

Cell 11 verifies both nested backbones' Grad-CAM layer, gradients, shapes and finite
heatmaps. Zero positive maps are labeled uninformative. Correct and incorrect
high-confidence examples are saved when available, rather than showing only
successes. Inspect background shortcuts, symptom regions and severe errors.
Grad-CAM cannot establish causality or prove the diagnosis.

## 6. External evaluation after explicit review

In Cell 13 set `RUN_EXTERNAL_EVALUATION=True`, extract your ZIP and adjust
EXTERNAL_ROOT to its actual test/class tree. The notebook displays folders and
`configs/plantdoc_mapping.json`. The repository's current folder names were checked
against the official classification repository, but local labels/images still
require review. Generic rust/blight/yellow-virus/healthy-looking labels are null
by default; do not guess a specific pathogen or healthy status from vague names.

Edit `external_spec['labels']` explicitly if you can justify additional mappings,
then set `EXTERNAL_MAPPING_REVIEWED=True` in Cell 14. Only targets in the actual
PlantVillage mapping are accepted. Mapping approval is not inferred from elapsed time.
Set independence confirmation only if provenance actually supports it.

The first execution writes `external/audit.csv`, `manifest.csv`, `mapping.json`
and `report.json`, then pauses with the expected review error while
`EXTERNAL_AUDIT_REVIEWED=False`. Inspect corruption, unsupported labels, overlap
exclusions, within-external duplicates and class support. Then set the flag True
and rerun. The audit screens all PlantVillage splits, uses exact RGB hashes and
dHash connected groups, and retains one same-label representative per near group.
False matches remain possible; do not quietly edit a frozen manifest.

External models predict over **all trained classes**. No probability renormalization
or removal of difficult model classes makes the test easier. Main external macro
metrics average compatible ground-truth classes; the full per-class report also
shows zero-support trained classes. Internal matched-class results help separate
class-composition effects from domain shift. All retained external images count
in unconditional accuracy, including rejected predictions.

The report records actual samples, per-class support, unsupported classes, interval,
ECE, rejection coverage, error predictions and the accuracy gap. A 90% point
estimate is recorded as such; it does not automatically establish the independent
real-world target. Small samples, ambiguous labels and internet-image provenance
are explicit limitations. Do not retrain/tune against these external scores.

## 7. Error analysis, demo and results

External error CSVs are saved. `analyze_external_errors()` creates a blank
annotation template for background, lighting, orientation, image quality and
disease severity; these labels are never invented from filenames/model scores.
Fill it from human review, then pass `annotations_csv=...` for exploratory subgroup
accuracy/recall/F1 and sample counts. This is post-test explanation, not tuning data.

Cell 15 launches Gradio with the validation-selected default. It shows the selected
trial/checkpoint version, top three, Grad-CAM, source-backed information and actual
saved evaluation. Slider 0 uses the saved validation threshold (or the documented
configured fallback); nonzero values are explicit manual overrides. Share links
are temporary and depend on the live Colab runtime.

`evaluation/final_comparison.csv/json` contains the final metric-by-model table.
Unavailable external values remain “Not evaluated”. Curves, per-class charts,
model comparison and internal-vs-external plots use measured data only. Keep actual
results/screenshots in Drive and complete the report from those files. Do not
replace README's pending results with fabricated or synthetic-test numbers.
