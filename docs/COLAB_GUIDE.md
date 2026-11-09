# Complete Colab execution guide

For optional visual interpretation after CNN prediction, follow
[AGENT_GUIDE.md](AGENT_GUIDE.md). The notebook includes a disabled-by-default
private Colab Secrets setup cell before the Gradio demo. This feature requires
both completed CNN checkpoints; training itself never calls the agent API.

For the current validation-trial and independent external-evaluation workflow,
follow [TRAINING_GUIDE.md](TRAINING_GUIDE.md). The main notebook now implements
that workflow. The baseline setup details below remain useful context.

## Dataset upload

Obtain the original PlantVillage **color** distribution, retaining its provenance.
Upstream: https://github.com/spMohanty/PlantVillage-Dataset; dataset paper:
https://arxiv.org/abs/1511.08060. Check current license/terms and record source,
download date and version. If the upstream download is impractical, use a documented
institutional copy/mirror of the same original data. No alternate dataset is silently
substituted. Do not use an augmented derivative as original PlantVillage.

Keep original folders, for example `Apple___Apple_scab`, `Potato___Early_blight`,
`Corn_(maize)___Northern_Leaf_Blight`, `Tomato___Bacterial_spot`. Include all available
classes for the four crops. Do not mix color, grayscale, segmented or augmented
copies. ZIP the color tree; parent folders are allowed.

Create `MyDrive/AgriVision-AI/datasets/` and upload the ZIP as `PlantVillage.zip`.
Do not put it in GitHub. The notebook extracts to `/content` for faster reads.
Alternatively point DATASET_DIR at an already extracted Drive directory and skip
extraction; Drive image I/O may be slower.

## Runtime and repository

Follow GITHUB_GUIDE.md, open the main notebook from GitHub, and save a working
copy to Drive. Select **Runtime → Change runtime type → T4 GPU**, if available.
Run the environment cell before dependency installation. The printed GPU list
should include a device. `nvidia-smi` reports driver compatibility; TensorFlow's
build metadata reports its compiled CUDA stack. Configuration later checks the
actual device of a completed matrix multiplication.

Set REPOSITORY_URL and clone. Reused checkouts must match the URL and have no
local edits; the imported src package must resolve to that checkout. Restart
the session if a previous checkout's modules are still imported.

Use the notebook installer around `requirements-colab.txt`. It excludes
TensorFlow and additionally constrains installed TensorFlow/Keras/NumPy/protobuf/
ml-dtypes/h5py versions. Conflicts stop installation rather than changing that
working stack; pip check must pass. If installation fails, use a compatible clean
runtime and inspect the dependency conflict instead of blindly upgrading packages.
If TensorFlow import/CUDA loading fails, start a fresh GPU runtime before replacing
the stack. Dependency ranges are compatibility bounds, not a claim every combination
has been tested. Actual installed versions are saved after setup.

Authorize the Drive prompt and review:

- DATASET_ZIP: uploaded archive in Drive.
- DATASET_DIR: temporary extracted images under `/content`.
- EXPERIMENT_DIR: persistent Drive storage, normally `.../experiments/baseline_01`.

Edit batch size/epochs before the first run. Use identical settings for both
architectures. A new config or data version requires a new experiment output
directory. The seed, config, precision policy, package versions and commit are recorded.
The original commit/config record is immutable on reconnect. Training source is
fingerprinted independently of its filesystem path; a changed implementation
requires its original source version or a new experiment directory. Existing
models/splits without these audit integrity records must be regenerated, not
silently relabeled as a new experiment.
Compatible GPUs use mixed_float16; classification logits/softmax remain float32.
CPU uses float32 and cannot establish GPU execution.

## Data audit and training

Run extraction and preparation. Inspect `dataset/report.json`, `audit.csv`,
`invalid_files.json`, `exact_duplicates.csv`, `near_duplicate_pairs.csv`,
`quarantined.csv`, samples and class-distribution graphs. Confirm all four crops
and the actual discovered disease classes. Exact label conflicts stop preparation;
near-label conflicts are quarantined. A class needs three independent groups.
Whole groups stay together, so split ratios are approximate. Review dHash matches
for false positives; the heuristic cannot find every biologically related image.

Reruns reuse the frozen manifest. Its checksum, mapping digest, integer labels,
split/group integrity, safe relative paths, split settings and actual image hashes are checked,
so missing/changed files fail clearly. Relative manifest paths allow moving the
same image tree between local storage and Colab. Inspect raw [0,255] preprocessing
and mild augmentation; do not add a second `/255` normalization.

Train EfficientNetB0, then MobileNetV3Large on the identical manifest. First use
downloads ImageNet weights. Stage 1 trains the head, Stage 2 fine-tunes a selected
backbone tail at a lower learning rate. BatchNorm statistics stay fixed. Best
validation loss can select a checkpoint from either stage.

Each `models/<architecture>/` folder stores `best.keras`, committed
`resume_<stage>_<epoch>_<id>.keras` files with optimizer state, `state.json`, `class_indices.json`,
`config.json`, `preprocessing.json` and `history.csv`. Graphs go to `graphs/`.
Drive checkpoint serialization can be slow. Reported training time is accumulated
epoch wall time, excluding most checkpoint writes; it is not total session duration.
The state records an immutable best-epoch pointer. Files are written atomically
before state is committed; best.keras is a repairable convenience copy. The current
resume and best epoch are retained, and superseded managed checkpoints are removed
only after the new commit. Failed writes may leave temporary/orphan files for review.
Legacy split reports without mapping/split-config integrity fields must be re-audited
into a new output directory; existing experiment files are not deleted or silently rewritten.

## Resume after disconnection

Reconnect a GPU runtime. Rerun inspection, clone, dependencies, Drive and identical
path/config setup. Extract the ZIP again if `/content` was lost. Reuse the same
EXPERIMENT_DIR, load the manifest and rerun the relevant training cell. Completed
epochs, stage, weights and optimizer resume from the last committed epoch. An
interrupted epoch repeats. Callback patience counters and augmentation ordering
restart; this is not bit-for-bit continuation. Completed runs skip retraining.
Use only one writer per experiment directory.

## Evaluation, explanations and demonstration

After validation-based model selection, evaluate both models. Save real accuracy,
macro precision/recall/F1, weighted F1, per-class reports, test predictions and
confusion matrices. Do not tune against the test set. Inference timing uses five
warmups and 30 synchronized batch-one forward passes, excluding preprocessing and
Grad-CAM. Compare timings only on the same hardware/policy.
Comparison now requires all requested current checkpoints/evaluations and the same
configuration/mapping/manifest. Its latency_comparable column must be true before
interpreting timing differences as a controlled architecture speed comparison.

Run the independent Grad-CAM cell for both models. It prints each discovered
layer and displays original/heatmap/overlay. Zero positive maps are identified.
Sensitivity maps do not establish correct diagnosis. Prediction extracts crop
from the joint crop/disease class; the confidence threshold is uncalibrated.

Launch Gradio, upload JPG/JPEG/PNG and select a trained architecture. The app
loads only completed project checkpoints. Evaluation is read from matching saved
files; missing/stale metrics show “Evaluation not yet available”. Information comes
from disease_info.json, with a fallback for unverified entries. Public share links
are temporary and depend on the live Colab session. Set SHARE=False for inline use.

## Common errors

| Symptom | Action |
|---|---|
| No GPU | Select GPU; free quota may be unavailable. |
| OOM | Use batch size 16 or 8 for both models in a new experiment. |
| Dataset missing | Check mount, ZIP path and extraction destination. |
| No classes | Inspect Crop___Disease folder names and original color distribution. |
| Too few groups | Review hash grouping; obtain more independent images. |
| Manifest/config mismatch | Restore original files/settings or start a new experiment. |
| ImageNet download fails | Restore network and retry; do not silently use random weights. |
| Disconnected Grad-CAM gradient | Share traceback and TensorFlow/Keras versions. |
| Zero positive heatmap | Treat as uninformative and inspect another image/class. |
| Missing checkpoint | Finish training or select the correct output directory. |

Use actual outputs/screenshots in the report and slides. Optional PlantDoc
evaluation is deferred: keep it separate, manually map compatible classes and
report unsupported labels/counts. A small external set cannot establish universal
field generalization.
