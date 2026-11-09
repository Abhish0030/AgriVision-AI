"""Regenerate the main notebook from its reviewable, modular workflow."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "id": f"agrivision-{len(cells):02d}",
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def section(title, explanation, source):
    add("markdown", f"## {title}\n\n{explanation}")
    add("code", source)


add("markdown", """
# AgriVision AI — Complete modular Colab workflow
**An Explainable Deep Learning-Based Multi-Crop Plant Disease Detection and Classification System**

Develop in VS Code → push code to GitHub → clone here → upload real PlantVillage
data to Drive → audit → train both architectures → evaluate → explain → demo.

Select **Runtime → Change runtime type → T4 GPU** when available. Run cells in order.
See `docs/COLAB_GUIDE.md` for troubleshooting and reconnection instructions.
This notebook imports project modules and never rewrites them. No output, accuracy
or screenshot has been pre-filled. Free Colab GPU access is not guaranteed.
""")
section("1. Inspect TensorFlow, GPU and CUDA before installation",
        "Expected: versions, build details and GPU devices. If TensorFlow cannot import, start a fresh Colab GPU runtime. Do not replace CUDA blindly.", """
import sys
import subprocess
import tensorflow as tf
print('Python:', sys.version)
print('TensorFlow:', tf.__version__)
print('CUDA build:', tf.sysconfig.get_build_info())
print('GPU devices:', tf.config.list_physical_devices('GPU'))
try:
    print(subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False).stdout)
except FileNotFoundError:
    print('nvidia-smi unavailable; select a GPU runtime before training.')
""")
section("2. Clone your GitHub repository",
        "Replace REPOSITORY_URL below. Expected: a commit identifier. Existing checkouts are reused without changing files. Never paste private access tokens into notebooks.", """
from pathlib import Path
REPOSITORY_URL = 'https://github.com/Abhish0030/AgriVision-AI.git'
CHECKOUT = Path('/content/AgriVision-AI')
if 'YOUR_USERNAME' in REPOSITORY_URL:
    raise ValueError('Set REPOSITORY_URL to your GitHub repository first.')
if not (CHECKOUT / '.git').is_dir():
    subprocess.check_call(['git', 'clone', REPOSITORY_URL, str(CHECKOUT)])
print(subprocess.check_output(['git', '-C', str(CHECKOUT), 'rev-parse', 'HEAD'], text=True))
sys.path.insert(0, str(CHECKOUT))
from src.bootstrap import validate_checkout  # noqa: E402 - checkout must exist before importing its modules
COMMIT = validate_checkout(CHECKOUT, REPOSITORY_URL)
print('Verified source commit:', COMMIT)
""")
section("3. Install supporting dependencies",
        "Expected: successful installation and pip check. The installer pins the already working TensorFlow, Keras, NumPy, protobuf, ml-dtypes and h5py versions. If requirements conflict, it stops instead of replacing the stack. Resolve errors in a compatible clean runtime before training.", """
from src.bootstrap import install_colab_dependencies
stack_versions = install_colab_dependencies(CHECKOUT, Path('/content/agrivision-tf-constraints.txt'))
print('Preserved TensorFlow stack:', stack_versions)
""")
section("4. Mount Drive and configure paths",
        "Authorize Drive. Upload the original color PlantVillage ZIP to MyDrive/AgriVision-AI/datasets/PlantVillage.zip. Checkpoints persist in EXPERIMENT_DIR. Use a new experiment directory if data or config changes. Expected: actual precision policy and device; T4 uses mixed_float16, CPU uses float32.", """
from google.colab import drive
from src.config import ProjectConfig, ProjectPaths
from src.environment import configure_runtime
from src.bootstrap import record_experiment
from src.utils import read_json, write_json
from IPython.display import Image as DisplayImage, display
drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/AgriVision-AI')
DATASET_ZIP = DRIVE_ROOT / 'datasets' / 'PlantVillage.zip'
DATASET_DIR = Path('/content/plantvillage_original_color')
EXPERIMENT_DIR = DRIVE_ROOT / 'experiments' / 'optimization_01'
config = ProjectConfig(checkpoint_monitor='val_macro_f1')  # Edit epochs/batch size BEFORE saving the trial plan.
paths = ProjectPaths.create(dataset=DATASET_DIR, output=EXPERIMENT_DIR).initialize()
record_experiment(paths, config, REPOSITORY_URL, COMMIT)
environment = configure_runtime(config, paths.output / 'runs')
config.save(paths.output / 'runs' / 'requested_config.json')
commit = subprocess.check_output(['git', '-C', str(CHECKOUT), 'rev-parse', 'HEAD'], text=True).strip()
write_json(paths.output / 'runs' / 'repository.json', {'url': REPOSITORY_URL, 'commit': commit})
print('Dataset:', paths.dataset)
print('Persistent output:', paths.output)
""")
section("5. Extract your uploaded dataset",
        "Expected: image files in original Crop___Disease class folders. Do not mix color, grayscale, segmented or augmented datasets. Missing ZIP: upload it to the path above. Existing folders are kept; extraction rejects traversal and overwrites. After a new Colab session, extract again into temporary storage.", """
from src.acquisition import extract_dataset_zip
if not DATASET_DIR.exists():
    extract_dataset_zip(DATASET_ZIP, DATASET_DIR)
else:
    print('Dataset already exists. Verify it belongs to this experiment.')
print('Images:', sum(p.suffix.lower() in {'.jpg', '.jpeg', '.png'} for p in DATASET_DIR.rglob('*')))
""")
section("6. Audit data and create or reuse frozen splits",
        "Expected: counts, invalid-file report, duplicate CSVs, approximate 70/15/15 group-stratified splits, distribution and samples. Review near_duplicate_pairs.csv and quarantined.csv before training. dHash grouping is heuristic and cannot guarantee absence of all biological or near-duplicate leakage. Missing crops or too few independent groups stop preparation.", """
from src.dataset import prepare_dataset, load_manifest
if (paths.output / 'dataset' / 'manifest.csv').exists():
    frame, mapping, dataset_report = load_manifest(paths, verify_files=True, config=config)
else:
    frame, mapping, dataset_report = prepare_dataset(paths, config)
display(dataset_report)
display(frame.groupby(['class_name', 'split']).size().unstack(fill_value=0))
for name in ('class_distribution.png', 'samples.png'):
    display(DisplayImage(filename=str(paths.output / 'graphs' / name)))
""")
section("7. Inspect preprocessing and augmentation",
        "Expected: float32 RGB pixels [0,255], original and transformed images. No second normalization is applied. Mild pose and brightness/contrast changes model camera variation; hue shifts are avoided because symptom color matters. Validation/test never use augmentation.", """
from src.preprocessing import resize_rgb
from src.augmentation import plot_augmentation
sample = resize_rgb(paths.dataset / frame.iloc[0].path, config.image_size)
print(sample.shape, sample.dtype, sample.min(), sample.max())
for profile in ('none', 'mild', 'robust'):
    plot_augmentation(sample, paths.output / 'graphs' / f'augmentation_{profile}.png', config.seed, profile)
    display(DisplayImage(filename=str(paths.output / 'graphs' / f'augmentation_{profile}.png')))
""")
section("8. Train controlled trials using validation only",
        "Expected: six trials by default — each model with none/mild/robust augmentation. Optional optimizer/tail/smoothing trials change one factor relative to mild. They are candidates, not proven improvements. Set flags BEFORE first execution. ImageNet downloads once; full runs may exceed one free Colab session. Use RUN_TRIAL_IDS to run a subset from the saved plan, then resume remaining trials. No test inference occurs here. GPU training is required; local synthetic tests are separate.", """
from src.experiments import build_trial_plan, run_trials
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('Select a Colab GPU runtime before real training. Do not report CPU software tests as this experiment.')
INCLUDE_ADAMW_TRIALS = False
INCLUDE_DEEPER_TAIL_TRIALS = False
INCLUDE_LABEL_SMOOTHING_TRIALS = False
trials = build_trial_plan(config, include_optimizer=INCLUDE_ADAMW_TRIALS,
                          include_fine_tuning=INCLUDE_DEEPER_TAIL_TRIALS,
                          include_smoothing=INCLUDE_LABEL_SMOOTHING_TRIALS)
display([{'trial_id': t['trial_id'], 'architecture': t['architecture'], 'config': t['config']} for t in trials])
RUN_TRIAL_IDS = None  # Or a list of IDs printed above; selection later requires the entire saved plan.
display(run_trials(paths, config, trials, only_trial_ids=RUN_TRIAL_IDS))
""")
section("9. Freeze winners using validation macro F1",
        "Expected: selection.json, version registry, one exported winner per architecture, and efficientnet_best.keras / mobilenet_best.keras aliases. No test scores influence selection. All planned trials must finish. After freezing, this experiment cannot add trials based on held-out results. If GPU memory failed, change batch size only in a NEW experiment for every trial.", """
from src.experiments import select_best_trials, export_selected_models
selection = select_best_trials(paths)
display(selection)
display(export_selected_models(paths))
for architecture in selection['winners']:
    display(DisplayImage(filename=str(paths.output / 'graphs' / f'{architecture}_training.png')))
""")
section("10. Evaluate the frozen winners on internal test data",
        "Expected: actual accuracy, macro/weighted F1, balanced accuracy, per-class reports, matrices, confidence diagnostics, comparison charts and timing. Saved matching results are reused on reconnection rather than retuning. If winning trial configs differ, the table discloses that; this is a comparison of selected pipelines rather than an equal-settings architecture-only claim. Latency excludes image I/O and Grad-CAM.", """
from src.experiments import evaluate_selected_models
display(evaluate_selected_models(paths))
for architecture in selection['winners']:
    display(read_json(paths.output / 'evaluation' / f'{architecture}.json'))
    display(DisplayImage(filename=str(paths.output / 'confusion_matrices' / f'{architecture}.png')))
    display(DisplayImage(filename=str(paths.output / 'evaluation' / f'{architecture}_per_class.png')))
display(DisplayImage(filename=str(paths.output / 'evaluation' / 'model_comparison.png')))
""")
section("11. Independently verify Grad-CAM for both models",
        "Expected: original, heatmap and overlay plus the independently discovered layer name. Grad-CAM computes ReLU(sum_k alpha_k A_k), with alpha_k equal to mean spatial gradient of the selected logit. A zero positive map is reported, not hidden. This is sensitivity evidence, not proof of correct diagnosis.", """
from src.gradcam import gradcam, save_error_examples
from PIL import Image
import numpy as np
import pandas as pd
test_image = paths.dataset / frame[frame.split == 'test'].iloc[0].path
for architecture in config.architectures:
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(paths.output / 'models' / architecture / 'best.keras', compile=False)
    cam = gradcam(model, test_image)
    assert np.isfinite(cam['values']).all()
    print(architecture, 'layer:', cam['layer'], '| zero positive map:', cam['zero_map'])
    for key in ('original', 'heatmap', 'overlay'):
        Image.fromarray(cam[key]).save(paths.output / 'predictions' / f'{architecture}_{key}.png')
        display(Image.fromarray(cam[key]))
    saved_predictions = pd.read_csv(paths.output / 'predictions' / f'{architecture}_test.csv')
    display(save_error_examples(model, saved_predictions, paths.dataset, paths.output / 'predictions' / architecture))
    del model
""")
section("12. Prediction and static disease information",
        "Set PREDICTION_IMAGE to a local or Drive leaf image. Expected: crop/class, confidence, top-three, predicted health status and warning when below threshold. All advice comes from disease_info.json; unverified records explicitly say so. High softmax confidence cannot establish reliability for unfamiliar images.", """
from src.prediction import PlantDiseasePredictor
from src.disease_information import information_markdown
PREDICTION_IMAGE = test_image
predictor = PlantDiseasePredictor(paths, selection['overall_best_architecture'])
prediction = predictor.predict_plant_disease(PREDICTION_IMAGE)
display({k: v for k, v in prediction.items() if k not in ('gradcam', 'information')})
print(information_markdown(prediction['information']))
display(Image.fromarray(prediction['gradcam']['overlay']))
del predictor
tf.keras.backend.clear_session()
""")
section("13. Optional external dataset: upload and explicitly review labels",
        "Skip only if no external data is available; external accuracy remains unverified. Upload a separate PlantDoc classification ZIP to Drive. Use its test directory initially; none of its images may enter our training. PlantDoc contains internet-scraped imagery, not guaranteed representative field sampling. Inspect actual folders/labels and the conservative mapping: ambiguous rust/blight/virus/healthy labels are null by default. Set the review flag yourself only after inspection. Do not guess missing classes or conflate generic labels with specific pathogens.", """
from src.utils import read_json
RUN_EXTERNAL_EVALUATION = False
EXTERNAL_ZIP = DRIVE_ROOT / 'datasets' / 'PlantDoc.zip'
EXTERNAL_EXTRACT = Path('/content/plantdoc_external')
EXTERNAL_ROOT = EXTERNAL_EXTRACT / 'PlantDoc-Dataset-master' / 'test'  # Adjust to your ZIP's actual tree.
if RUN_EXTERNAL_EVALUATION:
    if not EXTERNAL_EXTRACT.exists():
        extract_dataset_zip(EXTERNAL_ZIP, EXTERNAL_EXTRACT)
    print('External root:', EXTERNAL_ROOT)
    print('Folders:', sorted(p.name for p in EXTERNAL_ROOT.iterdir() if p.is_dir()))
    external_spec = read_json(CHECKOUT / 'configs' / 'plantdoc_mapping.json')
    display(external_spec)
else:
    print('External evaluation not run. The 90% independent real-world target is NOT verified.')
""")
section("14. Screen external overlap and evaluate without training on it",
        "Review mapping and provenance before changing EXTERNAL_MAPPING_REVIEWED. Review the generated external/audit.csv for false near matches and excluded classes. Models predict over ALL trained classes; external macro metrics use compatible truth classes. Matched internal-class results separate class-composition effects from domain shift. Wilson intervals/sample support and unconditional accuracy are reported. Rejection accuracy is conditional, not the 90% all-image target. A small or uncertain-provenance external set cannot verify broad real-world performance.", """
from src.external import prepare_external_dataset, load_external_manifest, evaluate_external
from src.error_analysis import analyze_external_errors
from src.experiments import generate_final_comparison
EXTERNAL_MAPPING_REVIEWED = False  # Required manual review, not a default approval.
INDEPENDENCE_CONFIRMED_BY_USER = False
EXTERNAL_AUDIT_REVIEWED = False  # Run once to generate reports; inspect them, then set True and rerun.
if RUN_EXTERNAL_EVALUATION:
    external_spec['reviewed'] = EXTERNAL_MAPPING_REVIEWED
    external_spec['independence_confirmed_by_user'] = INDEPENDENCE_CONFIRMED_BY_USER
    if not (paths.output / 'external' / 'manifest.csv').exists():
        external_frame, external_report = prepare_external_dataset(paths, EXTERNAL_ROOT, external_spec)
    else:
        external_frame, external_report = load_external_manifest(paths, EXTERNAL_ROOT)
    display(external_report)
    if not EXTERNAL_AUDIT_REVIEWED:
        raise RuntimeError('Review external/audit.csv, support counts, excluded classes and near-match screening before enabling EXTERNAL_AUDIT_REVIEWED.')
    for architecture in selection['winners']:
        tf.keras.backend.clear_session()
        display(evaluate_external(architecture, paths, EXTERNAL_ROOT))
        display(analyze_external_errors(paths, architecture))  # Fill the generated annotation template for subgroup analysis.
display(generate_final_comparison(paths))
""")
section("15. Gradio demonstration of the validation-selected model",
        "Expected: inline interface and optional temporary share link. Only project checkpoints are accepted; metrics are read from matching saved evaluation files. SHARE=True exposes a temporary public demo while the runtime is alive. It expires when hosting stops. Set False for inline-only use.", """
from src.app import create_app
SHARE = True
demo = create_app(paths, config)
demo.launch(share=SHARE, debug=True)
""")
add("markdown", """
## Prepare the submission from real outputs
Inspect both histories, confusion matrices, per-class reports and timing records.
Review correct and failed predictions and Grad-CAM examples. Save your real demo
screenshots. Follow `docs/REPORT_AND_PRESENTATION.md` and `docs/VIVA.md`.
No real experiment results exist until you execute this workflow on PlantVillage.
External results are available only if the optional cells ran on actual reviewed
data. The 90% independent-field target is a research goal, not a promised or
automatically established result. Validation thresholds do not calibrate softmax
or guarantee future precision. No external images are used for training.
""")
nb = {"cells": cells, "metadata": {"colab": {"name": "AgriVision_AI_Complete.ipynb", "provenance": []},
      "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks" / "AgriVision_AI_Complete.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(destination)
