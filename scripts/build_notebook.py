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
from src.utils import write_json
from IPython.display import Image as DisplayImage, display
drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/AgriVision-AI')
DATASET_ZIP = DRIVE_ROOT / 'datasets' / 'PlantVillage.zip'
DATASET_DIR = Path('/content/plantvillage_original_color')
EXPERIMENT_DIR = DRIVE_ROOT / 'experiments' / 'baseline_01'
config = ProjectConfig()  # Edit epochs or batch_size BEFORE first training.
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
plot_augmentation(sample, paths.output / 'graphs' / 'augmentation.png', config.seed)
display(DisplayImage(filename=str(paths.output / 'graphs' / 'augmentation.png')))
""")
section("8. Train EfficientNetB0",
        "Expected: real epoch logs, best.keras, resumable checkpoint with optimizer state, histories and graphs. First run downloads ImageNet weights. After disconnection rerun setup/extraction and this cell with unchanged config/output paths. Completed epochs resume; patience counters and augmentation ordering restart, so continuation is not bit-for-bit identical.", """
from src.training import train_architecture
print(train_architecture('EfficientNetB0', paths, config, resume=True))
display(DisplayImage(filename=str(paths.output / 'graphs' / 'EfficientNetB0_training.png')))
""")
section("9. Train MobileNetV3Large on the same manifest",
        "Expected: the same artifact types for MobileNetV3Large. Both use the same data, seed and schedule. Tail-layer count is a practical baseline, not an equal compute-budget comparison. If GPU memory is exhausted, use batch_size=16 in a NEW experiment, then run both models with that config.", """
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(config.seed)
print(train_architecture('MobileNetV3Large', paths, config, resume=True))
display(DisplayImage(filename=str(paths.output / 'graphs' / 'MobileNetV3Large_training.png')))
""")
section("10. Evaluate and compare",
        "Expected: actual test metrics, per-class reports, confusion matrices and comparison CSV. Checkpoints use validation loss; do not tune against the test set. Latency uses five warmups and 30 synchronized batch-one calls, excluding I/O and Grad-CAM. Interpret difficult classes using actual confusion matrices.", """
from src.evaluation import evaluate_architecture, compare_models
for architecture in config.architectures:
    tf.keras.backend.clear_session()
    result = evaluate_architecture(architecture, paths, config)
    display({k: result[k] for k in ('model', 'accuracy', 'precision_macro', 'recall_macro', 'macro_f1', 'weighted_f1', 'inference_ms_median')})
    display(DisplayImage(filename=str(paths.output / 'confusion_matrices' / f'{architecture}.png')))
display(compare_models(paths, config.architectures))
""")
section("11. Independently verify Grad-CAM for both models",
        "Expected: original, heatmap and overlay plus the independently discovered layer name. Grad-CAM computes ReLU(sum_k alpha_k A_k), with alpha_k equal to mean spatial gradient of the selected logit. A zero positive map is reported, not hidden. This is sensitivity evidence, not proof of correct diagnosis.", """
from src.gradcam import gradcam
from PIL import Image
import numpy as np
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
    del model
""")
section("12. Prediction and static disease information",
        "Set PREDICTION_IMAGE to a local or Drive leaf image. Expected: crop/class, confidence, top-three, predicted health status and warning when below threshold. All advice comes from disease_info.json; unverified records explicitly say so. High softmax confidence cannot establish reliability for unfamiliar images.", """
from src.prediction import PlantDiseasePredictor
from src.disease_information import information_markdown
PREDICTION_IMAGE = test_image
predictor = PlantDiseasePredictor(paths, 'EfficientNetB0')
prediction = predictor.predict_plant_disease(PREDICTION_IMAGE)
display({k: v for k, v in prediction.items() if k not in ('gradcam', 'information')})
print(information_markdown(prediction['information']))
display(Image.fromarray(prediction['gradcam']['overlay']))
del predictor
tf.keras.backend.clear_session()
""")
section("13. Gradio demonstration",
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
Optional PlantDoc, ConvNeXt and calibration are deferred; no external generalization
claim is made.
""")
nb = {"cells": cells, "metadata": {"colab": {"name": "AgriVision_AI_Complete.ipynb", "provenance": []},
      "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks" / "AgriVision_AI_Complete.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(destination)
