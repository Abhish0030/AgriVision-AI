"""Post-test error analysis from saved predictions, with optional human annotations."""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from .utils import write_json

ATTRIBUTES = ("background", "lighting", "orientation", "image_quality", "disease_severity")


def analyze_external_errors(paths, architecture, annotations_csv=None):
    folder = paths.output / "evaluation"
    predictions = pd.read_csv(folder / f"{architecture}_external_predictions.csv")
    predictions[predictions.label != predictions.predicted_label].to_csv(folder / f"{architecture}_external_errors.csv", index=False)
    if annotations_csv is None:
        template = predictions[["path"]].copy()
        for attribute in ATTRIBUTES:
            template[attribute] = ""
        path = folder / "external_annotations_template.csv"
        if not path.exists():
            template.to_csv(path, index=False)
        return {"status": "Subgroup analysis pending human annotations", "template": str(path),
                "note": "No lighting/background/severity labels are invented from filenames or predictions."}
    annotations = pd.read_csv(annotations_csv)
    if "path" not in annotations or annotations.path.duplicated().any() or not set(annotations.path).issubset(set(predictions.path)):
        raise ValueError("Annotations require unique paths belonging to the frozen external test set.")
    merged = predictions.merge(annotations, on="path", how="left", validate="one_to_one")
    rows = []
    for attribute in ATTRIBUTES:
        if attribute not in annotations:
            continue
        for category, group in merged.dropna(subset=[attribute]).groupby(attribute):
            ids = sorted(group.label.unique())
            _, recall, f1, _ = precision_recall_fscore_support(group.label, group.predicted_label, labels=ids, average="macro", zero_division=0)
            rows.append({"attribute": attribute, "category": str(category), "images": len(group),
                         "accuracy": float((group.label == group.predicted_label).mean()), "macro_f1": float(f1),
                         "balanced_accuracy": float(recall), "mean_confidence": float(group.confidence.mean())})
    pd.DataFrame(rows).to_csv(folder / f"{architecture}_external_subgroups.csv", index=False)
    result = {"groups": rows, "limitation": "Small annotated groups are exploratory error analysis, not a basis for tuning against the test set."}
    write_json(folder / f"{architecture}_external_subgroups.json", result)
    return result
