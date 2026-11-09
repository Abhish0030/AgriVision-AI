"""Independent external screening/evaluation with explicit reviewed label mapping."""
import hashlib
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from .artifacts import model_artifacts, validate_model_contract
from .dataset import BKTree, difference_hash, load_manifest
from .evaluation import plot_confusion, plot_per_class
from .metrics import classification_metrics, rejection_summary
from .preprocessing import ALLOWED_SUFFIXES, make_dataset, read_rgb
from .utils import confined_path, file_hash, read_json, validate_probabilities, write_json
from .validation import load_rejection_profile


def prepare_external_dataset(master, external_root, mapping_spec):
    external_root = Path(external_root).expanduser().resolve()
    if not external_root.is_dir():
        raise FileNotFoundError("Upload/extract the external dataset before preparing evaluation.")
    if external_root.is_relative_to(master.dataset) or master.dataset.is_relative_to(external_root):
        raise ValueError("External and PlantVillage directories must be separate.")
    if mapping_spec.get("reviewed") is not True or not mapping_spec.get("source_url"):
        raise ValueError("Review the explicit source-to-target mapping/provenance and set reviewed=True yourself.")
    targets = mapping_spec.get("labels", {})
    frame, mapping, pv_report = load_manifest(master)
    if any(target is not None and target not in mapping for target in targets.values()):
        raise ValueError("External mapping targets must exist in the actual trained PlantVillage class mapping.")
    folder = master.output / "external"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.csv").exists():
        raise FileExistsError("External manifest is frozen. Reuse it or choose a new experiment directory.")
    pv_pixels = set(frame.pixel_hash)
    pv_index = BKTree()
    for i, value in enumerate(frame.dhash):
        pv_index.add(int(value), i)
    radius = pv_report["near_duplicate_hamming_threshold"]
    audit, candidates, unsupported = [], [], {}
    for path in sorted(external_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        relative = path.relative_to(external_root).as_posix()
        ancestors = [part for part in path.relative_to(external_root).parts[:-1] if part in targets]
        source_label = ancestors[-1] if ancestors else path.parent.name
        target = targets.get(source_label)
        row = {"path": relative, "external_label": source_label, "class_name": target}
        if target is None:
            unsupported[source_label] = unsupported.get(source_label, 0) + 1
            audit.append({**row, "reason": "unsupported_or_ambiguous_label"})
            continue
        try:
            confined_path(external_root, relative)
            rgb = read_rgb(path)
            pixel_hash = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            dhash = difference_hash(rgb)
            if pixel_hash in pv_pixels or pv_index.query(dhash, radius):
                reason = "plantvillage_exact_overlap" if pixel_hash in pv_pixels else "plantvillage_near_overlap"
                audit.append({**row, "reason": reason})
                continue
            candidates.append({**row, "label": mapping[target], "sha256": file_hash(path), "pixel_hash": pixel_hash, "dhash": dhash})
        except (OSError, ValueError) as exc:
            audit.append({**row, "reason": "invalid_image", "detail": str(exc)})
    # Connected near-match groups preserve transitive relationships. Retain one
    # representative per same-label group; quarantine conflicting labels.
    parent = list(range(len(candidates)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    index = BKTree()
    for i, row in enumerate(candidates):
        for j in index.query(row["dhash"], radius):
            parent[find(i)] = find(j)
        index.add(row["dhash"], i)
    groups = {}
    for i, row in enumerate(candidates):
        groups.setdefault(find(i), []).append(row)
    retained = []
    for group_id, rows in groups.items():
        if len({row["class_name"] for row in rows}) != 1:
            audit.extend({**row, "reason": "conflicting_external_near_group"} for row in rows)
        else:
            retained.append({**rows[0], "group": group_id})
            audit.append({**rows[0], "reason": "retained"})
            audit.extend({**row, "reason": "external_exact_or_near_duplicate"} for row in rows[1:])
    pd.DataFrame(audit).to_csv(folder / "audit.csv", index=False)
    if not retained:
        raise ValueError("No compatible independent external images remain; review audit.csv and label mapping.")
    external = pd.DataFrame(retained).sort_values("path").reset_index(drop=True)
    external.to_csv(folder / "manifest.csv", index=False)
    write_json(folder / "mapping.json", mapping_spec)
    report = {"dataset_name": mapping_spec.get("dataset_name", "user_supplied_external"),
              "source_url": mapping_spec["source_url"], "mapping_reviewed": True,
              "independence_confirmed_by_user": mapping_spec.get("independence_confirmed_by_user") is True,
              "images": len(external), "supported_classes": external.class_name.value_counts().to_dict(),
              "unsupported_classes": unsupported, "excluded_reasons": pd.Series([row["reason"] for row in audit]).value_counts().to_dict(),
              "manifest_sha256": file_hash(folder / "manifest.csv"), "mapping_sha256": file_hash(folder / "mapping.json"),
              "plantvillage_manifest_sha256": pv_report["manifest_sha256"], "class_indices": mapping,
              "near_duplicate_hamming_threshold": radius,
              "limitation": "Hash screening is heuristic; manual provenance, label and near-match review are still required. No external images enter training."}
    write_json(folder / "report.json", report)
    return external, report


def load_external_manifest(master, external_root):
    folder = master.output / "external"
    report = read_json(folder / "report.json")
    if file_hash(folder / "manifest.csv") != report["manifest_sha256"] or file_hash(folder / "mapping.json") != report["mapping_sha256"]:
        raise ValueError("Frozen external manifest/mapping changed.")
    _, mapping, pv_report = load_manifest(master)
    if report["class_indices"] != mapping or report["plantvillage_manifest_sha256"] != pv_report["manifest_sha256"]:
        raise ValueError("External report belongs to different PlantVillage splits/labels.")
    frame = pd.read_csv(folder / "manifest.csv")
    for row in frame.itertuples():
        if mapping.get(row.class_name) != row.label or file_hash(confined_path(external_root, row.path)) != row.sha256:
            raise ValueError("External image/label differs from its frozen audit record.")
    return frame, report


def evaluate_external(architecture, master, external_root):
    selection = read_json(master.output / "evaluation" / "selection.json")
    external, audit = load_external_manifest(master, external_root)
    model_path, config, mapping, state, digest = model_artifacts(master, architecture)
    if selection["winners"][architecture]["model_sha256"] != digest:
        raise ValueError("External evaluation requires the frozen validation-selected checkpoint.")
    output = master.output / "evaluation" / f"{architecture}_external.json"
    if output.exists():
        previous = read_json(output)
        if previous.get("model_sha256") != digest or previous.get("external_manifest_sha256") != audit["manifest_sha256"]:
            raise ValueError("Existing external results belong to different artifacts; do not overwrite them.")
        return previous
    model = tf.keras.models.load_model(model_path, compile=False)
    validate_model_contract(model, architecture, mapping, config)
    probabilities = validate_probabilities(model.predict(make_dataset(external, external_root, config), verbose=0), len(mapping), len(external))
    ids = sorted(external.label.unique().tolist())
    result = classification_metrics(external.label.to_numpy(), probabilities, mapping, evaluated_ids=ids)
    # Matching truth classes avoids attributing a class-composition difference
    # entirely to domain shift. The output space always includes ALL model classes.
    internal_frame, _, _ = load_manifest(master, verify_files=True)
    matched = internal_frame[(internal_frame.split == "test") & internal_frame.label.isin(ids)]
    matched_probabilities = model.predict(make_dataset(matched, master.dataset, config), verbose=0)
    matched_result = classification_metrics(matched.label.to_numpy(), matched_probabilities, mapping, evaluated_ids=ids)
    result.update(model=architecture, model_sha256=digest, split="external_test", dataset=audit["dataset_name"],
                  external_manifest_sha256=audit["manifest_sha256"], manifest_sha256=state["signature"]["manifest_sha256"],
                  supported_classes=audit["supported_classes"], unsupported_classes=audit["unsupported_classes"],
                  macro_scope="Classes represented by external ground truth; predictions span every trained class.",
                  matched_internal_accuracy=matched_result["accuracy"], matched_internal_macro_f1=matched_result["macro_f1"],
                  accuracy_gap_internal_minus_external=matched_result["accuracy"] - result["accuracy"],
                  per_class_support_sufficient=all(count >= 20 for count in audit["supported_classes"].values()),
                  support_screen_note="Twenty images per class is a descriptive minimum, not a power/representativeness guarantee.",
                  external_90_percent_point_estimate_met=result["accuracy"] >= .9,
                  target_status="Not established: a screened compatible-class point estimate does not prove representative independent-field accuracy.",
                  independence_confirmed_by_user=audit["independence_confirmed_by_user"], limitation=audit["limitation"])
    profile = load_rejection_profile(model_path.parent, digest, state["signature"]["manifest_sha256"])
    if profile:
        result["rejection"] = rejection_summary(external.label.to_numpy(), probabilities, profile)
    write_json(output, result)
    pd.DataFrame(result["classification_report"]).T.to_csv(master.output / "evaluation" / f"{architecture}_external_classes.csv")
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    predictions = external[["path", "external_label", "class_name", "label"]].copy()
    predictions["predicted_label"] = probabilities.argmax(axis=1)
    predictions["confidence"] = probabilities.max(axis=1)
    predictions.to_csv(master.output / "evaluation" / f"{architecture}_external_predictions.csv", index=False)
    matrix = confusion_matrix(external.label, probabilities.argmax(axis=1), labels=list(range(len(mapping))))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(master.output / "evaluation" / f"{architecture}_external_confusion.csv")
    plot_confusion(matrix, names, master.output / "evaluation" / f"{architecture}_external_confusion.png")
    plot_per_class(result["classification_report"], names, master.output / "evaluation" / f"{architecture}_external_per_class.png")
    return result
