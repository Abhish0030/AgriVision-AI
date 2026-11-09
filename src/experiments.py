"""Controlled validation-only experiments followed by immutable final selection."""
from dataclasses import asdict, replace
import json
import re
import pandas as pd
import tensorflow as tf
from .artifacts import model_artifacts, validate_evaluation_record
from .config import ProjectConfig, ProjectPaths
from .dataset import load_manifest
from .evaluation import compare_models, evaluate_architecture
from .training import train_architecture
from .utils import atomic_copy, file_hash, read_json, training_source_hash, write_json
from .validation import score_validation


def build_trial_plan(base, include_optimizer=False, include_fine_tuning=False, include_smoothing=False):
    trials = []
    for architecture in base.architectures:
        candidates = {"no_augmentation": {"augmentation_profile": "none"},
                      "mild_baseline": {"augmentation_profile": "mild"},
                      "camera_robustness": {"augmentation_profile": "robust"}}
        if include_optimizer:
            candidates["adamw"] = {"augmentation_profile": "mild", "optimizer": "AdamW"}
        if include_fine_tuning:
            candidates["deeper_tail"] = {"augmentation_profile": "mild", "fine_tune_layers": base.fine_tune_layers + 20}
        if include_smoothing:
            candidates["label_smoothing"] = {"augmentation_profile": "mild", "label_smoothing": .05}
        for name, changes in candidates.items():
            config = replace(base, checkpoint_monitor="val_macro_f1", **changes)
            trials.append({"trial_id": f"{architecture}_{name}", "architecture": architecture,
                           "config": json.loads(json.dumps(asdict(config)))})
    return trials


def trial_paths(master, trial_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,90}", trial_id):
        raise ValueError("Use a simple safe trial identifier.")
    return ProjectPaths(master.repository, master.dataset, master.output / "trials" / trial_id).initialize()


def trial_config(trial):
    values = dict(trial["config"])
    for key in ("crops", "architectures", "image_size", "input_pixel_range"):
        values[key] = tuple(values[key])
    return ProjectConfig(**values)


def initialize_trial_plan(master, base, trials):
    _, _, report = load_manifest(master, config=base)
    if not trials or len({item["trial_id"] for item in trials}) != len(trials):
        raise ValueError("A plan needs unique, nonempty trials.")
    payload = {"trials": trials, "manifest_sha256": report["manifest_sha256"], "source_sha256": training_source_hash(),
               "selection_metric": "validation macro F1; then balanced accuracy; then lower NLL; then trial ID"}
    path = master.output / "runs" / "trial_plan.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("The trial plan/source/split changed. Start a new experiment directory.")
    if not path.exists():
        write_json(path, payload)
    for trial in trials:
        config = trial_config(trial)
        destination = trial_paths(master, trial["trial_id"])
        for name in ("manifest.csv", "class_indices.json", "report.json"):
            source = master.output / "dataset" / name
            target = destination.output / "dataset" / name
            if target.exists() and file_hash(target) != file_hash(source):
                raise ValueError("A trial's frozen manifest differs from the master split.")
            if not target.exists():
                atomic_copy(source, target)
        load_manifest(destination, config=config)
    return payload


def run_trials(master, base, trials, only_trial_ids=None):
    if (master.output / "evaluation" / "selection.json").exists():
        raise ValueError("Selection is frozen. Do not add/tune trials after looking at held-out results.")
    initialize_trial_plan(master, base, trials)
    chosen = set(only_trial_ids) if only_trial_ids is not None else {item["trial_id"] for item in trials}
    if not chosen.issubset({item["trial_id"] for item in trials}):
        raise ValueError("Requested trial is not in the saved plan.")
    for trial in trials:
        if trial["trial_id"] not in chosen:
            continue
        tf.keras.backend.clear_session()
        config = trial_config(trial)
        tf.keras.utils.set_random_seed(config.seed)
        paths = trial_paths(master, trial["trial_id"])
        train_architecture(trial["architecture"], paths, config, resume=True)
        score_validation(trial["architecture"], paths, config)
    # Rebuild a validation-only registry, including completed trials from earlier sessions.
    registry = []
    for trial in trials:
        paths = trial_paths(master, trial["trial_id"])
        path = paths.output / "evaluation" / f"{trial['architecture']}_validation.json"
        if path.exists():
            result = read_json(path)
            registry.append({"trial_id": trial["trial_id"], "architecture": trial["architecture"],
                             "config": trial["config"], "validation": result})
    write_json(master.output / "evaluation" / "validation_experiments.json", registry)
    table = pd.DataFrame([{**item["config"], "trial_id": item["trial_id"], "model": item["architecture"],
                           **{key: item["validation"][key] for key in ("accuracy", "macro_f1", "balanced_accuracy", "negative_log_likelihood", "training_seconds")}}
                          for item in registry])
    if not table.empty:
        for architecture in table.model.unique():
            mask = table.model == architecture
            baseline = table[mask & table.trial_id.str.endswith("_mild_baseline")]
            if len(baseline) == 1:
                table.loc[mask, "delta_macro_f1_vs_mild"] = table.loc[mask, "macro_f1"] - baseline.iloc[0].macro_f1
        def interpret_delta(delta):
            if pd.isna(delta):
                return "Baseline not finished"
            return "higher observed validation F1" if delta > 0 else "lower observed validation F1" if delta < 0 else "same observed validation F1"
        table["interpretation"] = table.delta_macro_f1_vs_mild.map(interpret_delta) if "delta_macro_f1_vs_mild" in table else "Baseline not finished"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = table.set_index("trial_id")[["accuracy", "macro_f1", "balanced_accuracy"]].plot.barh(xlim=(0, 1), figsize=(11, max(4, len(table) * .5)))
        ax.figure.tight_layout()
        ax.figure.savefig(master.output / "evaluation" / "validation_experiments.png", dpi=150)
        plt.close(ax.figure)
    table.to_csv(master.output / "evaluation" / "validation_experiments.csv", index=False)
    return table


def select_best_trials(master):
    plan_path = master.output / "runs" / "trial_plan.json"
    plan = read_json(plan_path)
    selected_path = master.output / "evaluation" / "selection.json"
    if selected_path.exists():
        selection = read_json(selected_path)
        if selection["plan_sha256"] != file_hash(plan_path):
            raise ValueError("Frozen selection belongs to a different plan.")
        return selection
    candidates = []
    for trial in plan["trials"]:
        paths = trial_paths(master, trial["trial_id"])
        _, _, _, state, digest = model_artifacts(paths, trial["architecture"])
        validation = read_json(paths.output / "evaluation" / f"{trial['architecture']}_validation.json")
        if validation.get("split") != "validation" or validation.get("model_sha256") != digest or validation.get("manifest_sha256") != plan["manifest_sha256"]:
            raise ValueError("Every planned trial must finish with a matching validation-only report before selection.")
        candidates.append({"trial_id": trial["trial_id"], "architecture": trial["architecture"], "model_sha256": digest,
                           "validation_macro_f1": validation["macro_f1"], "validation_balanced_accuracy": validation["balanced_accuracy"],
                           "validation_nll": validation["negative_log_likelihood"], "training_seconds": state["training_seconds"]})
    def rank(row):
        return (-row["validation_macro_f1"], -row["validation_balanced_accuracy"], row["validation_nll"], row["trial_id"])
    ordered = sorted(candidates, key=rank)
    winners = {}
    for row in ordered:
        winners.setdefault(row["architecture"], row)
    selection = {"plan_sha256": file_hash(plan_path), "manifest_sha256": plan["manifest_sha256"],
                 "source_sha256": plan["source_sha256"], "basis": plan["selection_metric"], "validation_only": True,
                 "winners": winners, "overall_best_architecture": ordered[0]["architecture"], "all_validation_trials": candidates}
    write_json(selected_path, selection)
    return selection


def export_selected_models(master):
    selection = read_json(master.output / "evaluation" / "selection.json")
    versions = {}
    for architecture, winner in selection["winners"].items():
        source_paths = trial_paths(master, winner["trial_id"])
        source_model, _, _, state, digest = model_artifacts(source_paths, architecture)
        if digest != winner["model_sha256"]:
            raise ValueError("Selected checkpoint changed after freezing selection.")
        folder = master.output / "models" / architecture
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "state.json").exists() and read_json(folder / "state.json").get("exported_from_trial") != winner["trial_id"]:
            raise FileExistsError("Destination contains a different model; use a new optimization experiment directory.")
        for filename in ("best.keras", "config.json", "class_indices.json", "preprocessing.json", "history.csv", "rejection.json"):
            atomic_copy(source_model.parent / filename, folder / filename)
        state.update(best_checkpoint="best.keras", latest_checkpoint="best.keras", exported_from_trial=winner["trial_id"])
        write_json(folder / "state.json", state)
        atomic_copy(source_paths.output / "graphs" / f"{architecture}_training.png", master.output / "graphs" / f"{architecture}_training.png")
        atomic_copy(source_paths.output / "graphs" / f"{architecture}_training.png", master.output / "evaluation" / f"{architecture}_training.png")
        atomic_copy(source_model.parent / "history.csv", master.output / "evaluation" / f"{architecture}_history.csv")
        atomic_copy(source_paths.output / "evaluation" / f"{architecture}_validation.json", master.output / "evaluation" / f"{architecture}_validation.json")
        alias = "efficientnet_best.keras" if architecture == "EfficientNetB0" else "mobilenet_best.keras"
        atomic_copy(folder / "best.keras", master.output / "models" / alias)
        versions[architecture] = {**winner, "version": f"{winner['trial_id']}:{digest[:12]}", "alias": alias}
    write_json(master.output / "models" / "model_versions.json", versions)
    return versions


def evaluate_selected_models(master):
    selection = read_json(master.output / "evaluation" / "selection.json")
    export_selected_models(master)
    for architecture in selection["winners"]:
        _, config, _, state, digest = model_artifacts(master, architecture)
        path = master.output / "evaluation" / f"{architecture}.json"
        if path.exists():
            validate_evaluation_record(read_json(path), architecture, digest, state["signature"]["manifest_sha256"])
        else:
            tf.keras.backend.clear_session()
            evaluate_architecture(architecture, master, config)
    compare_models(master, tuple(selection["winners"]), allow_config_differences=True)
    return generate_final_comparison(master)


def generate_final_comparison(master):
    selection = read_json(master.output / "evaluation" / "selection.json")
    manifest, _, _ = load_manifest(master)
    metrics = {"Internal Test Accuracy": "accuracy", "Internal Macro F1": "macro_f1", "Internal Balanced Accuracy": "balanced_accuracy",
               "Internal Precision (macro)": "precision_macro", "Internal Recall (macro)": "recall_macro", "Inference Time (median ms)": "inference_ms_median"}
    results = {}
    for architecture in selection["winners"]:
        _, _, _, state, digest = model_artifacts(master, architecture)
        internal_path = master.output / "evaluation" / f"{architecture}.json"
        external_path = master.output / "evaluation" / f"{architecture}_external.json"
        internal = read_json(internal_path) if internal_path.exists() else {}
        external = read_json(external_path) if external_path.exists() else {}
        if internal:
            validate_evaluation_record(internal, architecture, digest, state["signature"]["manifest_sha256"])
        if external:
            external_report = read_json(master.output / "external" / "report.json")
            if external.get("model_sha256") != digest or external.get("external_manifest_sha256") != external_report["manifest_sha256"]:
                raise ValueError("Final table contains stale external artifacts; refusing to report them.")
        results[architecture] = {title: internal.get(key) for title, key in metrics.items()}
        results[architecture].update({"External Test Accuracy": external.get("accuracy"), "External Macro F1 (compatible classes)": external.get("macro_f1"),
                                      "External images": external.get("samples"), "Training duration (seconds)": state["training_seconds"],
                                      "Training images": int((manifest.split == "train").sum()),
                                      "Internal test images": internal.get("test_images"), "External 90% target verified": "Not established"})
    write_json(master.output / "evaluation" / "final_comparison.json", results)
    table = pd.DataFrame(results)
    table.fillna("Not evaluated").to_csv(master.output / "evaluation" / "final_comparison.csv", index_label="Metric")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = pd.DataFrame({name: {"Internal": row["Internal Test Accuracy"], "External": row["External Test Accuracy"]} for name, row in results.items()}).T
    if values["Internal"].notna().any():
        ax = values.astype(float).plot.bar(ylim=(0, 1), figsize=(8, 4), ylabel="Unconditional accuracy")
        ax.figure.tight_layout()
        ax.figure.savefig(master.output / "evaluation" / "internal_vs_external.png", dpi=150)
        plt.close(ax.figure)
    return table.fillna("Not evaluated")
