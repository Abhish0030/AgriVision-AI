"""Opt-in paired evaluation on a frozen, independently screened holdout."""
import argparse
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .nemotron_agent import NemotronAgent
from .prediction_orchestrator import ARCHITECTURES, build_context


def summarize(labels, predictions, primary, outcomes, cnn_times, total_times):
    if primary not in ARCHITECTURES or not len(labels) or any(len(predictions[name]) != len(labels) for name in ARCHITECTURES):
        raise ValueError("Expected paired, nonempty classifier results.")
    def metrics(values):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, values, labels=sorted(set(labels)), average="macro", zero_division=0)
        return {"accuracy": float(accuracy_score(labels, values)), "precision_macro": float(precision),
                "recall_macro": float(recall), "macro_f1": float(f1)}
    conditions = {name: metrics(predictions[name]) for name in ARCHITECTURES}
    conditions["CNN_with_analysis"] = metrics(predictions[primary])
    successful = [item for item in outcomes if item["status"] == "ok"]
    attempted = sum(item["status"] != "disabled" for item in outcomes)
    return {"decision_rule": "Condition C always retains the preselected primary CNN class; no LLM override.",
            "primary": primary, "samples": len(labels), "conditions": conditions,
            "macro_scope": "Ground-truth classes present in this holdout; predictions span all trained classes.",
            "cnn_agreement_rate": float(np.mean(np.asarray(predictions[ARCHITECTURES[0]]) == np.asarray(predictions[ARCHITECTURES[1]]))),
            "agent_disagreement_rate_among_valid_responses": (sum(
                item["report"]["agent_assessment"]["agreement_with_cnn"] == "disagree" for item in successful) / len(successful)) if successful else None,
            "valid_agent_responses": len(successful), "attempted_agent_analyses": attempted,
            "api_failure_rate_per_analysis": (sum(item["status"] == "failed" for item in outcomes) / attempted) if attempted else None,
            "agent_latency_seconds_mean": float(np.mean([item["latency_seconds"] for item in outcomes])),
            "cnn_latency_seconds_mean": {name: float(np.mean(cnn_times[name])) for name in ARCHITECTURES},
            "condition_c_latency_seconds_mean": float(np.mean(total_times)),
            "note": "Paired subset only; C classification metrics equal the primary CNN by construction, not evidence of improvement. Agent visual claims are not ground-truth validated."}


def evaluate(paths, external_root, primary, limit=20, seed=42):
    # Import heavy TensorFlow code only when evaluation is explicitly requested.
    from ..external import load_external_manifest
    from ..prediction import PlantDiseasePredictor
    from ..utils import confined_path, write_json
    if primary not in ARCHITECTURES or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Select a supported primary CNN and 1–100 inference-time samples.")
    agent = NemotronAgent(paths.repository)
    if not agent.enabled or agent.mode != "detailed":
        raise ValueError("Configure private credentials/instructions and AGENT_MODE=detailed before this opt-in experiment.")
    external, audit = load_external_manifest(paths, external_root)
    if audit.get("independence_confirmed_by_user") is not True:
        raise ValueError("An independent reviewed external holdout is required.")
    sample = external.sample(n=min(limit, len(external)), random_state=seed).sort_values("path")
    predictors = {name: PlantDiseasePredictor(paths, name) for name in ARCHITECTURES}
    if predictors[ARCHITECTURES[0]].classes != predictors[ARCHITECTURES[1]].classes:
        raise ValueError("CNN class mappings must match.")
    folder = paths.output / "agent_evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder / "plan.json"
    if plan_path.exists():
        raise FileExistsError("Choose a new output experiment; the existing agent plan is frozen.")
    plan = {"primary": primary, "decision_rule": "Retain primary CNN classification unchanged",
            "external_manifest_sha256": audit["manifest_sha256"], "seed": seed,
            "sample_paths": sample.path.tolist(), "truth_classes": sample.class_name.tolist(),
            "model_sha256": {name: predictor.model_hash for name, predictor in predictors.items()}}
    write_json(plan_path, plan)  # Freeze selection and rule before viewing results.
    predictions = {name: [] for name in ARCHITECTURES}
    cnn_times = {name: [] for name in ARCHITECTURES}
    outcomes, totals = [], []
    for row in sample.itertuples():
        image = confined_path(external_root, row.path)
        records, durations = {}, {}
        for name, predictor in predictors.items():
            start = time.perf_counter()
            records[name] = predictor.predict_plant_disease(image, explain=False)
            durations[name] = time.perf_counter() - start
            predictions[name].append(records[name]["class_name"])
            cnn_times[name].append(durations[name])
        outcome = agent.analyze(image, build_context(records, primary))
        totals.append(sum(durations.values()) + outcome["latency_seconds"])
        outcomes.append(outcome)
    result = summarize(sample.class_name.tolist(), predictions, primary, outcomes, cnn_times, totals)
    result["provenance"] = plan
    result["latency_scope"] = "Single-image classification plus encoding/network/validation for C; excludes model loading and Grad-CAM for all conditions. Cold first inference included."
    write_json(folder / "summary.json", result)  # Aggregates only: no requests, prompts or raw reports.
    return result


def main():
    from ..config import ProjectPaths
    parser = argparse.ArgumentParser(description="Explicitly send a small held-out subset for paired visual analysis.")
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary", choices=ARCHITECTURES, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    paths = ProjectPaths.create(args.dataset, args.output)
    result = evaluate(paths, args.external_root, args.primary, args.limit)
    print("Paired evaluation saved locally. Samples:", result["samples"])


if __name__ == "__main__":
    main()
