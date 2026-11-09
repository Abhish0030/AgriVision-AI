# Initial repository audit — before implementation fixes

Date: 10 November 2026. Inspected all 20 Python files, the actual 28-cell
notebook (13 code cells), dependency/configuration files, 21 disease records,
all documentation and Git-eligible files. Existing useful implementations will
be preserved. This report records the pre-fix state; see AUDIT_FINAL.md for results.

Git: branch main; no commits, no remote; 38 untracked source files. Nothing was
staged, committed, pushed or removed. Local `.venv` and validation outputs exist
but are ignored. No real PlantVillage dataset or trained disease model is present.

## Requirement evidence

PASS means inspected implementation plus relevant local execution evidence;
it does not imply real-dataset performance, GPU or Colab verification. PARTIAL
means implementation exists with a material gap. NOT TESTED identifies an
environment-dependent acceptance test that has not run.

| Mandatory component | Initial status | Evidence / gap |
|---|---|---|
| Environment setup | PASS | TensorFlow/Keras setup, CPU fallback, GPU-capability policy implemented; prior CPU execution and current imports. T4 remains NOT TESTED. |
| PlantVillage handling | PARTIAL | Real folder schema, integrity hashes, synthetic audit tests pass; mapping/schema validation incomplete; real data NOT TESTED. |
| Four-crop selection | PASS | Tomato, Potato, Corn/maize, Apple discovered from folders; no rice assumption or invented classes. |
| Preprocessing | PASS | Shared RGB/Pillow + OpenCV resize; built-in backbone preprocessing, raw [0,255]; no double rescaling. |
| Training-only augmentation | PASS | Keras transforms applied only when training=True; deterministic validation/test path. |
| EfficientNetB0 | PASS | Default ImageNet request, correct head; random-initialized real backbone forward/update/save/load tested. Actual ImageNet load not yet tested. |
| MobileNetV3Large | PASS | Explicit include_preprocessing=True; real backbone forward/update/save/load tested without weight download. |
| Transfer learning | PASS | Frozen backbone and trainable float32 head; production requests ImageNet weights. |
| Fine-tuning | PASS | Configurable tail, frozen BatchNorm, lower LR and recompilation; CPU update tested. |
| Reusable training | PASS | Shared two-stage function; tiny synthetic integration test, not evidence of disease performance. |
| Checkpointing | PARTIAL | Full-model epoch state exists, but best.keras is overwritten in place and nonfinite logs are checked too late. |
| Evaluation | PARTIAL | Actual sklearn metrics saved; metadata/probability validation incomplete. |
| Confusion matrix | PASS | Fixed-label numeric CSV and rendered matrix exercised by synthetic integration. |
| Model comparison | FAIL | Probe accepts one evaluation without any checkpoint; stale artifacts and unlike configs can be compared. |
| Grad-CAM | PASS | Actual nested EfficientNetB0 and MobileNetV3Large maps successfully tested with finite values and matching overlay shape. |
| Single-image prediction | PARTIAL | Runs with temporary fixture checkpoint; config file not checked against training signature. |
| Confidence warning | PASS | Softmax threshold/top-three tested; clear uncalibrated-confidence limitation. |
| Disease information | PARTIAL | 17 affected records and 4 explicit healthy fallbacks; target-spot primary link redirects away from its source. |
| Gradio | PARTIAL | Blocks construction and direct callback tested; actual HTTP launch/upload not yet tested. |
| Complete Colab notebook | PARTIAL | Logical cells, modular imports, Drive paths and no fabricated outputs; dependency preservation/checkout identity insufficient; fresh Colab NOT TESTED. |
| Documentation | PARTIAL | Useful guides/viva exist; README needs explicit problem/objectives/stack/results/architecture summaries. |
| Dependencies | PARTIAL | Local pip check passes; Colab requirements exclude TF but do not constrain its installed dependency stack. |

## Detailed findings and fixes

| ID / area | Status | Issue and impact | Recommended fix |
|---|---|---|---|
| A01 Checkpoint transactions | HIGH | PersistentEpoch saves directly over best.keras. A failed save can destroy the last valid best model before state commits. Nonfinite training/validation values can be serialized to a resume model before JSON rejects them. | Use atomic file replacement and a versioned best-checkpoint pointer committed with state; validate epoch values before writes; preserve prior committed files until state succeeds. |
| A02 Frozen dataset metadata | HIGH | load_manifest verifies only the manifest CSV hash, not mapping/label agreement, split schema, group isolation or path confinement. A swapped mapping was accepted in a temporary probe. | Validate all metadata relations, contiguous indices, required splits/classes, group isolation, relative paths and integrity; record split config and mapping digest. |
| A03 Model artifacts | HIGH | Prediction trusts config.json without comparing it to the signature/model input shape; evaluation does not fully compare mapping/signature/model shape. | Centralize artifact checks for architecture, config, mapping, completion, model hash and tensor contract. |
| A04 Comparison provenance | HIGH | compare_models silently skips missing models and accepts evaluation files without matching checkpoints. Probe produced a one-model table. Only manifest equality is checked. | Require every requested evaluation; validate against current completed checkpoints and matching experiment config/mapping; disclose timing comparability. |
| A05 Colab bootstrap | HIGH | Excluding tensorflow from pip requirements does not stop NumPy/Keras/protobuf dependency changes. Existing checkout identity is not checked. | Constrain installed TF and its core dependencies during installation, check pip compatibility, validate reused checkout URL/dirty state and imported source location. |
| A06 Configuration | MEDIUM | ProjectConfig accepts NaN split fractions (reproduced), invalid dropout/LR values and wrong numeric types/dimension counts. | Validate finite scalars, integer fields, image shape/backbone size, choices and ranges before creating experiment artifacts. |
| A07 Inference and metrics errors | MEDIUM | Malformed evaluation JSON can crash after the app's protected prediction block. Probability shape/range/sum are not validated. | Handle unavailable/corrupt evaluation honestly, validate probabilities and artifact metadata, test invalid uploads/missing models. |
| A08 Extraction interruptions | MEDIUM | Extraction writes directly into the final folder, and notebook skips any existing directory. Interruption can leave a partial dataset mistaken for complete. | Extract into a unique staging directory and rename on success; validate archive names consistently across OSs. Preserve failed staging evidence. |
| A09 Source verification | HIGH | Target-spot primary PDF redirects to an unrelated holding page; additional HTML page loops in this browser. | Use the readable official current PDF at https://ask.ifas.ufl.edu/publication/PP351.pdf and recheck the summary. |
| A10 Publication/documentation | MEDIUM | README lacks explicit requested summaries; venv/, node_modules/ and several credential/cache variants are not excluded. | Expand README and narrowly extend ignore rules; keep useful source and placeholders. |
| A11 Acceptance evidence | MEDIUM | Existing three tests pass but do not test network launch/upload, corrupt metadata, interruption inside saving, actual ImageNet weights or all module imports. | Add focused regressions and a local HTTP launch/client check; separately run a small real-backbone ImageNet probe without downloading datasets. |
| A12 External-runtime acceptance | NOT TESTED | No authorized live Colab/Drive/T4 session or real PlantVillage experiment is available in this workspace. | User must run the notebook and preserve real results; do not describe local CPU checks as Colab/GPU success. |

No CRITICAL secret exposure, destructive action or absent core model implementation
was identified. HIGH findings should be fixed before publication. MEDIUM findings
will be addressed where testable. LOW optional improvements include a chosen
software license, multi-seed experiments, external PlantDoc evaluation and calibration;
none is silently implemented or claimed as an experiment result.

## Baseline checks actually executed during this audit

- Python/notebook syntax and nbformat schema: PASS.
- Ruff selected correctness rules: PASS.
- pip check: PASS.
- Existing unittest suite: 3 tests PASS in 33.671 seconds on Windows CPU.
- Focused temporary probes: reproduced NaN acceptance, swapped mapping acceptance
  and one-model/no-checkpoint comparison acceptance.
- Git-eligible inventory: no credential-pattern matches, model/dataset artifacts
  or files above 1 MB. This is pattern-based review, not a guarantee against all secrets.
- Disease references: all primary sources readable except the target-spot redirect;
  a current official PDF was found. Review evidence will be saved with final report.

## Scope limits

Repository clone from the eventual remote, Drive mounting/checkpoints on Drive,
T4 execution, real PlantVillage training/evaluation and Colab-hosted Gradio remain
NOT TESTED. Their existence in notebook code is not execution evidence.
