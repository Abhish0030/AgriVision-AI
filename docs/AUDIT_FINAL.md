# Final technical audit and GitHub readiness

Date: 10 November 2026. Read together with [the initial audit](AUDIT_INITIAL.md),
[validation detail](VALIDATION.md) and [source review](SOURCE_VERIFICATION.md).

**Decision: READY AFTER MANUAL VERIFICATION.** Source implementation and local
checks pass. Review the staged source/remote before publication. Fresh Colab,
Drive/T4 operation and real PlantVillage experiments remain unverified and must
not be advertised as completed. The repository may be published as an untrained
research pipeline with these explicit limits; it is not a validated field-diagnosis
product or a completed experimental-results submission.

## A. Completion and evidence

There are **22 mandatory components** in the request's requirements table.
All 22 have substantive implementation: **22/22 = 100% implementation coverage**.
After fixes, **20 PASS and 2 PARTIAL** under the evidence criteria below:
**20/22 = 90.9% fully locally checked components**. There are zero missing/FAIL
components. These percentages are checklist coverage, not accuracy, convergence,
test coverage percentage or a claim of full Colab acceptance.

PASS = inspected implementation with relevant executed local tests; PARTIAL =
implemented but its full environment-specific acceptance remains incomplete.
Real-data/GPU outcomes are listed separately as NOT TESTED, even for PASS code.

| Component | Final status | Actual implementation / evidence |
|---|---|---|
| Environment setup | PASS | TensorFlow/Keras versions, hardware/build inspection, capability-gated mixed precision, float32 CPU fallback; synchronized CPU check executed. T4 branch NOT TESTED. |
| Dataset handling | PASS | Actual PlantVillage folder convention, RGB validation, corruption audit and frozen relative manifests; synthetic fixtures executed. Real PlantVillage NOT TESTED. |
| Crop selection | PASS | Tomato, Potato, Corn/maize, Apple; required selected crops checked; no rice requirement or fabricated class list. |
| Preprocessing | PASS | Shared Pillow RGB/OpenCV resize and [0,255] tensors; built-in preprocessing; deterministic validation equals inference resize in tests. |
| Augmentation | PASS | Keras transforms only in training dataset path; validation/test deterministic; no extra normalization. |
| EfficientNetB0 | PASS | Actual ImageNet download/load, 224×224 forward pass, matching three-class test head and float32 output; no disease training claimed. |
| MobileNetV3Large | PASS | Actual ImageNet download/load, explicit built-in preprocessing and 224×224 forward pass; no disease training claimed. |
| Transfer learning | PASS | Frozen pretrained backbone and trainable custom head; real-backbone CPU training update and full-model reload tested. |
| Fine-tuning | PASS | Selected tail, lower LR/recompile, BatchNorm frozen; real-backbone CPU update and pretrained BatchNorm assertions tested. |
| Reusable training | PASS | One two-stage function, training-only class weights, early stop/LR reduction; temporary tiny-model integration executes both stages. |
| Checkpointing | PASS | Atomic saves, immutable best pointer, repairable alias, optimizer/epoch resumption and fault-injection checks. Actual Drive writes NOT TESTED. |
| Evaluation | PASS | Accuracy, macro precision/recall/F1, weighted F1, per-class JSON/CSV, output validation and provenance checked using temporary fixtures. |
| Confusion matrix | PASS | Fixed class order, numeric CSV and plot tested. |
| Model comparison | PASS | Requires all requested completed models and matching saved evaluations/config/mapping/source/splits. Missing and stale artifacts rejected. Timing context flag added. |
| Grad-CAM | PASS | Both actual nested backbones tested before/after reload and with ImageNet weights; analytic fixture verifies formula, positive normalization and zero negative class. |
| Prediction | PASS | Correct checkpoint/config/mapping/shape/hash/source checks; single image and top-three tested. |
| Uncertainty | PASS | Normalized softmax scores, configurable threshold, explicit low-score warning; uncalibrated limitation preserved. |
| Disease information | PASS | 17 source-backed affected records, 4 explicit healthy fallbacks, dynamic fallback for other labels; 14 unique current references reviewed; target-spot link fixed. |
| Gradio | PASS | Blocks, callbacks and actual localhost HTTP server/client image upload; prediction and three image outputs retrieved. Browser visual/Colab sharing NOT TESTED. |
| Main Colab notebook | PARTIAL | All 13 code cells, order, imports, safe bootstrap, Drive paths, both models, evaluation and demo inspected; schema/syntax clean. Fresh remote clone/Colab/Drive/T4 execution NOT TESTED. |
| README/documentation | PASS | Requested overview/problem/objectives/stack/dataset/architecture/setup/training/evaluation/Grad-CAM/demo/results-pending/limits/future/references included; dedicated guides and audit records. |
| Dependencies | PARTIAL | Local installation/imports/pip check pass; constrained installer preserves installed TF stack locally. A fresh real Colab dependency resolution remains NOT TESTED. |

## B. Findings resolved

| Area / initial ID | Severity | Final status | Fix and regression evidence |
|---|---|---|---|
| Checkpoint transactions A01 | HIGH | FIXED | Save to unique temporary .keras and rename. Commit immutable epoch pointer before publishing best.keras. Preserve old files/state after simulated save/state failures; reject nonfinite logs before saving. |
| Manifest metadata A02 | HIGH | FIXED | Mapping digest/schema, label agreement, class coverage, split/group isolation, safe relative paths, split config and image hashes validated. Swapped-label/path/split regressions reject bad metadata. |
| Model integrity A03 | HIGH | FIXED | Shared artifacts module checks architecture, completion, signature config/mapping, model hash, source fingerprint and input/output contract. Probability tensors checked. |
| Model comparison A04 | HIGH | FIXED | Missing evaluations no longer skipped; current checkpoint identities and complete experiment signatures required. Invalid/stale files rejected; recorded timing-context match exposed separately. |
| Colab bootstrap A05 | HIGH | FIXED locally | Installed TF/Keras/NumPy/protobuf/ml-dtypes/h5py are constrained during pip resolution. Local installer passes. Wrong/dirty/stale checkouts rejected; actual Colab gate remains. |
| Configuration A06 | MEDIUM | FIXED | Finite values, numeric types, ranges, class/model choices, minimum image dimensions and lower fine-tune LR validated. NaN probe now fails correctly. |
| Inference/metric errors A07 | MEDIUM | FIXED | Corrupt/unreadable/stale evaluation returns unavailable status; malformed probabilities rejected; missing model and invalid image tested. |
| Extraction A08 | MEDIUM | FIXED | Extract into a unique staging folder and publish the destination only after success; cross-platform traversal/symlink guards retained. Failed extraction preserves diagnostic staging files. |
| Disease reference A09 | HIGH | FIXED | Replaced obsolete target-spot URL/looping alternate with the readable official PP351 PDF; prevention text checked against current cultural-management guidance. |
| Publication/docs A10 | MEDIUM | FIXED | README expanded; ignore rules cover additional environments, node_modules, key/certificate/token files. Repeatable publication inventory added; no user files deleted. |
| Evidence A11 | MEDIUM | FIXED locally | Expanded from 3 baseline tests to 12, actual ImageNet checks, analytic Grad-CAM and localhost HTTP upload. No huge dataset download. |
| Runtime acceptance A12 | NOT TESTED | OPEN manual | Fresh Colab remote clone, Drive mount/writes/resume and T4 mixed-precision training require the user's runtime. |
| Additional provenance review | MEDIUM | FIXED | Training-affecting source fingerprint and immutable original Colab commit/config prevent silent experiment reuse after implementation changes. |

No unresolved CRITICAL or HIGH source defect is known after these checks. LOW
future work includes choosing a software license, additional seed experiments,
external datasets and calibration. A license was not invented on the owner's behalf.
No feature was removed to pass tests.

## C. Executed tests and results

Baseline run: original 3 unittest tests PASS in 33.671 seconds. Focused temporary
pre-fix probes reproduced NaN acceptance, mismatched mapping acceptance and a
one-model comparison with no checkpoint. Those failures led to regression tests.

Final unittest run: **12 tests PASS in 44.009 seconds**, Windows CPU,
Python 3.13.7 / TensorFlow 2.21.0 / Keras 3.15.1 / NumPy 2.2.6 / Gradio 5.50.0.

| Executed unittest | Result |
|---|---|
| test_archive_and_configuration_guards | PASS — config round-trip, split validation, malicious ZIP rejection. |
| test_dataset_pipeline_and_saved_metrics | PASS — corruption/exact/near matches, splitting, two stages, simulated post-epoch disconnect/resume, prediction/metrics/comparison and changed-file rejection. |
| test_real_backbones_gradcam_training_and_reload | PASS — both real randomly initialized backbones, forward/head update, full-model/optimizer reload, Grad-CAM and fine-tuning update. |
| test_import_every_module_and_preserve_stack_constraints | PASS — all src imports, installed stack constraints and immutable experiment provenance. |
| test_invalid_config_and_probability_contract | PASS — NaN/infinite/range/type/shape/choice guards and normalized probability contract. |
| test_gradcam_matches_an_analytic_positive_and_negative_class | PASS — exact expected map, default/explicit class selection and correct zero negative-class map. |
| test_checkout_validation_with_temporary_git_repository | PASS — correct source identity accepted, other remote and dirty tree rejected; only a synthetic temporary repository is committed. |
| test_manifest_integrity_labels_paths_and_split_configuration | PASS — swapped labels, traversal, invalid split/label and changed split configuration rejected. |
| test_validation_preprocessing_is_deterministic_and_shared | PASS — repeat validation tensors match and equal inference resize, with raw pixel range retained. |
| test_atomic_checkpoint_and_failed_state_commit_preserve_best | PASS — interrupted model save/state commit preserve prior best; NaN epoch rejected; alias repaired. |
| test_archive_interruption_and_cross_platform_paths | PASS — partial extraction not published; Windows/POSIX escapes rejected; valid extraction succeeds. |
| test_artifact_tampering_comparison_and_http_upload | PASS — missing model/evaluation, source/config tampering, stale/corrupt metrics and invalid image handling; localhost Gradio launch/upload/prediction/three image artifacts. |

Other actual checks:

- Python AST + all notebook cells + nbformat schema + disease-record schema: PASS.
- Ruff selected correctness rules: PASS.
- pip dependency consistency: PASS.
- Constrained supporting-library installation into the existing isolated local
  environment: PASS, no TensorFlow-stack version changes. Not real Colab.
- Environment command: synchronized CPU execution, float32 fallback and saved
  environment/dependency metadata. No usable TensorFlow GPU in native Windows.
- ImageNet script: **EfficientNetB0 PASS and MobileNetV3Large PASS**, both 224×224,
  float32 three-class untrained heads and nested Grad-CAM. Layers were top_conv and
  conv_1 respectively. A MobileNet random-head/class map was zero; this is reported
  as uninformative, not fabricated. Analytic tests additionally verify nonzero maps.
- Disease source review: 14 unique current primary references readable;
  obsolete target-spot redirect detected and replaced. See SOURCE_VERIFICATION.md.
- Git-eligible source inventory, credential-pattern review, large-artifact and
  ignore checks: no detected publication blockers. This is not a formal CVE audit
  or proof that every possible secret format has been detected.

Ignored local logs and metadata are under outputs/audit-*. Synthetic metrics and
temporary trained fixture files are not project research results and are not
published. Application-weight caches remain ignored under outputs/audit-cache.
Third-party TensorFlow/Keras/Gradio warnings occurred; no failing test is concealed.

## D. Remaining issues and manual actions

1. Create/select the intended GitHub repository, inspect current remotes and staged
   source before pushing. Current workspace has no origin or commits. The audit
   did not stage/commit/push this project or overwrite Git history.
2. Run the notebook in a fresh actual Colab runtime: clone your remote, constrain
   dependencies, authorize Drive, verify the actual T4 device/precision, and confirm
   persistent saves/resumption. Local HTTP launch is not Colab acceptance.
3. Supply actual original-color PlantVillage with recorded source/license, review
   classes/duplicates/quarantines and split proportions, then train/evaluate both
   architectures. No real accuracy, runtime estimate or field-generalization result
   is available before that work. dHash still cannot guarantee all biological
   independence or catch every transformed duplicate.
4. Review the actual browser interface and collect real demo screenshots/results
   for the report. No screenshot was manufactured during this audit.
5. If reusing pre-audit artifacts, preserve them and re-audit into a new output
   directory: older files lack the new integrity/source records. Do not hand-edit
   signatures to make unrelated models appear compatible.

These limits are visible in README and guides; experimental results remain
pending. Optional license choice, PlantDoc, calibration and multi-seed research
do not justify claiming that an unrun experiment has passed.

## E. GitHub preparation

Run these yourself after reviewing the source and setting the correct origin.
Add origin only if none exists; never overwrite an existing remote automatically.

```powershell
git status
git remote -v
git remote add origin https://github.com/YOUR_USERNAME/AgriVision-AI.git
git add .
git diff --cached --stat
git diff --cached
git commit -m "Initial release: AgriVision AI"
git push origin main
```

Replace YOUR_USERNAME. Omit remote add when origin already points to the intended
repository. The requested core sequence is git status → git add . → git commit
with the message above → git push origin main. No automatic push, forced history
change or deletion of existing user files was performed.
