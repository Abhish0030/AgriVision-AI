# Optimization validation and experiment status

Date: 10 November 2026. This is software verification, not measured disease-model
performance. See [the plan](OPTIMIZATION_PLAN.md) and [Colab guide](TRAINING_GUIDE.md).

## Actual experiment status

| Requested outcome | Current evidence |
|---|---|
| Models trained on real data | None; both training pipelines are prepared |
| Real dataset used | None supplied; data/ contains only .gitkeep |
| Real training/testing image counts | Not available until dataset audit |
| Actual internal accuracy / macro F1 | Not evaluated |
| Actual external accuracy / macro F1 | Not evaluated |
| Best-performing architecture | Unknown until validation experiments |
| Actual GPU training duration | Not measured |
| 90% independent real-world target | Not verified; research target only |
| Colab/T4/Drive execution | Not executed; no connected runtime was available |

Cached ImageNet weights and temporary CPU models are not trained project disease
checkpoints. Synthetic scores are not copied into README or research results.
Real training must run in Colab using the user's actual data.

## Executed local checks

- Existing 12-test regression suite passed after core training/metric changes:
  47.556 seconds on Windows CPU.
- Expanded full suite: **15 tests PASS**, 112.754 seconds. It covers the prior audit,
  real-backbone forward/update/save/load, dataset integrity, Grad-CAM, inference
  and a localhost Gradio server/client image upload.
- After final artifact-layout/count changes, the three optimization tests were
  rerun: **3 PASS**, 33.098 seconds.
- AST/notebook syntax, nbformat and disease schema: PASS, including 31 Python
  files and all 15 notebook code cells. No notebook outputs are stored.
- Ruff selected correctness checks and pip dependency consistency: PASS.
  No new runtime dependency was required.
- Both real architectures loaded actual ImageNet weights at 224×224 and passed
  CPU forward/Grad-CAM/BatchNorm checks. Layers were top_conv and conv_1.
  Zero positive maps are reported rather than concealed.
- Publication inventory found no detected credentials, datasets, checkpoints,
  large artifacts or broken local links among Git-eligible files. Pattern checks
  cannot prove the absence of every secret format.

## New optimization test evidence

1. Known two-class metrics reproduce accuracy, balanced accuracy, macro F1 and
   ECE. Rejection fitting refuses inadequate validation support and finds an
   operating point on a deliberately high-quality synthetic fixture. Conditional
   accepted accuracy never replaces unconditional accuracy.
2. None/mild/robust augmentation outputs are finite and bounded. AdamW and optional
   smoothed sparse-label loss train a tiny temporary model, save/load and continue
   updating. The registered macro-F1 metric matches a known confusion case.
3. Six temporary trials execute both stages. Test evaluation is patched to fail
   if called during trials; it is not called. Winners follow validation F1 and
   frozen selection prevents adding trials. Exports, version metadata and requested
   root aliases load. Internal testing occurs after selection.
4. External mapping approval is required. A copied PlantVillage image, an external
   duplicate and unsupported labels are excluded. Remaining external fixtures
   never alter training manifests. Both selected models use the complete trained
   output space, compatible truth-class metrics, matched internal comparisons and
   explicit small-sample/target limitations. Modified files fail integrity checks.
5. Validation rejection/version metadata loads in prediction. Matching saved
   internal/external metrics display. Error templates, human-provided subgroup tags
   and correct/incorrect Grad-CAM examples are exercised.

## Still requiring actual Colab/data

- T4 mixed precision, speed, convergence, real class/image counts and distributions.
- Drive authorization, persistent saves and recovery in a real Colab session.
- A genuinely independent, adequately supported field test with reviewed labels
  and provenance. PlantDoc internet images do not automatically establish field
  representativeness or biological independence.
- Whether augmentation, AdamW, deeper tails or smoothing improve performance.
  They are candidates, not observed improvements on real data.
- Calibrated softmax probabilities. ECE/Brier/NLL and validation thresholds diagnose
  confidence; no temperature scaling or future-precision guarantee is claimed.

The code/documentation commit contains no real experiment results. Retain future
actual outputs in Drive, review them and publish results separately if appropriate.
Never use synthetic scores as disease results or tune against held-out outcomes.
