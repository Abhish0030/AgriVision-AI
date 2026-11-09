# Training optimization plan and current evidence

The starting repository is commit 75ab0dd on main, synced with its existing
GitHub remote. Inspection found a working two-stage TensorFlow implementation,
shared raw-pixel preprocessing, grouped splits, resumable checkpoints, nested
Grad-CAM and a checkpoint-backed Gradio demo. There are **no real dataset images,
trained plant-disease checkpoints, class mappings or real evaluation results** in
this workspace. Cached ImageNet application weights are not disease classifiers.
No connected Colab browser/runtime is available to this agent.

## Research target

At least 90% accuracy on a genuinely independent real-world test set is a target,
not an achieved result or guarantee. Report unconditional accuracy, macro F1,
balanced accuracy, per-class recall and sample counts. Rejection coverage and
accuracy among accepted predictions are separate quantities, never replacements
for all-image test accuracy.

## Controlled sequence

1. Audit original-color PlantVillage for the four crops, then freeze one grouped,
   class-stratified 70/15/15 manifest. Reuse it for every trial.
2. Default six trials: each architecture with no augmentation, existing mild
   augmentation and a conservative robustness profile. All other settings and
   random seeds remain fixed. Noise/blur/saturation changes are experimental;
   they are not assumed beneficial.
3. Optional separately enabled trials change only optimizer (AdamW), tail/LR,
   or label smoothing relative to the mild baseline. Do not enable every option
   without a time budget and a reason. The saved plan lists exact configurations.
4. Save validation macro F1/balanced recall/per-class results and epoch histories.
   Choose the best checkpoint/trial using validation only. The full trial plan
   must finish before selection is frozen. No trial routine evaluates the test set.
5. Learn a warning/rejection threshold from validation predictions, with a minimum
   accepted sample count and a Wilson lower-bound screen. This is an observed
   validation operating point, not calibrated probability or a field guarantee.
6. Freeze selection and export one winner per architecture with model hashes and
   trial/version metadata. Evaluate their internal test sets after selection;
   prevent adding/tuning trials inside that frozen experiment afterward.
7. Evaluate compatible PlantDoc/field labels using an explicitly reviewed mapping.
   Keep all external images out of training/validation. Screen external overlaps
   against every PlantVillage split and deduplicate the external set. Review the
   screening reports: perceptual hashes cannot establish true biological independence.
8. Report full internal results and an internal comparison restricted to the same
   truth classes as the external set. Predictions still range over all trained
   classes; never renormalize to make external evaluation easier.
9. Inspect correct/incorrect Grad-CAM examples for both models and display actual
   versioned metrics in Gradio. Commit/push code and pending-experiment documentation;
   keep datasets, model weights and future measured results separate.

## Access and validity limits

Actual disease-model training must run in Google Colab with the user's data and
Drive authorization. The notebook will contain complete executable cells. Local
synthetic training is solely software validation and will not be reported as a
PlantVillage experiment.

PlantDoc's authors describe internet-scraped images. Its existence does not by
itself establish a representative, independently collected agricultural field
test. Review labels, provenance, sample support and duplicates before interpreting
an external point estimate. If samples/classes are insufficient or independence
cannot be established, state that the 90% real-world target is not verified.

Primary references: [PlantDoc paper](https://arxiv.org/abs/1911.10317),
[classification repository](https://github.com/pratikkayal/PlantDoc-Dataset),
[Keras AdamW](https://keras.io/api/optimizers/adamw/),
[Keras loss APIs](https://keras.io/api/losses/probabilistic_losses/).
