# Project report and presentation preparation

These are outlines, not completed experiments. Fill results only after executing
the real dataset/training/evaluation workflow. No screenshot or metric is invented.
For the optimization study, use `validation_experiments.csv`, the frozen selection,
`final_comparison.csv`, per-class reports, actual Grad-CAM examples and external
support/overlap reports. See [TRAINING_GUIDE.md](TRAINING_GUIDE.md). Keep controlled
PlantVillage results, external point estimates and the unverified 90% research
target separate in the report and slides.

## Report outline

1. **Abstract:** goal, four crops, architectures, evaluation protocol. Add actual
   results and scope after experiments.
2. **Introduction:** leaf-image classification as an aid to agricultural inspection.
3. **Problem statement:** multiclass crop/disease recognition from a single image.
4. **Research motivation:** compare accuracy/efficiency and inspect model sensitivity.
5. **Literature review:** PlantVillage dataset, EfficientNet, MobileNetV3, Grad-CAM;
   distinguish published findings from your own experiments. See REFERENCES.md.
6. **Existing system:** the linked Hugging Face Space is a functional reference
   supplied in the brief; do not infer its architecture/data/accuracy. It was sleeping
   when inspected, so its live behavior was not verified.
7. **Proposed system:** modular TensorFlow pipeline, fair comparison, transparent UI.
8. **Dataset description:** actual source/license, counts, classes, dimensions,
   corruption, duplicates, quarantines and split distribution from report.json.
9. **Methodology:** audit, group splitting, shared preprocessing, augmentation,
   two-stage optimization and validation-based selection.
10. **System architecture:** dataset → audit/manifest → RGB pipeline → two backbones
    → checkpoint/evaluation → Grad-CAM/prediction → Gradio.
11. **Mathematical foundations:** equations and explanations in VIVA.md.
12. **Model training:** actual config, GPU/policy, epochs, histories, interruptions.
13. **Experimental results:** saved model comparison, per-class reports and confusion
    matrices; no placeholders presented as measurements.
14. **Grad-CAM explainability:** actual correct/incorrect examples for both backbones;
    show when positive maps are absent or focus on background.
15. **Discussion:** class confusion, imbalance, efficiency and observed shortcuts.
16. **Limitations:** controlled imagery, dHash heuristic, single-seed baseline,
    uncalibrated confidence, domain shift and unverified field performance.
17. **Conclusion:** answer research questions only as supported by actual results.
18. **Future scope:** compatible PlantDoc external evaluation, calibration,
    repeated-seed experiments and more diverse field images.
19. **References:** primary research, framework docs, dataset and disease sources.

## Twelve-slide outline

| Slide | Content | Evidence to insert after execution |
|---|---|---|
| 1 | Title, student and supervisor | Actual institution/project details |
| 2 | Introduction | Project scope and use case |
| 3 | Problem statement | Four-crop classification and reliability limits |
| 4 | Literature review | Primary paper citations, no borrowed accuracy claims |
| 5 | Proposed methodology | Pipeline diagram and group split protocol |
| 6 | Dataset | Real distribution and sample-image figures |
| 7 | Model architectures | Backbone/head diagrams and parameter counts |
| 8 | Training | Actual config, hardware and both training curves |
| 9 | Experimental results | Saved comparison and difficult-class confusion |
| 10 | Explainable AI | Grad-CAM examples for both models |
| 11 | Live demo | Own trained models, uncertainty warning, source links |
| 12 | Conclusion and future scope | Evidence-supported findings and limitations |

## Results interpretation worksheet

- Which model has greater macro F1? Is accuracy consistent with per-class performance?
- Which classes have low recall or are confused? Check support and background bias.
- Which model is faster on the same hardware? Disclose that preprocessing and
  explanation cost are excluded from the forward-pass timing.
- Does fine-tuning improve validation loss? Did either stage overfit?
- Do heatmaps differ between correct and incorrect predictions? Avoid causal claims.
- Were some near-match groups quarantined? How does that affect class coverage?
- Was external evaluation performed? If not, explicitly state that it was not.

## Demonstration checklist

Load completed models and actual saved metrics before evaluation day. Prepare
your own leaf examples, at least one low-confidence case when observed, an invalid
upload, both model selections and both Grad-CAM outputs. Do not manufacture a
prediction score to force a warning; you may adjust the displayed heuristic
threshold and disclose that change. Keep Drive checkpoints and a local output
backup. A Colab share link is temporary and requires a live runtime.
