# Sources and attribution

## Primary research and framework documentation

- Hughes and Salathé, *An open access repository of images on plant health to
  enable the development of mobile disease diagnostics*:
  https://arxiv.org/abs/1511.08060
- Upstream PlantVillage repository:
  https://github.com/spMohanty/PlantVillage-Dataset
- Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks*: https://arxiv.org/abs/1905.11946
- Howard et al., *Searching for MobileNetV3*:
  https://arxiv.org/abs/1905.02244
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization*: https://arxiv.org/abs/1610.02391
- Keras applications: https://keras.io/api/applications/efficientnet/ and
  https://keras.io/api/applications/mobilenet/
- Keras transfer learning: https://keras.io/guides/transfer_learning/
- Keras Grad-CAM example: https://keras.io/examples/vision/grad_cam/
- TensorFlow mixed precision: https://www.tensorflow.org/guide/mixed_precision
- TensorFlow deterministic operations:
  https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
- Gradio Blocks documentation: https://www.gradio.app/docs/gradio/blocks

The supplied functional reference is
https://huggingface.co/spaces/rsj56/crop-disease-detection. Its public page was
sleeping at inspection. No source code, training data, architecture, live behavior
or performance was assumed or copied. AgriVision branding and implementation are original.

## Disease information review

`disease_info.json` records source URLs and review dates for the standard 17
affected classes in the four-crop subset. Summaries are deliberately short and
nonchemical. Most sources are University of Minnesota Extension; spider mite
and yellow leaf curl records use University of California IPM; target spot uses
University of Florida IFAS publications. Every entry links directly to its source.
These are educational sources, not locally personalized treatment prescriptions.

The four healthy-class records intentionally have no disease advice and use the
unverified-information fallback. Unknown label spellings do the same. `verified`
means the displayed static summary was checked against the linked source, not
that a model prediction is verified. Re-review links and content before submission;
sources and regional recommendations may change.
