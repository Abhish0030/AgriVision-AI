# Methodology

The classifier predicts a joint crop/disease class. Crop is decoded from that class,
not obtained from an independent crop detector.

## Integrity and splitting

Pillow validates JPEG/PNG, handles EXIF orientation and converts to RGB. File
SHA-256 records integrity. Decoded-pixel SHA-256 detects exact RGB copies across
encodings; one copy is retained, while conflicting labels stop preparation.

A 64-bit dHash compares adjacent grayscale thumbnail values. A Hamming-distance
BK-tree searches for near matches. Connected components keep transitive matches
together globally. Mixed-label components are quarantined. Within each class,
components are shuffled with a fixed seed and allocated according to remaining
image-count targets. Every class requires three independent groups. This is
class-stratified group splitting; exact proportions are not guaranteed.

dHash may over-group or miss similar images. Review the CSVs. The protocol is
not plant-identity/site-disjoint because those identities are unavailable.

## Inputs and models

The same Pillow RGB decoder and OpenCV bilinear resize serve every stage. Pixels
remain float32 [0,255]. EfficientNetB0 retains built-in preprocessing;
MobileNetV3Large sets include_preprocessing=True. A Rescaling layer is asserted
and recorded. No external ImageNet normalization or second `/255` is used.

Only training uses horizontal flip, mild rotation, zoom, translation and brightness/
contrast changes. Hue shifts are avoided to preserve symptom color. Both models
use ImageNet backbone → global average pooling → dropout → float32 logits/softmax.

Stage 1 freezes the backbone; Stage 2 unfreezes a configurable tail with a lower
Adam learning rate. BatchNorm remains frozen and the backbone is called with
training=False. Recompilation follows trainability changes. Sparse cross-entropy
uses integer labels. Inverse-frequency weights use training counts only.

Early stopping and ReduceLROnPlateau monitor validation loss. A custom persistent
callback performs the ModelCheckpoint role and also commits optimizer, epoch,
stage and history. Best validation checkpoint is independent of the resume
checkpoint. Test results never select checkpoints. Config/manifest changes require
a new experiment directory.
Training-affecting modules are fingerprinted with normalized text for portability
between Windows and Linux. Resume/inference reject changed source; comparison
also requires matching source fingerprints. Colab records the original repository
commit/config without overwriting them on reconnect.

Resumption restores completed epoch and optimizer state but resets patience and
augmentation ordering. Checkpoint data is written before its state pointer.
Only the previous committed resume checkpoint is deleted after the next state
commit when it is not also the best epoch. State points to an immutable saved
best epoch; best.keras is restored from that pointer if alias publication fails.
Nonfinite logs are rejected before checkpoint writes. Split/config/mapping
integrity is checked before training/evaluation. Interruptions may leave orphan
files; storage failures still require backups.
Concurrent writers to one experiment are unsupported.

## Evaluation and explanations

Both models use one frozen manifest and fixed label order. Save accuracy, macro
precision/recall/F1, weighted F1, per-class reports, raw confusion counts and
per-image predictions. Undefined per-class divisions return zero. Report parameter
counts, epoch time and synchronized batch-one latency with hardware/policy context.
A single seed is a baseline; repeated seeds would strengthen later research.

Grad-CAM separately discovers the last convolution in each nested backbone. An
auxiliary model exposes that activation and final backbone output; the shared
head reconstructs the logit score inside GradientTape. Spatial mean gradients
weight feature maps, followed by ReLU, safe normalization, resize and overlay.
Zero maps remain zero and are reported. Grad-CAM reveals sensitivity, not correctness.

The confidence threshold is an uncalibrated heuristic. No leaf-content detection,
universal disease coverage or field-generalization guarantee is claimed. Static
disease information is educational prevention advice, without chemical dosages.
