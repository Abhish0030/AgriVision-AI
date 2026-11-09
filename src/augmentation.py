"""Mild training-only transformations preserve visible leaf symptoms."""
import tensorflow as tf


def build_augmentation(seed=42):
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, fill_mode="reflect", seed=seed + 1),
        layers.RandomZoom(0.10, fill_mode="reflect", seed=seed + 2),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed + 3),
        layers.RandomBrightness(0.08, value_range=(0, 255), seed=seed + 4),
        layers.RandomContrast(0.08, seed=seed + 5),
    ], name="training_augmentation")


def plot_augmentation(image, output, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    augmenter = build_augmentation(seed)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(np.asarray(image, dtype="uint8"))
    axes[0].set_title("Original")
    for i in range(1, 4):
        axes[i].imshow(np.clip(augmenter(tf.cast(image[None], tf.float32), training=True)[0].numpy(), 0, 255).astype("uint8"))
        axes[i].set_title("Training augmentation")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
