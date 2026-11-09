"""Mild training-only transformations preserve visible leaf symptoms."""
import tensorflow as tf


class RobustCameraVariation(tf.keras.layers.Layer):
    """Small experimental saturation/noise/blur changes, applied to training only."""
    def __init__(self, seed, **kwargs):
        super().__init__(dtype="float32", **kwargs)
        self.seed = seed

    def call(self, inputs, training=False):
        if not training:
            return inputs
        x = tf.cast(inputs, tf.float32)
        x = tf.image.random_saturation(x / 255., .95, 1.05, seed=self.seed) * 255.
        kernel = tf.constant([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
        kernel = tf.tile(kernel[:, :, None, None], [1, 1, 3, 1])
        blurred = tf.nn.depthwise_conv2d(tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT"), kernel, [1, 1, 1, 1], "VALID")
        mask = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], seed=self.seed + 1) < .15
        x = tf.where(mask, blurred, x)
        return tf.clip_by_value(x + tf.random.normal(tf.shape(x), stddev=1.5, seed=self.seed + 2), 0., 255.)


def build_augmentation(seed=42, profile="mild"):
    layers = tf.keras.layers
    if profile == "none":
        return tf.keras.Sequential([layers.Identity(dtype="float32")], name="no_augmentation")
    if profile not in {"mild", "robust"}:
        raise ValueError("Unknown augmentation profile.")
    transforms = [
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, fill_mode="reflect", seed=seed + 1),
        layers.RandomZoom(0.10, fill_mode="reflect", seed=seed + 2),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed + 3),
        layers.RandomBrightness(0.08, value_range=(0, 255), seed=seed + 4),
        layers.RandomContrast(0.08, seed=seed + 5),
    ]
    if profile == "robust":
        transforms.append(RobustCameraVariation(seed + 6))
    transforms.append(layers.ReLU(max_value=255., dtype="float32"))
    return tf.keras.Sequential(transforms, name=f"{profile}_training_augmentation")


def plot_augmentation(image, output, seed=42, profile="mild"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    augmenter = build_augmentation(seed, profile)
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
