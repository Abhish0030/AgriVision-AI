"""Gradient-weighted class activation maps for nested application backbones."""
import cv2
import numpy as np
import tensorflow as tf
from .models import get_backbone
from .preprocessing import read_rgb, resize_rgb


def convolution_layer_name(model):
    backbone = get_backbone(model)
    for layer in reversed(backbone.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) and len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No 2-D convolutional layer found in backbone.")


def gradcam(model, image, class_index=None, alpha=.4):
    if not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be in [0,1].")
    original = read_rgb(image)
    size = tuple(int(v) for v in model.input_shape[1:3])
    tensor = tf.convert_to_tensor(resize_rgb(original, size)[None], tf.float32)
    backbone = get_backbone(model)
    target = convolution_layer_name(model)
    probe = tf.keras.Model(backbone.inputs, [backbone.get_layer(target).output, backbone.output])
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        activation, features = probe(tensor, training=False)
        pooled = model.get_layer("global_pool")(features)
        pooled = model.get_layer("head_dropout")(pooled, training=False)
        logits = model.get_layer("logits")(pooled)
        selected = int(tf.argmax(logits[0])) if class_index is None else int(class_index)
        if not 0 <= selected < int(logits.shape[-1]):
            raise ValueError("Class index outside model output range.")
        score = logits[:, selected]  # Pre-softmax score reduces saturation artifacts.
    gradients = tape.gradient(score, activation)
    if gradients is None:
        raise RuntimeError("Grad-CAM graph is disconnected from the selected feature maps.")
    gradients = tf.cast(gradients, tf.float32)
    activation = tf.cast(activation, tf.float32)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(weights * activation, axis=-1))[0]
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap)).numpy()
    if not np.isfinite(heatmap).all():
        raise RuntimeError("Grad-CAM contains nonfinite values.")
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = np.uint8(np.clip((1 - alpha) * original + alpha * colored, 0, 255))
    return {"original": original, "heatmap": colored, "overlay": overlay,
            "values": heatmap, "layer": target, "class_index": selected,
            "zero_map": bool(np.max(heatmap) == 0)}
