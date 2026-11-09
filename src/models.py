"""ImageNet transfer-learning backbones with explicit, stable float32 heads."""
import tensorflow as tf

ARCHITECTURES = {"EfficientNetB0": tf.keras.applications.EfficientNetB0,
                 "MobileNetV3Large": tf.keras.applications.MobileNetV3Large}


def build_model(architecture, num_classes, config, weights="imagenet"):
    if architecture not in ARCHITECTURES or num_classes < 2:
        raise ValueError("Choose a supported architecture and at least two classes.")
    kwargs = {"include_top": False, "weights": weights, "input_shape": (*config.image_size, 3)}
    if architecture == "MobileNetV3Large":
        kwargs["include_preprocessing"] = True
    backbone = ARCHITECTURES[architecture](**kwargs)
    if not any(isinstance(layer, tf.keras.layers.Rescaling) for layer in backbone.layers):
        raise RuntimeError("Expected built-in preprocessing was not found; inspect the installed Keras version.")
    backbone.trainable = False
    inputs = tf.keras.Input((*config.image_size, 3), dtype="float32", name="rgb_pixels_0_255")
    # Inference mode keeps BatchNorm statistics fixed during both training stages.
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
    features = tf.keras.layers.Dropout(config.dropout, name="head_dropout")(features)
    logits = tf.keras.layers.Dense(num_classes, dtype="float32", name="logits")(features)
    probabilities = tf.keras.layers.Activation("softmax", dtype="float32", name="probabilities")(logits)
    return tf.keras.Model(inputs, probabilities, name=architecture)


def get_backbone(model):
    nested = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if len(nested) != 1:
        raise ValueError("Expected one nested Keras application backbone.")
    return nested[0]


def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])


def enable_fine_tuning(model, layers):
    backbone = get_backbone(model)
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= max(0, len(backbone.layers) - layers) and not isinstance(layer, tf.keras.layers.BatchNormalization)


def preprocessing_description(model):
    backbone = get_backbone(model)
    return {"input": "RGB float32 pixels [0,255]", "resize": "OpenCV INTER_LINEAR",
            "backbone": backbone.name, "built_in_rescaling": [layer.get_config() for layer in backbone.layers
                                                                 if isinstance(layer, tf.keras.layers.Rescaling)],
            "output_dtype": model.output.dtype}
