"""ImageNet transfer-learning backbones with explicit, stable float32 heads."""
import tensorflow as tf

ARCHITECTURES = {"EfficientNetB0": tf.keras.applications.EfficientNetB0,
                 "MobileNetV3Large": tf.keras.applications.MobileNetV3Large}


@tf.keras.utils.register_keras_serializable(package="AgriVision")
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.matrix = self.add_weight(name="confusion", shape=(self.num_classes, self.num_classes), initializer="zeros", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        weights = None if sample_weight is None else tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
        self.matrix.assign_add(tf.math.confusion_matrix(labels, predicted, self.num_classes, weights=weights, dtype=tf.float32))

    def result(self):
        true_positive = tf.linalg.diag_part(self.matrix)
        return tf.reduce_mean(tf.math.divide_no_nan(2 * true_positive, tf.reduce_sum(self.matrix, 0) + tf.reduce_sum(self.matrix, 1)))

    def reset_state(self):
        self.matrix.assign(tf.zeros_like(self.matrix))

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}


@tf.keras.utils.register_keras_serializable(package="AgriVision")
class SparseSmoothedCrossentropy(tf.keras.losses.Loss):
    """Integer labels stay integer; optional smoothing is applied inside the loss."""
    def __init__(self, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        target = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), tf.shape(y_pred)[-1], dtype=tf.float32)
        return tf.keras.losses.categorical_crossentropy(target, tf.cast(y_pred, tf.float32), label_smoothing=self.label_smoothing)

    def get_config(self):
        return {**super().get_config(), "label_smoothing": self.label_smoothing}


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


def compile_model(model, learning_rate, config=None):
    adamw = config is not None and config.optimizer == "AdamW"
    optimizer = (tf.keras.optimizers.AdamW(learning_rate, weight_decay=config.weight_decay)
                 if adamw else tf.keras.optimizers.Adam(learning_rate))
    smoothing = config.label_smoothing if config else 0.0
    loss = SparseSmoothedCrossentropy(smoothing, dtype="float32") if smoothing else tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"), MacroF1(model.output_shape[-1])])


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
