"""Shared RGB decoding and resizing. Never divide model inputs by 255."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_rgb(image):
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError("Upload a JPG, JPEG or PNG image.")
        with Image.open(path) as loaded:
            if loaded.format not in {"JPEG", "PNG"}:
                raise ValueError("Image content must be JPEG or PNG.")
            loaded.load()
            return np.array(ImageOps.exif_transpose(loaded).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError("Expected a nonempty uint8 RGB image.")
    return array.copy()


def resize_rgb(image, image_size=(224, 224)):
    import cv2
    rgb = read_rgb(image)
    # Same OpenCV bilinear resize is used in tf.data and interactive inference.
    return cv2.resize(rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def make_dataset(frame, dataset_root, config, training=False):
    import tensorflow as tf
    from .augmentation import build_augmentation
    paths = [str(Path(dataset_root) / relative) for relative in frame["path"]]
    labels = frame["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(frame), seed=config.seed, reshuffle_each_iteration=True)

    def decode(path, label):
        def load(raw):
            return resize_rgb(raw.numpy().decode("utf-8"), config.image_size)
        image = tf.py_function(load, [path], Tout=tf.float32)
        image.set_shape((*config.image_size, 3))
        return image, label

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(config.batch_size)
    if training:
        augmentation = build_augmentation(config.seed)
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
