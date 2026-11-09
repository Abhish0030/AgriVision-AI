"""Small serialization and provenance helpers."""
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save_model(model, path):
    """A failed save leaves the previously published file untouched."""
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.keras")
    model.save(temporary)
    temporary.replace(path)


def atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def confined_path(root, relative):
    """Manifest/archive paths cannot escape a root on either Windows or POSIX."""
    from pathlib import PurePosixPath, PureWindowsPath
    if not isinstance(relative, str) or not relative:
        raise ValueError("Expected a nonempty relative path.")
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    if posix.is_absolute() or windows.drive or windows.root or ".." in posix.parts or ".." in windows.parts or "\\" in relative:
        raise ValueError(f"Path must be portable and relative: {relative}")
    root = Path(root).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes its root: {relative}")
    return resolved


def validate_mapping(mapping):
    if not isinstance(mapping, dict) or len(mapping) < 2:
        raise ValueError("Expected a class mapping with at least two classes.")
    if any(not isinstance(name, str) or "___" not in name or type(index) is not int for name, index in mapping.items()):
        raise ValueError("Invalid class names or integer class indices.")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class indices must be unique and contiguous from zero.")


def validate_probabilities(values, num_classes, num_samples=None):
    import numpy as np
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != num_classes or (num_samples is not None and len(values) != num_samples):
        raise ValueError("Probability tensor shape does not match samples/class mapping.")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Model output must be finite normalized softmax probabilities.")
    return values


def training_source_hash():
    """Portable fingerprint of source that affects the training experiment."""
    folder = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("config.py", "dataset.py", "preprocessing.py", "augmentation.py", "models.py", "training.py", "utils.py", "metrics.py", "validation.py", "experiments.py", "external.py"):
        digest.update(name.encode("utf-8"))
        digest.update((folder / name).read_text(encoding="utf-8").encode("utf-8"))
    return digest.hexdigest()
