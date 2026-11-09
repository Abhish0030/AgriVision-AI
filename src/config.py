"""Portable experiment settings; paths never depend on the current directory."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

SUPPORTED_ARCHITECTURES = ("EfficientNetB0", "MobileNetV3Large")
SUPPORTED_CROPS = ("Tomato", "Potato", "Corn", "Apple")


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    crops: tuple[str, ...] = ("Tomato", "Potato", "Corn", "Apple")
    architectures: tuple[str, ...] = ("EfficientNetB0", "MobileNetV3Large")
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    head_epochs: int = 5
    fine_tune_epochs: int = 10
    head_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    dropout: float = 0.3
    fine_tune_layers: int = 30
    patience: int = 4
    near_duplicate_distance: int = 4
    class_weighting: bool = True
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    confidence_threshold: float = 0.70
    request_mixed_precision: bool = True
    deterministic_ops: bool = True
    input_pixel_range: tuple[float, float] = (0.0, 255.0)
    built_in_preprocessing: bool = True
    dataset_name: str = "PlantVillage"

    def __post_init__(self):
        integer_fields = ("seed", "batch_size", "head_epochs", "fine_tune_epochs", "fine_tune_layers", "patience", "near_duplicate_distance")
        for key in integer_fields:
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key in {"seed", "near_duplicate_distance"} else 1):
                raise ValueError(f"{key} must be an integer in its valid nonnegative/positive range.")
        if not 0 <= self.seed < 2**31:
            raise ValueError("seed must be in [0, 2**31).")
        for key in ("head_learning_rate", "fine_tune_learning_rate", "dropout", "train_fraction", "validation_fraction", "test_fraction", "confidence_threshold"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number.")
        if min(self.head_learning_rate, self.fine_tune_learning_rate) <= 0 or self.fine_tune_learning_rate >= self.head_learning_rate:
            raise ValueError("Learning rates must be positive; fine-tuning must use a lower rate.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1).")
        if not isinstance(self.image_size, (tuple, list)) or len(self.image_size) != 2 or any(type(size) is not int or size < 32 for size in self.image_size):
            raise ValueError("image_size must contain two integers >=32 for both application backbones.")
        for key, choices in (("crops", SUPPORTED_CROPS), ("architectures", SUPPORTED_ARCHITECTURES)):
            values = getattr(self, key)
            if not values or len(set(values)) != len(values) or any(value not in choices for value in values):
                raise ValueError(f"{key} must contain unique supported names: {choices}.")
        for key in ("request_mixed_precision", "deterministic_ops", "class_weighting", "built_in_preprocessing"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean.")
        if self.dataset_name != "PlantVillage":
            raise ValueError("This primary pipeline requires explicitly supplied PlantVillage data.")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(x <= 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-8:
            raise ValueError("Split fractions must be positive and sum to one.")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be in (0, 1].")
        if self.batch_size < 1 or min(self.image_size) < 1:
            raise ValueError("Batch size and image dimensions must be positive.")
        if min(self.head_epochs, self.fine_tune_epochs, self.fine_tune_layers, self.patience) < 1:
            raise ValueError("Epochs, fine-tune layers and patience must be positive.")
        if not 0 <= self.near_duplicate_distance <= 8:
            raise ValueError("Near-duplicate Hamming distance must be between 0 and 8.")
        if not self.built_in_preprocessing or self.input_pixel_range != (0.0, 255.0):
            raise ValueError("This implementation requires built-in preprocessing and raw [0,255] pixels.")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("crops", "architectures", "image_size", "input_pixel_range"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class ProjectPaths:
    repository: Path
    dataset: Path
    output: Path

    @classmethod
    def create(cls, dataset=None, output=None):
        import os
        root = Path(__file__).resolve().parents[1]
        return cls(root, Path(dataset or os.getenv("AGRIVISION_DATA", root / "data" / "PlantVillage")).expanduser().resolve(),
                   Path(output or os.getenv("AGRIVISION_OUTPUT", root / "outputs")).expanduser().resolve())

    def initialize(self):
        for name in ("models", "graphs", "confusion_matrices", "predictions", "evaluation", "dataset", "runs"):
            (self.output / name).mkdir(parents=True, exist_ok=True)
        return self
