"""Two-stage training with full-model epoch checkpoints and persistent run state."""
from dataclasses import asdict
import time
import copy
from uuid import uuid4
import numpy as np
import pandas as pd
import tensorflow as tf
from .dataset import load_manifest
from .models import build_model, compile_model, enable_fine_tuning, preprocessing_description
from .preprocessing import make_dataset
from .utils import atomic_copy, atomic_save_model, confined_path, file_hash, read_json, training_source_hash, write_json
from .config import SUPPORTED_ARCHITECTURES


def restore_best_alias(folder, state):
    """State points to an immutable committed epoch; best.keras is a repairable alias."""
    source = confined_path(folder, state.get("best_checkpoint", "best.keras"))
    if not source.is_file():
        raise ValueError("Committed best checkpoint is missing; restore it from backup.")
    alias = folder / "best.keras"
    if source != alias and (not alias.is_file() or file_hash(alias) != file_hash(source)):
        atomic_copy(source, alias)
    return alias


class PersistentEpoch(tf.keras.callbacks.Callback):
    """Commit checkpoint before state; preserve optimizer and completed epoch."""
    def __init__(self, folder, state, stage):
        super().__init__()
        self.folder, self.state, self.stage = folder, state, stage

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items()}
        if not {"loss", "val_loss"}.issubset(logs) or not all(np.isfinite(value) for value in logs.values()):
            raise RuntimeError("Nonfinite or missing epoch metrics; preserving the last committed checkpoints.")
        seconds = time.perf_counter() - self.started
        previous = self.state.get("latest_checkpoint")
        old_best = self.state.get("best_checkpoint")
        filename = f"resume_{self.stage}_{epoch + 1:04d}_{uuid4().hex[:8]}.keras"
        atomic_save_model(self.model, self.folder / filename)
        next_state = copy.deepcopy(self.state)
        if logs["val_loss"] < next_state.get("best_val_loss", float("inf")):
            next_state.update(best_val_loss=logs["val_loss"], best_checkpoint=filename)
        next_state.update(stage=self.stage, next_epoch=epoch + 1, latest_checkpoint=filename,
                          training_seconds=self.state.get("training_seconds", 0) + seconds)
        next_state.setdefault("history", []).append({"stage": self.stage, "epoch": epoch + 1, **logs})
        write_json(self.folder / "state.json", next_state)
        self.state.clear()
        self.state.update(next_state)
        restore_best_alias(self.folder, self.state)
        retained = {filename, self.state.get("best_checkpoint")}
        for obsolete in {previous, old_best} - retained - {None, "best.keras"}:
            if obsolete.startswith("resume_"):
                confined_path(self.folder, obsolete).unlink(missing_ok=True)


def train_architecture(architecture, paths, config, resume=True):
    if architecture not in SUPPORTED_ARCHITECTURES or architecture not in config.architectures:
        raise ValueError("Select an architecture enabled in the experiment configuration.")
    frame, mapping, report = load_manifest(paths, verify_files=True, config=config)
    folder = paths.output / "models" / architecture
    folder.mkdir(parents=True, exist_ok=True)
    signature = {"architecture": architecture, "config": asdict(config), "class_indices": mapping,
                 "manifest_sha256": report["manifest_sha256"], "implementation_sha256": training_source_hash()}
    # Round-trip JSON normalizes tuples for a stable comparison after reconnection.
    import json
    signature = json.loads(json.dumps(signature))
    if (folder / "state.json").exists():
        if not resume:
            raise FileExistsError("Training artifacts already exist. Select a new output directory for a new experiment.")
        state = read_json(folder / "state.json")
        if state["signature"] != signature:
            raise ValueError("Run configuration, source or manifest changed; use a new output directory.")
        restore_best_alias(folder, state)
        if state.get("complete"):
            if not (folder / "best.keras").exists() or file_hash(folder / "best.keras") != state["best_model_sha256"]:
                raise ValueError("Completed checkpoint is missing or differs from its training record.")
            # Regenerate derived artifacts if interrupted just after the completion commit.
            pd.DataFrame(state["history"]).to_csv(folder / "history.csv", index=False)
            plot_history(state["history"], paths.output / "graphs" / f"{architecture}_training.png")
            return folder / "best.keras"
        model = tf.keras.models.load_model(confined_path(folder, state["latest_checkpoint"]))
    else:
        if (folder / "best.keras").exists():
            raise FileExistsError("Model exists without state. Use a clean experiment directory.")
        state = {"signature": signature, "stage": "head", "next_epoch": 0, "history": [], "training_seconds": 0}
        tf.keras.utils.set_random_seed(config.seed)
        model = build_model(architecture, len(mapping), config)
        compile_model(model, config.head_learning_rate)
        write_json(folder / "class_indices.json", mapping)
        config.save(folder / "config.json")
        write_json(folder / "preprocessing.json", preprocessing_description(model))
    train = frame[frame.split == "train"]
    validation = frame[frame.split == "validation"]
    train_ds = make_dataset(train, paths.dataset, config, training=True)
    val_ds = make_dataset(validation, paths.dataset, config)
    weights = None
    if config.class_weighting:
        counts = train.label.value_counts()
        weights = {int(label): len(train) / (len(mapping) * int(count)) for label, count in counts.items()}
    for stage, epochs in (("head", config.head_epochs), ("fine_tune", config.fine_tune_epochs)):
        if stage in state.get("finished_stages", []):
            continue
        if stage != state["stage"]:
            model = tf.keras.models.load_model(folder / "best.keras", compile=False)
            enable_fine_tuning(model, config.fine_tune_layers)
            compile_model(model, config.fine_tune_learning_rate)
            state.update(stage=stage, next_epoch=0)
        checkpoint = PersistentEpoch(folder, state, stage)
        callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                     tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=2, min_lr=1e-7),
                     checkpoint]
        model.fit(train_ds, validation_data=val_ds, initial_epoch=state["next_epoch"], epochs=epochs,
                  class_weight=weights, callbacks=callbacks, verbose=2, shuffle=False)
        if not (folder / "best.keras").exists():
            raise RuntimeError("No finite validation checkpoint was produced. Inspect training logs.")
        state.setdefault("finished_stages", []).append(stage)
        write_json(folder / "state.json", state)
    state.update(complete=True, best_model_sha256=file_hash(folder / "best.keras"))
    write_json(folder / "state.json", state)
    pd.DataFrame(state["history"]).to_csv(folder / "history.csv", index=False)
    plot_history(state["history"], paths.output / "graphs" / f"{architecture}_training.png")
    return folder / "best.keras"


def plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        axis.plot(range(1, len(frame) + 1), frame[metric], label="Training")
        axis.plot(range(1, len(frame) + 1), frame["val_" + metric], label="Validation")
        axis.set(xlabel="Completed epoch (both stages)", ylabel=metric)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
