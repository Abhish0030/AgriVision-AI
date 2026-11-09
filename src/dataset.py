"""PlantVillage audit, duplicate grouping and deterministic class-stratified splits."""
import hashlib
import random
import numpy as np
import pandas as pd
from PIL import Image
from .preprocessing import ALLOWED_SUFFIXES, read_rgb
from .utils import confined_path, file_hash, read_json, validate_mapping, write_json


def split_settings(config):
    return {"seed": config.seed, "crops": list(config.crops), "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction, "test_fraction": config.test_fraction,
            "near_duplicate_distance": config.near_duplicate_distance}


class BKTree:
    """Hamming-distance index for practical 64-bit perceptual-hash grouping."""
    def __init__(self):
        self.root = None

    def query(self, value, radius):
        found, pending = [], [self.root] if self.root else []
        while pending:
            other, indices, children = pending.pop()
            distance = (value ^ other).bit_count()
            if distance <= radius:
                found.extend(indices)
            pending.extend(node for edge, node in children.items() if distance - radius <= edge <= distance + radius)
        return found

    def add(self, value, index):
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = (value, [index], {})
                return
            node = node[2][distance]


def difference_hash(rgb):
    gray = np.array(Image.fromarray(rgb).convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def class_from_path(path, root, crops):
    candidates = [part for part in path.relative_to(root).parts[:-1] if "___" in part]
    if len(candidates) != 1:
        return None
    label = candidates[0]
    crop = label.split("___")[0]
    return label if any(crop == c or crop.startswith(c + "_") for c in crops) else None


def prepare_dataset(paths, config):
    paths.initialize()
    destination = paths.output / "dataset"
    if (destination / "manifest.csv").exists():
        raise FileExistsError("A frozen split already exists. Reuse load_manifest(); choose a new output directory to re-audit.")
    if not paths.dataset.is_dir():
        raise FileNotFoundError(f"PlantVillage folder not found: {paths.dataset}")
    rows, invalid, unsupported = [], [], 0
    for path in sorted(paths.dataset.rglob("*")):
        if not path.is_file():
            continue
        label = class_from_path(path, paths.dataset, config.crops)
        if label is None:
            unsupported += 1
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        try:
            confined_path(paths.dataset, path.relative_to(paths.dataset).as_posix())
            rgb = read_rgb(path)
            canonical = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            rows.append(dict(path=path.relative_to(paths.dataset).as_posix(), class_name=label,
                             sha256=file_hash(path), pixel_hash=canonical, dhash=difference_hash(rgb),
                             height=rgb.shape[0], width=rgb.shape[1]))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            invalid.append({"path": path.relative_to(paths.dataset).as_posix(), "error": str(exc)})
    write_json(destination / "invalid_files.json", invalid)
    if not rows:
        raise ValueError("No supported images found. Use original PlantVillage color class folders named Crop___Disease.")
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "audit.csv", index=False)
    # Canonical pixel hashes also find exact decoded images with different encodings.
    conflicting = frame.groupby("pixel_hash")["class_name"].nunique()
    if (conflicting > 1).any():
        raise ValueError("Identical images have conflicting labels. Review audit.csv and resolve before splitting.")
    duplicates = frame[frame.duplicated("pixel_hash", keep="first")]
    duplicates.to_csv(destination / "exact_duplicates.csv", index=False)
    frame = frame.drop_duplicates("pixel_hash").reset_index(drop=True)
    present = frame.class_name.str.split("___").str[0]
    missing = [c for c in config.crops if not any(x == c or x.startswith(c + "_") for x in present)]
    if missing:
        raise ValueError(f"Required crops missing: {missing}. Supply all four original color subsets.")

    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    index, pairs = BKTree(), []
    for i, row in frame.iterrows():
        value = int(row.dhash)
        for j in index.query(value, config.near_duplicate_distance):
            parent[find(i)] = find(j)
            pairs.append((frame.iloc[j]["path"], row.path))
        index.add(value, i)
    frame["group"] = [find(i) for i in range(len(frame))]
    pd.DataFrame(pairs, columns=["path_a", "path_b"]).to_csv(destination / "near_duplicate_pairs.csv", index=False)
    # Near-match conflicts are quarantined for manual review, never assigned across splits.
    mixed = frame.groupby("group").class_name.nunique()
    quarantine = frame[frame.group.isin(mixed[mixed > 1].index)]
    quarantine.to_csv(destination / "quarantined.csv", index=False)
    frame = frame[~frame.group.isin(mixed[mixed > 1].index)].copy()
    classes = sorted(frame.class_name.unique())
    if set(classes) != set(row["class_name"] for row in rows):
        raise ValueError("Duplicate quarantine removed an entire class. Review quarantine before continuing.")
    mapping = {name: i for i, name in enumerate(classes)}
    validate_mapping(mapping)
    frame["label"] = frame.class_name.map(mapping)
    frame["split"] = ""
    rng = random.Random(config.seed)
    for name, subset in frame.groupby("class_name"):
        groups = list(subset.groupby("group").groups.items())
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(f"{name}: fewer than 3 independent groups; cannot create three leakage-safe splits.")
        # Allocate whole groups within each class. Exact image proportions cannot always be met.
        targets = np.array([config.train_fraction, config.validation_fraction, config.test_fraction]) * len(subset)
        counts = np.zeros(3)
        split_names = ("train", "validation", "test")
        for k, (_, indices) in enumerate(groups):
            empty = np.flatnonzero(counts == 0)
            remaining = len(groups) - k
            choice = int(empty[0]) if len(empty) == remaining else int(np.argmax(targets - counts))
            frame.loc[list(indices), "split"] = split_names[choice]
            counts[choice] += len(indices)
    frame = frame.sort_values(["label", "path"]).reset_index(drop=True)
    assert frame.groupby("group").split.nunique().max() == 1
    frame.to_csv(destination / "manifest.csv", index=False)
    write_json(destination / "class_indices.json", mapping)
    fingerprint = file_hash(destination / "manifest.csv")
    report = {"dataset": "PlantVillage (user supplied; verify provenance)", "raw_supported_images": len(rows),
              "retained_images": len(frame), "classes": len(classes), "corrupted": len(invalid),
              "exact_duplicates_removed": len(duplicates), "quarantined_near_matches": len(quarantine),
              "near_duplicate_pairs": len(pairs), "near_duplicate_hamming_threshold": config.near_duplicate_distance,
              "unsupported_files": unsupported, "seed": config.seed, "manifest_sha256": fingerprint,
              "class_indices_sha256": file_hash(destination / "class_indices.json"), "split_config": split_settings(config),
              "counts": pd.crosstab(frame.class_name, frame.split).to_dict(),
              "image_dimensions": frame.groupby(["width", "height"]).size().rename("count").reset_index().to_dict("records"),
              "imbalance_max_min": float(frame.class_name.value_counts().max() / frame.class_name.value_counts().min()),
              "limitation": "dHash is heuristic, not proof of no biological/near-duplicate leakage; review pairs and dataset provenance."}
    write_json(destination / "report.json", report)
    plot_dataset(frame, paths)
    return frame, mapping, report


def load_manifest(paths, verify_files=False, config=None):
    folder = paths.output / "dataset"
    frame = pd.read_csv(folder / "manifest.csv")
    mapping = read_json(folder / "class_indices.json")
    report = read_json(folder / "report.json")
    if file_hash(folder / "manifest.csv") != report["manifest_sha256"]:
        raise ValueError("Frozen manifest changed. Use a new experiment directory.")
    validate_mapping(mapping)
    if not report.get("class_indices_sha256") or file_hash(folder / "class_indices.json") != report["class_indices_sha256"]:
        raise ValueError("Class mapping changed or its integrity record is absent. Re-audit into a new output directory.")
    required = {"path", "class_name", "label", "split", "group", "sha256", "pixel_hash"}
    if not required.issubset(frame.columns) or frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Frozen manifest is empty or missing required values/columns.")
    if set(frame.class_name) != set(mapping) or not pd.api.types.is_integer_dtype(frame.label):
        raise ValueError("Manifest classes/labels differ from the class mapping.")
    if not (frame.class_name.map(mapping) == frame.label).all():
        raise ValueError("Manifest integer labels disagree with class_indices.json.")
    splits = {"train", "validation", "test"}
    if set(frame.split) != splits or any(set(group.split) != splits for _, group in frame.groupby("class_name")):
        raise ValueError("Every class must occur in all three supported splits.")
    if frame.path.duplicated().any() or frame.pixel_hash.duplicated().any() or frame.groupby("group").split.nunique().max() != 1:
        raise ValueError("Manifest contains duplicated paths/pixels or a group crossing splits.")
    if config is not None and report.get("split_config") != split_settings(config):
        raise ValueError("Frozen split settings differ from configuration. Use a new experiment directory.")
    for relative in frame.path:
        confined_path(paths.dataset, relative)
    if verify_files:
        for row in frame.itertuples():
            path = confined_path(paths.dataset, row.path)
            if not path.is_file() or file_hash(path) != row.sha256:
                raise ValueError(f"Missing or changed image: {path}")
    return frame, mapping, report


def plot_dataset(frame, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = pd.crosstab(frame.class_name, frame.split)
    ax = counts.plot.barh(stacked=True, figsize=(11, max(5, len(counts) * .3)))
    ax.set_xlabel("Images after duplicate audit")
    ax.figure.tight_layout()
    ax.figure.savefig(paths.output / "graphs" / "class_distribution.png", dpi=160)
    plt.close(ax.figure)
    samples = frame.groupby("class_name", sort=True).first().reset_index()
    fig, axes = plt.subplots((len(samples) + 3) // 4, 4, figsize=(14, 3 * ((len(samples) + 3) // 4)), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, samples.itertuples()):
        axis.imshow(read_rgb(paths.dataset / row.path))
        axis.set_title(row.class_name.replace("___", "\n"), fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.output / "graphs" / "samples.png", dpi=150)
    plt.close(fig)
