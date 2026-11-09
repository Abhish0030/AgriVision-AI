"""Conservative Colab dependency setup and repository identity checks."""
from importlib import metadata
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
import json
from dataclasses import asdict
from .utils import read_json, write_json


def write_colab_constraints(path):
    from packaging.specifiers import SpecifierSet
    tensorflow_version = metadata.version("tensorflow")
    if tensorflow_version not in SpecifierSet(">=2.19,<2.22"):
        raise RuntimeError("This project targets TensorFlow >=2.19,<2.22. Use a compatible runtime; do not blindly upgrade it.")
    pinned = {name: metadata.version(name) for name in ("tensorflow", "keras", "numpy", "protobuf", "ml-dtypes", "h5py")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{name}=={version}" for name, version in pinned.items()) + "\n", encoding="utf-8")
    return pinned


def install_colab_dependencies(repository, constraints_path):
    import tensorflow as tf
    print("Before installation:", tf.__version__, tf.config.list_physical_devices("GPU"))
    pinned = write_colab_constraints(constraints_path)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(Path(repository) / "requirements-colab.txt"),
                           "--constraint", str(constraints_path), "--upgrade-strategy", "only-if-needed"])
    after = {name: metadata.version(name) for name in pinned}
    if after != pinned:
        raise RuntimeError("TensorFlow stack changed unexpectedly. Restart in a clean runtime and inspect pip output.")
    subprocess.check_call([sys.executable, "-m", "pip", "check"])
    return pinned


def validate_checkout(checkout, repository_url):
    """Reject a stale/wrong/edited checkout rather than recording misleading provenance."""
    checkout = Path(checkout).resolve()
    parsed = urlsplit(repository_url)
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use a plain HTTPS GitHub repository URL without embedded credentials.")
    remote = subprocess.check_output(["git", "-C", str(checkout), "remote", "get-url", "origin"], text=True).strip()
    def normalize(value):
        return value.removesuffix("/").removesuffix(".git").lower()
    if normalize(remote) != normalize(repository_url):
        raise ValueError("Existing checkout belongs to another remote. Choose another CHECKOUT directory.")
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip():
        raise ValueError("Checkout contains local changes. Preserve/review them before using it for an experiment.")
    import src
    if Path(src.__file__).resolve().parent != checkout / "src":
        raise RuntimeError("A stale src module is imported from another directory. Restart the Colab session.")
    return subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()


def record_experiment(paths, config, repository_url, commit):
    """Do not overwrite the original Colab commit/config provenance on reconnect."""
    path = paths.output / "runs" / "experiment.json"
    record = json.loads(json.dumps({"repository_url": repository_url, "commit": commit, "config": asdict(config)}))
    if path.exists() and read_json(path) != record:
        raise ValueError("Experiment commit/config differs from its recorded provenance. Use a new experiment directory.")
    if not path.exists():
        write_json(path, record)
    return record
