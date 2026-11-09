"""Conservative Colab dependency setup and repository identity checks."""
from importlib import metadata
from importlib import import_module
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
import json
from dataclasses import asdict


STACK_IMPORTS = {"tensorflow": "tensorflow", "keras": "keras", "numpy": "numpy",
                 "protobuf": "google.protobuf", "ml-dtypes": "ml_dtypes", "h5py": "h5py"}
PROJECT_IMPORTS = {**STACK_IMPORTS, "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn",
                   "scikit-learn": "sklearn", "opencv-python-headless": "cv2", "pillow": "PIL",
                   "gradio": "gradio", "packaging": "packaging", "httpx": "httpx",
                   "python-dotenv": "dotenv", "pydantic": "pydantic"}


def verify_project_dependencies(repository, expected_versions=None):
    """Validate project imports and their active dependency graph, not the global environment.

    Returns actual versions for direct requirements and the protected TensorFlow
    stack. Optional extras are followed only when requested by an active dependency.
    Nothing is installed or upgraded here.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
    requirements = [Requirement("tensorflow>=2.19,<2.22")]
    for line in (Path(repository) / "requirements-colab.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirements.append(Requirement(line))
    requirements.extend(Requirement(name) for name in STACK_IMPORTS)
    expected_versions = {canonicalize_name(name): version for name, version in (expected_versions or {}).items()}
    versions, issues, visited = {}, [], set()
    pending = [(requirement, "project", frozenset()) for requirement in requirements]
    while pending:
        requirement, parent, parent_extras = pending.pop()
        if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in {"", *parent_extras}):
            continue
        name = canonicalize_name(requirement.name)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            issues.append(f"{parent} requires missing package {name}")
            continue
        if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
            issues.append(f"{parent} requires {name}{requirement.specifier}; installed {installed}")
        if name in expected_versions and installed != expected_versions[name]:
            issues.append(f"protected {name} changed: expected {expected_versions[name]}, installed {installed}")
        if parent == "project":
            versions[name] = installed
        signature = (name, frozenset(requirement.extras))
        if signature in visited:
            continue
        visited.add(signature)
        for dependency in metadata.requires(name) or []:
            pending.append((Requirement(dependency), name, signature[1]))
    for name, version in sorted(versions.items()):
        module_name = PROJECT_IMPORTS.get(name)
        if module_name is None:
            issues.append(f"no import verification configured for project package {name}")
            continue
        try:
            module = import_module(module_name)
            # Detect a stale already-imported binary stack after a pip replacement.
            if name in STACK_IMPORTS and Version(module.__version__) != Version(version):
                issues.append(f"loaded {name} differs from installed metadata; restart the runtime")
        except Exception as exc:
            issues.append(f"{name} import failed ({type(exc).__name__})")
    if issues:
        raise RuntimeError("Project dependency verification failed:\n- " + "\n- ".join(sorted(set(issues))))
    print("Verified project package versions:", json.dumps(versions, sort_keys=True))
    return versions


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
    installed = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(Path(repository) / "requirements-colab.txt"),
                                "--constraint", str(constraints_path), "--upgrade-strategy", "only-if-needed"], check=False)
    # Project imports/requirements are authoritative. Global Colab conflicts can
    # involve packages outside this graph; never repair them blindly.
    verify_project_dependencies(repository, expected_versions=pinned)
    if installed.returncode:
        print("Warning: pip install returned a nonzero status, but all project dependencies verified successfully.")
    print("Preserved TensorFlow stack:", json.dumps(pinned, sort_keys=True))
    print("After installation: GPU devices:", tf.config.list_physical_devices("GPU"))
    try:
        checked = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False)
        if checked.returncode:
            print("Warning: global pip check reports conflicts outside the verified project dependency graph. "
                  "These are nonfatal; no unrelated packages were upgraded or removed.")
            print((checked.stdout + checked.stderr).strip())
        else:
            print("Global pip check:", checked.stdout.strip() or "passed")
    except OSError as exc:
        print(f"Warning: global pip check could not run ({type(exc).__name__}); project verification passed.")
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
    from .utils import read_json, write_json
    path = paths.output / "runs" / "experiment.json"
    record = json.loads(json.dumps({"repository_url": repository_url, "commit": commit, "config": asdict(config)}))
    if path.exists() and read_json(path) != record:
        raise ValueError("Experiment commit/config differs from its recorded provenance. Use a new experiment directory.")
    if not path.exists():
        write_json(path, record)
    return record
