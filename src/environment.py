"""Inspect the installed TensorFlow stack before installing other dependencies."""
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from .utils import write_json


def inspect_environment():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        smi = "nvidia-smi unavailable"
    result = {"python": sys.version, "platform": platform.platform(), "tensorflow": tf.__version__,
              "keras": getattr(tf.keras, "__version__", "bundled"), "cuda_build": tf.sysconfig.get_build_info(),
              "gpus": [tf.config.experimental.get_device_details(g) for g in gpus], "nvidia_smi": smi}
    print(result)
    return result


def configure_runtime(config, output):
    import tensorflow as tf
    result = inspect_environment()
    tf.keras.utils.set_random_seed(config.seed)
    if config.deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print("Memory growth unchanged:", exc)
    capable = bool(gpus) and all(isinstance(d.get("compute_capability"), (tuple, list))
                                and tuple(d["compute_capability"]) >= (7, 0) for d in result["gpus"])
    policy = "mixed_float16" if config.request_mixed_precision and capable and tf.test.is_built_with_cuda() else "float32"
    tf.keras.mixed_precision.set_global_policy(policy)
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        tensor = tf.matmul(tf.ones((8, 8)), tf.ones((8, 8)))
    tensor.numpy()
    if gpus and "GPU" not in tensor.device.upper():
        raise RuntimeError("Requested GPU computation did not execute on a GPU.")
    result.update(precision_policy=policy, check_device=tensor.device, seed=config.seed,
                  recorded_at_utc=datetime.now(timezone.utc).isoformat())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "environment.json", result)
    (output / "requirements-resolved.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print("Precision:", policy, "| Check device:", tensor.device)
    return result
