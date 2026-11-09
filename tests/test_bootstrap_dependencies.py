"""CPU-only installer tests; subprocess installation and Colab conflicts are mocked."""
import contextlib
import io
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from src.bootstrap import STACK_IMPORTS, PROJECT_IMPORTS, install_colab_dependencies, verify_project_dependencies


CONFLICTS = """ipython 7.34.0 requires jedi, which is not installed.
hf-gradio requires gradio-client<3.0,>=2.0, but you have gradio-client 1.14.0.
google-genai requires pydantic>=2.12.5, but you have pydantic 2.12.3.
python-fasthtml requires starlette>=1.0.1, but you have starlette 0.50.0.
google-adk requires starlette<2,>=1.3.1, but you have starlette 0.50.0.
"""


class BootstrapDependencyChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "requirements-colab.txt").write_text(
            "numpy>=1.26,<2.3\ngradio>=5.49,<6\npydantic>=2.10,<2.13\n", encoding="utf-8")
        self.versions = {"tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.2.6", "protobuf": "6.33.0",
                         "ml-dtypes": "0.5.3", "h5py": "3.15.1", "gradio": "5.50.0", "pydantic": "2.12.3",
                         "gradio-client": "1.14.0", "starlette": "0.50.0", "fastapi": "0.121.0"}
        self.dependencies = {"gradio": ["gradio-client==1.14.0", "fastapi>=0.115,<1", "starlette>=0.40,<1"],
                             "fastapi": ["pydantic>=2,<3", "starlette>=0.40,<1"],
                             "ipython": ["jedi>=0.16"], "google-genai": ["pydantic>=2.12.5"],
                             "tensorflow": ["keras>=3.12", "numpy>=1.26", "optional-missing; extra == 'gpu-fixture'"]}
        self.modules = {PROJECT_IMPORTS[name]: SimpleNamespace(__version__=version) for name, version in self.versions.items()
                        if name in PROJECT_IMPORTS}
        self.modules["tensorflow"].config = SimpleNamespace(list_physical_devices=lambda _: ["mock-working-gpu"])
        self.required = []
        def version(name):
            if name not in self.versions:
                raise metadata.PackageNotFoundError(name)
            return self.versions[name]
        def requires(name):
            self.required.append(name)
            return self.dependencies.get(name, [])
        self.enter = contextlib.ExitStack()
        self.addCleanup(self.enter.close)
        self.enter.enter_context(patch("src.bootstrap.metadata.version", side_effect=version))
        self.enter.enter_context(patch("src.bootstrap.metadata.requires", side_effect=requires))
        self.enter.enter_context(patch("src.bootstrap.import_module", side_effect=lambda name: self.modules[name]))
        self.enter.enter_context(patch.dict("sys.modules", {"tensorflow": self.modules["tensorflow"]}))

    def test_global_conflicts_warn_and_pins_and_gpu_are_unchanged(self):
        responses = [SimpleNamespace(returncode=0), SimpleNamespace(returncode=1, stdout=CONFLICTS, stderr="")]
        output = io.StringIO()
        with patch("src.bootstrap.subprocess.run", side_effect=responses) as run, contextlib.redirect_stdout(output):
            pinned = install_colab_dependencies(self.root, self.root / "constraints.txt")
        self.assertEqual(pinned, {name: self.versions[name] for name in STACK_IMPORTS})
        self.assertEqual(pinned["keras"], "3.13.2")
        self.assertIn("mock-working-gpu", output.getvalue())
        self.assertIn("nonfatal", output.getvalue())
        self.assertIn(CONFLICTS.strip(), output.getvalue())
        self.assertNotIn("google-genai", self.required)
        self.assertNotIn("ipython", self.required)
        install = run.call_args_list[0].args[0]
        self.assertIn("--constraint", install)
        self.assertIn("only-if-needed", install)
        self.assertNotIn("--upgrade", install)
        self.assertTrue(run.call_args_list[1].kwargs["capture_output"])
        self.assertIn("keras==3.13.2", (self.root / "constraints.txt").read_text())

    def test_actual_project_version_and_transitive_conflicts_are_fatal(self):
        for package, value in (("gradio", "6.0.0"), ("gradio-client", "2.0.0"), ("starlette", "1.3.1"), ("numpy", "2.3.0")):
            previous = self.versions[package]
            self.versions[package] = value
            with self.subTest(package=package), self.assertRaisesRegex(RuntimeError, package):
                verify_project_dependencies(self.root)
            self.versions[package] = previous

    def test_missing_direct_transitive_and_failed_import_are_fatal(self):
        for package in ("pydantic", "fastapi"):
            previous = self.versions.pop(package)
            with self.subTest(package=package), self.assertRaisesRegex(RuntimeError, "missing package " + package):
                verify_project_dependencies(self.root)
            self.versions[package] = previous
        with patch("src.bootstrap.import_module", side_effect=ImportError("simulated binary/import failure")):
            with self.assertRaisesRegex(RuntimeError, "import failed"):
                verify_project_dependencies(self.root)

    def test_protected_and_loaded_versions_must_match(self):
        with self.assertRaisesRegex(RuntimeError, "protected keras changed"):
            verify_project_dependencies(self.root, {"keras": "3.12.0"})
        self.modules["numpy"].__version__ = "2.1.0"
        with self.assertRaisesRegex(RuntimeError, "loaded numpy differs"):
            verify_project_dependencies(self.root)

    def test_active_extras_are_checked_and_unused_extras_are_ignored(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verify_project_dependencies(self.root)
        self.dependencies["gradio"].append("fastapi[test-fixture]")
        self.dependencies["fastapi"].append("required-extra-missing; extra == 'test-fixture'")
        with self.assertRaisesRegex(RuntimeError, "required-extra-missing"):
            verify_project_dependencies(self.root)

    def test_pip_failure_does_not_hide_project_errors_or_fail_a_valid_environment(self):
        responses = [SimpleNamespace(returncode=1), SimpleNamespace(returncode=0, stdout="No broken requirements found.", stderr="")]
        output = io.StringIO()
        with patch("src.bootstrap.subprocess.run", side_effect=responses), contextlib.redirect_stdout(output):
            install_colab_dependencies(self.root, self.root / "constraints.txt")
        self.assertIn("nonzero status", output.getvalue())
        self.versions["gradio-client"] = "2.1.0"
        with patch("src.bootstrap.subprocess.run", return_value=SimpleNamespace(returncode=0)), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "gradio-client"):
                install_colab_dependencies(self.root, self.root / "constraints.txt")


if __name__ == "__main__":
    unittest.main()
