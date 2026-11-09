"""Parse Python/notebook code and validate the notebook schema without executing it."""
import ast
import json
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
files = [*root.joinpath("src").rglob("*.py"), *root.joinpath("scripts").glob("*.py"),
         *root.joinpath("tests").glob("*.py"), root / "app.py"]
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
notebook = nbformat.read(root / "notebooks" / "AgriVision_AI_Complete.ipynb", as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
        assert cell.execution_count is None and not cell.outputs, "Clear notebook outputs before committing."
database = json.loads((root / "disease_info.json").read_text(encoding="utf-8"))
for entry in database.values():
    assert all(k in entry for k in ("crop", "disease", "description", "symptoms", "prevention", "source_url", "verified"))
    assert not entry["verified"] or entry["source_url"].startswith("https://")
print(f"Static checks passed: {len(files)} Python files, notebook schema/cells, {len(database)} disease records.")
