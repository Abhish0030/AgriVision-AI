"""Publication inventory; report secret-pattern locations without printing values."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
files = sorted(set(path.decode("utf-8") for path in result.split(b"\0") if path))
patterns = {
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{20,}\b|github_pat_[A-Za-z0-9_]{20,}",
    "AWS key": r"\bAKIA[A-Z0-9]{16}\b",
    "API secret": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b",
    "credential assignment": r'''(?i)(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*["'][^"'\s]{10,}["']''',
}
issues = []
for relative in files:
    path = ROOT / relative
    if not path.is_file():
        continue
    if path.stat().st_size > 5_000_000:
        issues.append((relative, "file over 5 MB; review before publication"))
    if path.suffix.lower() in {".keras", ".h5", ".ckpt", ".zip", ".pem", ".key", ".p12", ".pfx"}:
        issues.append((relative, "model/archive/key artifact eligible for publication"))
    if relative.startswith(("data/", "outputs/", ".venv/", "venv/", "node_modules/", "secrets/")) and path.name != ".gitkeep":
        issues.append((relative, "runtime/data/dependency file eligible for publication"))
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example") or path.name in {"kaggle.json", "token.json", "token.pickle", "secrets.json", "agent_system_prompt.txt"}:
        issues.append((relative, "credential filename eligible for publication"))
    content = path.read_text(encoding="utf-8", errors="replace")
    for kind, pattern in patterns.items():
        for match in re.finditer(pattern, content):
            issues.append((relative, f"{kind} pattern at line {content.count(chr(10), 0, match.start()) + 1}"))
    if path.suffix == ".md":
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if "://" not in target and not target.startswith("#") and not (path.parent / target.split("#")[0]).exists():
                issues.append((relative, f"broken local documentation link: {target}"))
for relative, issue in issues:
    print(relative, ":", issue)
if issues:
    raise SystemExit(1)
print(f"Publication inventory passed for {len(files)} Git-eligible files: no detected credential patterns, large artifacts or broken local links.")
print("Pattern checks cannot prove the absence of every secret; review git diff --cached before pushing.")
