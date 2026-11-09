"""Load private instructions without embedding them in distributed source."""
import os
from pathlib import Path


def load_private_prompt(repository):
    value = os.environ.get("AGENT_SYSTEM_PROMPT", "").strip()
    if not value:
        path = Path(repository) / "secrets" / "agent_system_prompt.txt"
        if path.is_file() and path.stat().st_size <= 32000:
            value = path.read_text(encoding="utf-8").strip()
    return value if 20 <= len(value) <= 32000 else None
