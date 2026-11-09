"""Single-image multimodal analysis. No training, model changes or raw logging."""
import base64
from io import BytesIO
import json
import logging
import os
import time
from PIL import Image
from dotenv import load_dotenv
from .agent_prompts import load_private_prompt
from .openrouter_client import DEFAULT_MODEL, OpenRouterClient, AgentUnavailable
from .response_validator import AgentReport, validate_response

LOGGER = logging.getLogger(__name__)
FALLBACK = "Additional visual analysis is unavailable. The original CNN prediction remains unchanged."


def image_data_url(image):
    # A new RGB image strips metadata and is resized before transmission.
    import numpy as np
    from ..preprocessing import read_rgb
    rgb = read_rgb(image)
    resized = Image.fromarray(np.asarray(rgb, dtype="uint8"))
    resized.thumbnail((640, 640))
    stream = BytesIO()
    resized.save(stream, format="JPEG", quality=80)
    if stream.tell() > 900000:
        raise ValueError("image_too_large")
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class NemotronAgent:
    def __init__(self, repository, client=None):
        try:
            load_dotenv(repository / ".env", override=False)
            self._prompt = load_private_prompt(repository)
        except (OSError, UnicodeError, ValueError):
            self._prompt = None
        self._key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.mode = os.environ.get("AGENT_MODE", "detailed").lower()
        self.enabled = bool(self._key and self._prompt and self.mode in {"detailed", "disagreement"})
        self._client = client or OpenRouterClient(self._key, os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
        if not self.enabled:
            reason = "missing API credential" if not self._key else "missing private instructions or disabled mode"
            LOGGER.info("Optional visual analysis disabled: %s; CNN-only operation remains available.", reason)

    def analyze(self, image, context):
        start = time.perf_counter()
        deadline = time.monotonic() + 50.
        if not self.enabled or (self.mode == "disagreement" and context["cnn_analysis"]["models_agree"]):
            return {"status": "disabled", "report": None, "latency_seconds": 0.}
        try:
            messages = [{"role": "system", "content": self._prompt}, {"role": "user", "content": [
                {"type": "text", "text": json.dumps(context, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": image_data_url(image)}}]}]
            # One fresh retry for malformed JSON; no rejected output is fed back.
            for attempt in range(2):
                raw = self._client.complete(messages, AgentReport.model_json_schema(), deadline=deadline)
                try:
                    report = validate_response(raw, context, (self._key, self._prompt))
                    return {"status": "ok", "report": report, "latency_seconds": time.perf_counter() - start}
                except (ValueError, TypeError):
                    if attempt:
                        raise AgentUnavailable("invalid_report") from None
        except Exception:
            # Provider exceptions, secrets and raw payloads must not cross this boundary.
            LOGGER.warning("Optional visual analysis unavailable; preserving CNN results.")
        return {"status": "failed", "report": None, "latency_seconds": time.perf_counter() - start}
