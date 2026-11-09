"""Bounded REST calls. Exceptions contain status codes, never response bodies."""
import time
import httpx

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"


class AgentUnavailable(Exception):
    """Safe internal reason, without provider messages or credentials."""


class OpenRouterClient:
    def __init__(self, key, model=DEFAULT_MODEL, transport=None, sleep=time.sleep,
                 timeout=20., retries=2):
        self._key = key
        self.model = model
        self._transport = transport
        self._sleep = sleep
        self.timeout = min(30., max(1., float(timeout)))
        self.retries = min(2, max(0, int(retries)))

    def complete(self, messages, schema=None, *, deadline=None):
        if not self._key:
            raise AgentUnavailable("missing_key")
        deadline = deadline if deadline is not None else time.monotonic() + 45.
        payload = {"model": self.model, "messages": messages, "max_tokens": 1800,
                   "temperature": 0, "provider": {"require_parameters": True}}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "leaf_assessment", "strict": True, "schema": schema}}
        # Redirects are disabled so bearer credentials cannot follow another origin.
        with httpx.Client(transport=self._transport, follow_redirects=False,
                          timeout=self.timeout, trust_env=False) as client:
            for attempt in range(self.retries + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AgentUnavailable("deadline")
                client.timeout = httpx.Timeout(min(self.timeout, remaining))
                try:
                    # Limit response size; do not retain or log raw provider errors.
                    with client.stream("POST", BASE_URL + "/chat/completions",
                                       headers={"Authorization": "Bearer " + self._key}, json=payload) as response:
                        status = response.status_code
                        if status == 200:
                            chunks, count = [], 0
                            for chunk in response.iter_bytes():
                                if time.monotonic() >= deadline:
                                    raise AgentUnavailable("deadline")
                                count += len(chunk)
                                if count > 128000:
                                    raise AgentUnavailable("response_too_large")
                                chunks.append(chunk)
                            import json
                            try:
                                result = json.loads(b"".join(chunks))
                                content = result["choices"][0]["message"]["content"]
                                if not isinstance(content, str) or len(content) > 24000:
                                    raise ValueError()
                                return content
                            except (ValueError, KeyError, IndexError, TypeError):
                                raise AgentUnavailable("invalid_envelope") from None
                        if status not in {408, 429, 500, 502, 503, 504}:
                            raise AgentUnavailable("authentication" if status in {401, 403} else "provider_unavailable")
                        delay = min(4., 2. ** attempt)
                        try:
                            delay = min(4., max(delay, float(response.headers.get("retry-after", 0))))
                        except ValueError:
                            pass
                except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
                    delay = min(4., 2. ** attempt)
                if attempt < self.retries:
                    self._sleep(min(delay, max(0., deadline - time.monotonic())))
        raise AgentUnavailable("request_failed")
