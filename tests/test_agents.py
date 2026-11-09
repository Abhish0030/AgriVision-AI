"""Mocked provider reliability tests; these are not live API or accuracy evidence."""
import copy
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import httpx
from src.agents.openrouter_client import OpenRouterClient, AgentUnavailable
from src.agents.nemotron_agent import NemotronAgent, image_data_url
from src.agents.prediction_orchestrator import build_context, interpret, report_html, ARCHITECTURES
from src.agents.response_validator import validate_response
from src.agents.evaluation import summarize


def fixture():
    info = {"verified": True, "description": "Verified fixture description", "symptoms": [],
            "source_url": "https://example.org/verified", "prevention": ["Use clean tools."]}
    record = {"class_name": "Apple___healthy", "crop": "Apple", "disease": "healthy", "confidence_percent": 83.,
              "top_three": [{"class": "Apple___healthy", "probability": .83}, {"class": "Apple___scab", "probability": .17}],
              "information": info}
    records = {name: copy.deepcopy(record) for name in ARCHITECTURES}
    context = build_context(records, ARCHITECTURES[0])
    report = {"visual_analysis": {"visible_symptoms": ["Green leaf"], "image_quality": "Slight blur", "possible_disease_indicators": []},
              "cnn_analysis": context["cnn_analysis"], "agent_assessment": {"possible_match": "uncertain",
              "agreement_with_cnn": "uncertain", "explanation": "Visual evidence is insufficient."},
              "reliability": {"status": "uncertain", "warning": "Consult an agricultural expert."},
              "recommendations": ["Use clean tools."], "information_sources": ["https://example.org/verified"]}
    return records, context, report


class AgentChecks(unittest.TestCase):
    def test_strict_grounding_and_untrusted_output(self):
        records, context, report = fixture()
        self.assertEqual(validate_response(json.dumps(report), context), report)
        for section, field, value in [("cnn_analysis", "models_agree", False),
                                       ("cnn_analysis", "models_agree", "true"),
                                       ("agent_assessment", "possible_match", "Invented___disease"),
                                       ("agent_assessment", "explanation", "<script>alert(1)</script>")]:
            altered = copy.deepcopy(report)
            altered[section][field] = value
            with self.assertRaises(ValueError):
                validate_response(json.dumps(altered), context)
        for field, value in [("recommendations", ["Invented treatment"]), ("information_sources", ["https://evil.example/"])]:
            altered = {**report, field: value}
            with self.assertRaises(ValueError):
                validate_response(json.dumps(altered), context)
        with self.assertRaises(ValueError):
            validate_response(json.dumps(report), context, ("Green leaf",))
        with self.assertRaises(ValueError):
            validate_response('{"a":1,"a":2}', context)
        self.assertEqual(records[ARCHITECTURES[0]]["confidence_percent"], 83.)

    def test_http_requests_and_rate_limit_retry(self):
        received, delays = [], []
        def handler(request):
            received.append(json.loads(request.content))
            if len(received) == 1:
                return httpx.Response(429, headers={"retry-after": "1"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        client = OpenRouterClient("dummy-key", transport=httpx.MockTransport(handler), sleep=delays.append)
        self.assertEqual(client.complete([{"role": "user", "content": "test"}]), "OK")
        self.assertEqual(delays, [1.])
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]["model"], "nvidia/nemotron-3.5-super-vl:free")

    def test_auth_missing_key_unavailable_invalid_envelope_and_timeout(self):
        for status in (401, 403, 404, 400, 503):
            calls = []
            def handler(request):
                calls.append(1)
                return httpx.Response(status, text="Sensitive provider content")
            client = OpenRouterClient("dummy-key", transport=httpx.MockTransport(handler), sleep=lambda _: None)
            with self.assertRaises(AgentUnavailable) as exc:
                client.complete([])
            self.assertNotIn("Sensitive", str(exc.exception))
            self.assertEqual(len(calls), 3 if status == 503 else 1)
        with self.assertRaises(AgentUnavailable):
            OpenRouterClient("").complete([])
        def timeout(request):
            raise httpx.ReadTimeout("private transport detail")
        with self.assertRaises(AgentUnavailable):
            OpenRouterClient("dummy-key", transport=httpx.MockTransport(timeout), sleep=lambda _: None).complete([])
        client = OpenRouterClient("dummy-key", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})))
        with self.assertRaises(AgentUnavailable):
            client.complete([])

    def test_missing_private_setup_fallback_and_no_secondary_call(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            agent = NemotronAgent(Path(folder))
            self.assertFalse(agent.enabled)
            loader = Mock()
            result = interpret(None, ARCHITECTURES[0], fixture()[0][ARCHITECTURES[0]], loader, agent)
            self.assertEqual(result["status"], "disabled")
            loader.assert_not_called()
            self.assertIn("unchanged", report_html(result))

    def test_invalid_private_file_and_repeated_invalid_report_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            (root / "secrets").mkdir()
            (root / "secrets" / "agent_system_prompt.txt").write_bytes(b"\xff\xfe")
            self.assertFalse(NemotronAgent(root).enabled)
        records, context, _ = fixture()
        client = Mock(complete=Mock(return_value="not JSON"))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "dummy-key", "AGENT_SYSTEM_PROMPT": "Private test fixture instructions.", "AGENT_MODE": "detailed"}, clear=True):
            outcome = NemotronAgent(Path(folder), client=client).analyze(None, context)
            self.assertEqual(outcome["status"], "failed")
            # Image validation fails before the provider is called.
            client.complete.assert_not_called()
            import numpy as np
            outcome = NemotronAgent(Path(folder), client=client).analyze(np.zeros((8, 8, 3), dtype=np.uint8), context)
            self.assertEqual(outcome["status"], "failed")
            self.assertEqual(client.complete.call_count, 2)
            agent = NemotronAgent(Path(folder), client=client)
            result = interpret(None, ARCHITECTURES[0], records[ARCHITECTURES[0]], Mock(side_effect=FileNotFoundError()), agent)
            self.assertEqual(result["status"], "failed")

    def test_image_json_retry_and_immutable_cnn(self):
        import base64
        from io import BytesIO
        import numpy as np
        from PIL import Image
        image = np.zeros((1200, 800, 3), dtype=np.uint8)
        url = image_data_url(image)
        with Image.open(BytesIO(base64.b64decode(url.split(",")[1]))) as encoded:
            self.assertLessEqual(max(encoded.size), 640)
            self.assertFalse(encoded.getexif())
        records, context, report = fixture()
        before = copy.deepcopy(records)
        captured = []
        def handler(request):
            captured.append(json.loads(request.content))
            content = "invalid JSON" if len(captured) == 1 else json.dumps(report)
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        client = OpenRouterClient("dummy-key", transport=httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "dummy-key", "AGENT_SYSTEM_PROMPT": "Private test fixture instructions.", "AGENT_MODE": "detailed"}, clear=True):
            agent = NemotronAgent(Path(folder), client=client)
            outcome = interpret(image, ARCHITECTURES[0], records[ARCHITECTURES[0]],
                                lambda _: Mock(predict_plant_disease=Mock(return_value=records[ARCHITECTURES[1]])), agent)
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["response_format"]["type"], "json_schema")
        self.assertTrue(captured[0]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(records, before)
        self.assertNotIn("Private test", report_html(outcome))
        self.assertNotIn("dummy-key", report_html(outcome))

    def test_fixed_evaluation_rule_and_failure_denominators(self):
        report = fixture()[2]
        predictions = {ARCHITECTURES[0]: ["a", "a"], ARCHITECTURES[1]: ["a", "b"]}
        outcomes = [{"status": "ok", "report": report, "latency_seconds": .3},
                    {"status": "failed", "report": None, "latency_seconds": .4}]
        result = summarize(["a", "b"], predictions, ARCHITECTURES[0], outcomes,
                           {name: [.1, .2] for name in ARCHITECTURES}, [.5, .7])
        self.assertEqual(result["conditions"]["CNN_with_analysis"], result["conditions"][ARCHITECTURES[0]])
        self.assertEqual(result["cnn_agreement_rate"], .5)
        self.assertEqual(result["api_failure_rate_per_analysis"], .5)

    def test_gradio_initial_result_precedes_network_and_stale_result_is_discarded(self):
        import threading
        from types import SimpleNamespace
        import numpy as np
        from src.app import create_app
        from src.config import ProjectConfig
        records, _, report = fixture()
        result = {**records[ARCHITECTURES[0]], "gradcam": {
            "layer": "test", "zero_map": False, "original": np.zeros((8, 8, 3)),
            "heatmap": np.zeros((8, 8, 3)), "overlay": np.zeros((8, 8, 3))},
            "health_status": "Predicted healthy", "uncertainty_warning": "",
            "model_version": "test", "rejection_threshold": .7, "threshold_source": "configured heuristic"}
        entered, release = threading.Event(), threading.Event()
        def wait_for_network(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test network wait timed out")
            return {"status": "ok", "report": report, "latency_seconds": .1}
        predictor = SimpleNamespace(classes=["Apple___healthy"], predict_plant_disease=lambda *a, **k: result,
                                    evaluation_metrics=lambda: "Evaluation not yet available")
        agent = SimpleNamespace(enabled=True, analyze=wait_for_network)
        with tempfile.TemporaryDirectory() as folder, patch("src.app.NemotronAgent", return_value=agent), \
                patch("src.app.PlantDiseasePredictor", return_value=predictor):
            paths = SimpleNamespace(repository=Path(folder), output=Path(folder))
            demo = create_app(paths, ProjectConfig())
            functions = {entry.fn.__name__: entry.fn for entry in demo.fns.values() if entry.fn}
            request = SimpleNamespace(session_hash="session-a")
            initial = functions["analyze"]("image.png", ARCHITECTURES[0], 0, request)
            self.assertEqual(len(initial), 10)
            self.assertFalse(entered.is_set())
            results = []
            worker = threading.Thread(target=lambda: results.append(functions["background_analysis"](
                "image.png", ARCHITECTURES[0], 0, request)))
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                # While the remote response is pending, a new initial prediction completes.
                newer = functions["analyze"]("new.png", ARCHITECTURES[0], 0, request)
                self.assertEqual(len(newer), 10)
            finally:
                release.set()
                worker.join(5)
                demo.close()
            self.assertEqual(results, [{"__type__": "update"}])


if __name__ == "__main__":
    unittest.main()
