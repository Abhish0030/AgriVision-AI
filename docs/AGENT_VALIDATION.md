# Background agent integration validation

Local validation on 2026-11-10, Windows, Python 3.13.7, TensorFlow 2.21.0,
Keras 3.15.1, Gradio 5.50.0. No GPU/Colab execution was performed for this change.

## Executed checks

- Final `python -m unittest discover -s tests -v`: **23 tests passed** in
  **69.595 seconds**. Includes 8 agent checks and the existing 15 CPU regression
  tests. Temporary synthetic fixtures exercise software, not crop accuracy.
- Existing real TensorFlow backbone/Grad-CAM and saved-model tests passed, as did
  the localhost Gradio HTTP image-upload regression with the unchanged ten-output
  `/analyze` interface. Background event behavior was tested directly with a
  blocked mocked provider: initial results return first, a newer CNN prediction
  proceeds during the network wait, and stale analysis is discarded.
- Mocked provider tests covered text/image request construction, strict JSON,
  immutable CNN outputs, malformed output retries, missing/invalid credentials,
  unreadable private instructions, missing secondary checkpoint fallback, timeout,
  rate-limit retry, unavailable provider/model responses, source/advice grounding,
  metadata-free image resizing, and the fixed evaluation rule/failure denominator.
- `ruff check src scripts tests`, recursive Python/notebook static checks and
  `pip check` passed. The notebook remains unexecuted with no saved outputs.
- The repository publication scanner passed. Git ignore checks confirm exclusion
  of `.env`, `secrets/agent_system_prompt.txt`, `secrets.json` and runtime outputs.
  Credentials and private instructions are absent from the distributed code.
- The public OpenRouter catalog was reached successfully and listed the requested
  model with image input and structured-output parameters. This is catalog
  evidence, not evidence of successful authenticated inference. See the
  [public catalog](https://openrouter.ai/api/v1/models).

## Not executed / remaining verification

No API key was configured. Authenticated text, multimodal image and live validated
JSON responses were **not tested**. Invalid-key/rate-limit/network cases were
simulated, not live provider failures. There were no actual trained disease
checkpoints or independent disease images available for this integration's paired
evaluation. No disease accuracy, F1, field-performance increase, symptom accuracy
or API latency/failure statistics have been established.

The installed Gradio/Keras versions emit deprecation warnings, Windows asyncio
resource warnings and TensorFlow retracing warnings in the test suite; tests
completed successfully. Dependency bounds retain Gradio 5 compatibility. A future
Gradio 6 migration requires a separate compatibility review.

To finish live validation, configure the key and private instructions, complete
both existing CNN training workflows, run the explicit diagnostics on one leaf
image, then run the preregistered independent paired experiment. Follow
[AGENT_GUIDE.md](AGENT_GUIDE.md). Keep credentials, private instructions, runtime
payloads and real data outside Git. Condition C retains the primary CNN class;
its classification metrics equal that primary CNN by construction.
