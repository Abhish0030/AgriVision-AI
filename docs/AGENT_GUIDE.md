# Optional background visual interpretation

The TensorFlow/Keras classifiers remain authoritative. Their classes, softmax
scores, Grad-CAM and saved evaluation records are unchanged. The selected CNN
result is returned first. A separate Gradio event runs the other CNN without a
second Grad-CAM, then requests single-image visual interpretation. Only one
backbone is cached at a time; switching models may add loading latency. Both
completed project checkpoints are required for the detailed report. Missing
credentials, instructions, checkpoints or provider access preserves CNN-only use.
The original `/analyze` API retains its three inputs and ten outputs. The browser
chains the interpretation event; API clients calling `/analyze` receive CNN-only
results. The interpretation backend has no public named API endpoint.

## Private setup in VS Code

Install `requirements.txt`. Copy `.env.example` to `.env`, then enter your key
privately in that ignored file. Never paste credentials in chat, command-line
arguments, screenshots, notebook cells or GitHub. The default configurable model
is `nvidia/nemotron-3.5-super-vl:free`; availability and limits can change. No
automatic substitution to another or paid model is performed.

The private instruction file prepared in this workspace is
`secrets/agent_system_prompt.txt`. It is deliberately excluded from Git, so a
fresh clone needs the operator's separately transferred private instructions.
Keep a private backup. You may instead supply `AGENT_SYSTEM_PROMPT` as an
environment secret. Do not publish its contents. New operators must privately
author instructions consistent with the public response schema and grounding
rules in `src/agents/response_validator.py` before enabling this feature.

Modes: `AGENT_MODE=detailed` runs interpretation after each prediction;
`disagreement` calls the provider only when CNN classes differ; `off` disables it.
Restart/recreate the Gradio app after changing secrets or mode. Missing setup
emits a sanitized internal log and a plain unavailable message in the report.

## Google Colab

Follow the existing training notebook first. Add `OPENROUTER_API_KEY` and
`AGENT_SYSTEM_PROMPT` in Colab's **Secrets** sidebar and grant this notebook access
to those two secrets. Transfer the private instruction file's contents directly
into the latter secret outside notebook cells. The optional setup cell immediately
before the demo loads them without printing them. Set `ENABLE_PRIVATE_ANALYSIS`
to `True` only when ready to send an uploaded inference image. Leaving it `False`
explicitly selects CNN-only mode. Never save secret values as notebook output.
Model checkpoints continue to use the existing persistent Drive output directory.

Only the current resized RGB JPEG (maximum edge 640 pixels), both CNN classes,
crops, diseases, unchanged confidence/top-three scores and relevant verified
disease records are sent. EXIF and local paths are omitted. No training collection
is uploaded. OpenRouter and its serving provider necessarily receive the image,
context and system instructions to perform inference; private-from-Git/UI does
not mean hidden from the API provider. Review their current data policies before
using sensitive images or public share links. The application does not persist
request payloads, responses or prompts in experiment reports.

## Validation and failure handling

Requests use OpenRouter's HTTPS chat-completions endpoint with redirects disabled,
strict JSON Schema and parameter-compatible routing. There are at most two
exponential-backoff transport retries (bounded Retry-After) and one fresh retry
for invalid JSON. Authentication/configuration errors are not retried. A shared
50-second deadline covers the analysis attempts, with per-operation HTTP timeouts
up to 20 seconds; an in-flight socket operation may finish after the deadline
before control returns. Rejected responses and provider error bodies are never
displayed or logged. Oversized responses, extra/duplicate fields, changed CNN
fields, inconsistent agreement, unsupported class suggestions and unverified
recommendations/source URLs fail closed. The UI escapes all generated prose.
Long request waits use a separate event concurrency group from CNN prediction.
New predictions/uploads/resets invalidate stale reports; bounded backend state
is isolated by Gradio session, contains no secrets and holds no image bytes.

Visual descriptions and explanations remain unverified generated interpretations.
Agreement cannot establish diagnosis. Prevention/source entries must be exact
subsets of verified `disease_info.json` records. Existing disease information is
shown independently even when the agent fails. No agent confidence is added to
softmax scores, and the agent has no access to train or modify model weights.

## Checks and live diagnostics

```powershell
python -m unittest tests.test_agents -v
python -m src.agents.diagnostics
python -m src.agents.diagnostics --image "C:\private\one-leaf.jpg"
```

Diagnostics prints statuses only. Without a key/private instructions, authenticated
checks are explicitly skipped. The image check requires both actual project
checkpoints; use `AGRIVISION_OUTPUT` to select their existing Drive/local folder.
Tests mock success, malformed JSON, missing/invalid credentials, timeout, rate
limits, unavailable models, grounding, image resizing and stale UI completion.
Mocked requests do not verify real provider behavior. The existing full CPU suite
also tests real Gradio HTTP upload and classifier/Grad-CAM regressions.

## Preregistered paired evaluation

First prepare and review the independent external manifest using the existing
PlantDoc workflow. Select the primary model using validation only. Never choose
it after inspecting these agent-test outcomes. The command below freezes a
seeded subset, truth labels, hashes and decision rule in
`outputs/agent_evaluation/plan.json` before requests start. The existing output
experiment must contain both trained checkpoints and the reviewed external
manifest; the destination `agent_evaluation` subfolder must be new. A run does not
overwrite an existing plan. Use a separately prepared experiment to repeat it.

```powershell
python -m src.agents.evaluation --dataset "C:\datasets\PlantVillage" --external-root "C:\datasets\PlantDoc\test" --output "C:\experiments\agrivision" --primary EfficientNetB0 --limit 20
```

This explicitly authorized evaluation sends only sampled held-out images, one
inference request at a time; the default is 20 and the maximum 100. It does not
send training data or automatically run in Colab. Set `AGENT_MODE=detailed`.
In Colab the same module can be imported and called with the mounted `paths`,
external root and validation-selected primary. Require explicit approval of the
sampled images before execution under your own data-sharing rules.

Conditions A/B independently measure EfficientNetB0/MobileNetV3Large. Condition C
retains the preregistered primary CNN class on every sample, including API failure.
Consequently C accuracy, precision, recall and macro F1 equal that primary CNN
by construction. Reports keep conditions separate and measure CNN agreement,
agent disagreement among valid reports, API failures per attempted analysis,
valid-response coverage and wall-clock latencies. Macro averaging covers observed
truth classes, with predictions over all trained classes. Latencies exclude
loading and Grad-CAM for every condition; C includes both classifiers and API
encoding/network/validation. The cold first forward pass is included. A small
subset is not a broad field-performance estimate. Agent symptom correctness needs
separate expert annotations, which this experiment does not provide.

No disease accuracy or benefit from this integration has been established.

## Protocol references

- [OpenRouter public model catalog](https://openrouter.ai/api/v1/models)
- [Structured output contract](https://openrouter.ai/docs/guides/features/structured-outputs)
- [Multimodal image input](https://openrouter.ai/blog/tutorials/send-image-to-llm/)
- [Provider data collection policies](https://openrouter.ai/privacy)

Generic code can be public; credentials, private instructions and runtime payloads
remain excluded. Publication checks flag Git-eligible private instruction files
even if they were accidentally force-added. Do not force-add ignored files.
