"""Green Gradio Blocks demo; model metrics always come from saved evaluation."""
import gradio as gr
from collections import OrderedDict
import threading
import time
from .agents.nemotron_agent import NemotronAgent, FALLBACK
from .agents.prediction_orchestrator import interpret, report_html
from .disease_information import information_markdown
from .prediction import PlantDiseasePredictor
from .utils import read_json


LIMITATIONS = ("Research demonstration for supported crop classes. PlantVillage has controlled backgrounds; "
               "field performance may be worse. Softmax confidence is uncalibrated. This system does not detect "
               "all diseases, verify leaf content, or replace agricultural diagnosis. Grad-CAM indicates "
               "class-sensitive regions and does not establish a correct diagnosis.")


def create_app(paths, config):
    cache = {}
    agent = NemotronAgent(paths.repository)
    inference_lock = threading.RLock()
    jobs_lock = threading.Lock()
    jobs = OrderedDict()
    selection_path = paths.output / "evaluation" / "selection.json"
    selected_default = read_json(selection_path)["overall_best_architecture"] if selection_path.exists() else config.architectures[0]
    if not selection_path.exists():
        available = [name for name in config.architectures if (paths.output / "models" / name / "best.keras").exists()]
        if available:
            selected_default = available[0]

    def load(name):
        if name not in cache:
            # Keep at most one backbone in memory on small runtimes.
            cache.clear()
            cache[name] = PlantDiseasePredictor(paths, name)
        return cache[name]

    def invalidate(request: gr.Request):
        with jobs_lock:
            jobs.pop(request.session_hash, None)
        return ""

    def analyze(image, architecture, threshold, request: gr.Request = None):
        if not image:
            raise gr.Error("Upload a JPG, JPEG or PNG leaf image first.")
        try:
            with inference_lock:
                predictor = load(architecture)
                result = predictor.predict_plant_disease(image, None if threshold == 0 else threshold)
            if request is not None and request.session_hash:
                # Store only minimal classifier data in bounded backend memory.
                reduced = {key: value for key, value in result.items() if key != "gradcam"}
                with jobs_lock:
                    jobs[request.session_hash] = (time.monotonic(), str(image), architecture, threshold, reduced)
                    jobs.move_to_end(request.session_hash)
                    while len(jobs) > 64:
                        jobs.popitem(last=False)
        except (OSError, ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        cam = result["gradcam"]
        summary = (f"**Crop:** {result['crop']}  \n**Disease/class:** {result['disease']}  \n"
                   f"**Confidence:** {result['confidence_percent']:.2f}%  \n**Health status:** {result['health_status']}  \n"
                   f"**Model version:** `{result['model_version']}`  \n**Warning threshold:** {result['rejection_threshold']:.3f} ({result['threshold_source']})")
        warning = result["uncertainty_warning"] or "No low-confidence warning triggered. Confidence is not a reliability guarantee."
        explanation = f"Grad-CAM layer: `{cam['layer']}`. Warm colors indicate positive contribution to the selected class score."
        if cam["zero_map"]:
            explanation += " This image has a zero positive-activation map; the overlay is not informative."
        metrics = predictor.evaluation_metrics()
        if isinstance(metrics, str):
            metrics = {"status": metrics}
        return (summary, {item["class"]: item["probability"] for item in result["top_three"]}, warning,
                cam["original"], cam["heatmap"], cam["overlay"], explanation,
                information_markdown(result["information"]), metrics,
                "Supported classes:\n\n" + "\n".join("- " + label for label in predictor.classes))

    def background_analysis(image, architecture, threshold, request: gr.Request):
        session = request.session_hash
        with jobs_lock:
            job = jobs.get(session)
        if not job or job[1:4] != (str(image), architecture, threshold) or time.monotonic() - job[0] > 180:
            return gr.skip()
        if not agent.enabled:
            return FALLBACK
        # Only model execution holds this lock; network waiting cannot delay a new CNN result.
        def secondary(name):
            class LockedPredictor:
                def predict_plant_disease(self, *args, **kwargs):
                    with inference_lock:
                        return load(name).predict_plant_disease(*args, **kwargs)
            return LockedPredictor()
        outcome = interpret(image, architecture, job[4], secondary, agent)
        with jobs_lock:
            if jobs.get(session) is not job:
                return gr.skip()  # An upload/reset/new prediction invalidated this result.
            jobs.pop(session, None)
        return report_html(outcome)

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), title="AgriVision AI") as demo:
        gr.Markdown("# 🌿 AgriVision AI\nIntelligent Plant Disease Detection and Explainable AI")
        with gr.Row():
            image = gr.Image(type="filepath", label="Upload a leaf image", sources=["upload"], format="png")
            with gr.Column():
                architecture = gr.Dropdown(list(config.architectures), value=selected_default, label="Model (validation-selected default when available)")
                threshold = gr.Slider(0, 1, value=0, step=.01, label="Warning threshold override — 0 uses saved validation threshold / fallback")
                analyze_button = gr.Button("Analyze", variant="primary")
                clear = gr.Button("Upload another image / Clear")
        summary = gr.Markdown("Upload an image and select Analyze. Training must be completed first.")
        top = gr.Label(num_top_classes=3, label="Top three predictions")
        warning = gr.Markdown()
        with gr.Row():
            original = gr.Image(label="Original", interactive=False)
            heatmap = gr.Image(label="Grad-CAM heatmap", interactive=False)
            overlay = gr.Image(label="Overlay", interactive=False)
        explanation = gr.Markdown("Grad-CAM does not prove a prediction is correct.")
        information = gr.Markdown()
        gr.Markdown("Optional visual interpretation sends this uploaded image to an external analysis service when configured. "
                    "It does not change the classifier prediction or confidence.")
        analysis = gr.HTML(label="AI-assisted visual analysis", value="")
        with gr.Accordion("Model information", open=True):
            classes = gr.Markdown("Supported classes are loaded from the selected trained model.")
            metrics = gr.JSON(value={"status": "Evaluation not yet available"}, label="Actual saved evaluation")
            gr.Markdown(LIMITATIONS)
        outputs = [summary, top, warning, original, heatmap, overlay, explanation, information, metrics, classes]
        event = analyze_button.click(analyze, [image, architecture, threshold], outputs, concurrency_limit=1,
                                     concurrency_id="cnn", show_progress="full", api_name="analyze")
        event.success(background_analysis, [image, architecture, threshold], analysis, concurrency_limit=1,
                      concurrency_id="visual_analysis", api_name=False, show_progress="minimal")
        image.change(invalidate, outputs=analysis, queue=False, api_name=False)
        architecture.change(invalidate, outputs=analysis, queue=False, api_name=False)
        threshold.change(invalidate, outputs=analysis, queue=False, api_name=False)
        reset = gr.ClearButton([image, *outputs, analysis], value="Reset all")
        reset.click(invalidate, outputs=analysis, queue=False, api_name=False)
        clear.click(invalidate, outputs=analysis, queue=False, api_name=False)
        clear.click(lambda: (None, "", None, "", None, None, None, "", "", {"status": "Evaluation not yet available"}, ""),
                    outputs=[image, *outputs], queue=False)
    return demo.queue(default_concurrency_limit=1)
