"""Green Gradio Blocks demo; model metrics always come from saved evaluation."""
import gradio as gr
from .disease_information import information_markdown
from .prediction import PlantDiseasePredictor
from .utils import read_json


LIMITATIONS = ("Research demonstration for supported crop classes. PlantVillage has controlled backgrounds; "
               "field performance may be worse. Softmax confidence is uncalibrated. This system does not detect "
               "all diseases, verify leaf content, or replace agricultural diagnosis. Grad-CAM indicates "
               "class-sensitive regions and does not establish a correct diagnosis.")


def create_app(paths, config):
    cache = {}
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

    def analyze(image, architecture, threshold):
        if not image:
            raise gr.Error("Upload a JPG, JPEG or PNG leaf image first.")
        try:
            predictor = load(architecture)
            result = predictor.predict_plant_disease(image, None if threshold == 0 else threshold)
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
        with gr.Accordion("Model information", open=True):
            classes = gr.Markdown("Supported classes are loaded from the selected trained model.")
            metrics = gr.JSON(value={"status": "Evaluation not yet available"}, label="Actual saved evaluation")
            gr.Markdown(LIMITATIONS)
        outputs = [summary, top, warning, original, heatmap, overlay, explanation, information, metrics, classes]
        analyze_button.click(analyze, [image, architecture, threshold], outputs, concurrency_limit=1, show_progress="full", api_name="analyze")
        gr.ClearButton([image, *outputs], value="Reset all")
        clear.click(lambda: (None, "", None, "", None, None, None, "", "", {"status": "Evaluation not yet available"}, ""),
                    outputs=[image, *outputs], queue=False)
    return demo.queue(default_concurrency_limit=1)
