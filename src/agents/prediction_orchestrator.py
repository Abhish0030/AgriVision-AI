"""Compare independently trained classifiers without altering their predictions."""
from html import escape
from .nemotron_agent import FALLBACK

ARCHITECTURES = ("EfficientNetB0", "MobileNetV3Large")


def build_context(predictions, primary):
    if set(predictions) != set(ARCHITECTURES):
        raise ValueError("Both trained CNN outputs are required.")
    reduced = {name: {key: result[key] for key in (
        "class_name", "crop", "disease", "confidence_percent", "top_three")} for name, result in predictions.items()}
    facts = []
    for result in predictions.values():
        info = result["information"]
        if info.get("verified") is True and info.get("source_url", "").startswith("https://") and info not in facts:
            facts.append(info)
    return {"predictions": reduced, "primary_class": predictions[primary]["class_name"],
            "cnn_analysis": {"efficientnet_prediction": predictions[ARCHITECTURES[0]]["class_name"],
                             "mobilenet_prediction": predictions[ARCHITECTURES[1]]["class_name"],
                             "models_agree": predictions[ARCHITECTURES[0]]["class_name"] == predictions[ARCHITECTURES[1]]["class_name"]},
            "verified_information": facts}


def interpret(image, primary, primary_result, loader, agent):
    if not agent.enabled:
        return {"status": "disabled", "report": None, "latency_seconds": 0.}
    predictions = {primary: primary_result}
    try:
        for name in ARCHITECTURES:
            if name != primary:
                predictions[name] = loader(name).predict_plant_disease(image, explain=False)
        return agent.analyze(image, build_context(predictions, primary))
    except Exception:
        return {"status": "failed", "report": None, "latency_seconds": 0.}


def report_html(outcome):
    report = outcome.get("report")
    if not report:
        return FALLBACK
    def paragraph(label, text):
        return "<p><strong>" + label + ":</strong> " + escape(text) + "</p>"
    visual, assessment = report["visual_analysis"], report["agent_assessment"]
    output = paragraph("Visual observations (tentative)", "; ".join(visual["visible_symptoms"]) or "None reported")
    output += paragraph("Image quality", visual["image_quality"])
    output += paragraph("Possible indicators", "; ".join(visual["possible_disease_indicators"]) or "Uncertain")
    output += paragraph("CNN agreement", "Agree" if report["cnn_analysis"]["models_agree"] else "Disagree; review recommended")
    output += paragraph("Visual assessment vs selected CNN", assessment["agreement_with_cnn"])
    output += paragraph("Explanation", assessment["explanation"])
    output += paragraph("Uncertainty", report["reliability"]["warning"])
    output += paragraph("Prevention from verified records", "; ".join(report["recommendations"]) or "No verified recommendation returned")
    output += paragraph("Information sources", "; ".join(report["information_sources"]) or "No verified source returned")
    return output + "<p>Visual observations are unverified model-generated interpretations. CNN predictions and confidence are unchanged.</p>"
