"""Retrieve static, reviewed information; never generate treatment advice."""
from pathlib import Path
from .utils import read_json


def class_display(label):
    crop, _, disease = label.partition("___")
    return ("Maize (Corn)" if crop.startswith("Corn") else crop, disease.replace("_", " "))


def get_disease_information(label, json_path=None):
    path = Path(json_path) if json_path else Path(__file__).resolve().parents[1] / "disease_info.json"
    database = read_json(path) if path.exists() else {}
    crop, disease = class_display(label)
    fallback = {"crop": crop, "disease": disease, "verified": False,
                "description": "Information not verified yet", "symptoms": [], "prevention": [], "source_url": None}
    entry = database.get(label, fallback)
    return entry if entry.get("verified") and entry.get("source_url") else {**fallback, "crop": crop, "disease": disease}


def information_markdown(entry):
    if not entry["verified"]:
        return "Information not verified yet"
    symptoms = "\n".join("- " + item for item in entry["symptoms"])
    prevention = "\n".join("- " + item for item in entry["prevention"])
    extra = f"\n\n[Additional source]({entry['additional_source_url']})" if entry.get("additional_source_url") else ""
    return (f"{entry['description']}\n\n**Common symptoms**\n\n{symptoms}\n\n"
            f"**General prevention precautions**\n\n{prevention}\n\n"
            f"[Source: {entry.get('source_title', 'University extension')}]({entry['source_url']}){extra}\n\n"
            "General educational information. Ask a local agricultural expert about diagnosis and treatment.")
