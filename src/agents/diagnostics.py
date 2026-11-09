"""Explicit live checks. Print only statuses; never responses or credentials."""
import argparse
from pathlib import Path
import httpx
from .nemotron_agent import NemotronAgent
from .openrouter_client import BASE_URL, AgentUnavailable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="Optional single leaf image; requires both trained CNN checkpoints.")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    agent = NemotronAgent(repository)
    try:
        response = httpx.get(BASE_URL + "/models", timeout=20, follow_redirects=False)
        response.raise_for_status()
        model = next((m for m in response.json()["data"] if m["id"] == agent._client.model), None)
        print("Public catalog reachable:", True)
        print("Configured model lists image support:", bool(model and "image" in model.get("architecture", {}).get("input_modalities", [])))
    except (httpx.HTTPError, ValueError, KeyError):
        print("Public catalog check unavailable")
    if not agent.enabled:
        print("Authenticated text/image checks NOT RUN: private configuration missing or mode disabled.")
        return
    try:
        text = agent._client.complete([{"role": "system", "content": agent._prompt},
                                       {"role": "user", "content": "Connection diagnostic. Return the word OK only."}])
        print("Authenticated text response received:", bool(text.strip()))
    except AgentUnavailable:
        print("Authenticated text check failed (details withheld).")
    if args.image:
        from ..config import ProjectPaths
        from ..prediction import PlantDiseasePredictor
        from .prediction_orchestrator import ARCHITECTURES, build_context
        paths = ProjectPaths.create()
        try:
            predictions = {name: PlantDiseasePredictor(paths, name).predict_plant_disease(args.image, explain=False)
                           for name in ARCHITECTURES}
            outcome = agent.analyze(args.image, build_context(predictions, ARCHITECTURES[0]))
            print("Authenticated image + validated JSON:", outcome["status"] == "ok")
        except Exception:
            print("Image check unavailable; check image and trained artifacts privately.")
    else:
        print("Image check NOT RUN: provide --image pointing to one local leaf image.")


if __name__ == "__main__":
    main()
