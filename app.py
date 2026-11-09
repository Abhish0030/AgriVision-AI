"""Local launch: python app.py --output /path/to/persistent/experiment."""
import argparse
from src.app import create_app
from src.config import ProjectConfig, ProjectPaths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    create_app(ProjectPaths.create(output=args.output).initialize(), ProjectConfig()).launch(share=args.share)
