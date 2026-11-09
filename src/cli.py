"""Run the same pipeline from VS Code's terminal or Google Colab."""
import argparse
from .config import ProjectConfig, ProjectPaths


def main():
    parser = argparse.ArgumentParser(description="AgriVision AI research pipeline")
    parser.add_argument("command", choices=("environment", "prepare", "train", "evaluate", "compare", "demo"))
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--architecture", choices=("EfficientNetB0", "MobileNetV3Large"), default="EfficientNetB0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    config = ProjectConfig.load(args.config) if args.config else ProjectConfig()
    paths = ProjectPaths.create(args.dataset, args.output).initialize()
    if args.command == "environment":
        from .environment import configure_runtime
        configure_runtime(config, paths.output / "runs")
    elif args.command == "prepare":
        from .dataset import prepare_dataset
        print(prepare_dataset(paths, config)[2])
    elif args.command == "train":
        from .environment import configure_runtime
        from .training import train_architecture
        configure_runtime(config, paths.output / "runs")
        print(train_architecture(args.architecture, paths, config))
    elif args.command == "evaluate":
        from .environment import configure_runtime
        from .evaluation import evaluate_architecture
        configure_runtime(config, paths.output / "runs")
        print(evaluate_architecture(args.architecture, paths, config))
    elif args.command == "compare":
        from .evaluation import compare_models
        print(compare_models(paths, config.architectures))
    elif args.command == "demo":
        from .app import create_app
        create_app(paths, config).launch(share=args.share)


if __name__ == "__main__":
    main()
