"""Run the same pipeline from VS Code's terminal or Google Colab."""
import argparse
from .config import ProjectConfig, ProjectPaths


def main():
    parser = argparse.ArgumentParser(description="AgriVision AI research pipeline")
    parser.add_argument("command", choices=("environment", "prepare", "train", "evaluate", "compare", "demo", "trials", "select", "final-evaluate", "external-prepare", "external-evaluate", "summary"))
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--architecture", choices=("EfficientNetB0", "MobileNetV3Large"), default="EfficientNetB0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--external-root")
    parser.add_argument("--mapping")
    parser.add_argument("--include-adamw", action="store_true")
    parser.add_argument("--include-deeper-tail", action="store_true")
    parser.add_argument("--include-smoothing", action="store_true")
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
    elif args.command == "trials":
        from .environment import configure_runtime
        from .experiments import build_trial_plan, run_trials
        configure_runtime(config, paths.output / "runs")
        trials = build_trial_plan(config, args.include_adamw, args.include_deeper_tail, args.include_smoothing)
        print(run_trials(paths, config, trials))
    elif args.command == "select":
        from .experiments import select_best_trials, export_selected_models
        print(select_best_trials(paths))
        print(export_selected_models(paths))
    elif args.command == "final-evaluate":
        from .experiments import evaluate_selected_models
        print(evaluate_selected_models(paths))
    elif args.command == "external-prepare":
        from .external import prepare_external_dataset
        from .utils import read_json
        if not args.external_root or not args.mapping:
            parser.error("external-prepare needs --external-root and --mapping with reviewed=true")
        print(prepare_external_dataset(paths, args.external_root, read_json(args.mapping))[1])
    elif args.command == "external-evaluate":
        from .external import evaluate_external
        if not args.external_root:
            parser.error("external-evaluate needs --external-root")
        print(evaluate_external(args.architecture, paths, args.external_root))
    elif args.command == "summary":
        from .experiments import generate_final_comparison
        print(generate_final_comparison(paths))


if __name__ == "__main__":
    main()
