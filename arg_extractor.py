import argparse
import yaml
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_yaml_config(filepath="config.yaml"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found at {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Segmentation Models")

    # Only essential CLI overrides
    parser.add_argument('--config_path', type=str, default="config.yaml", help='Path to YAML config file')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('--model_name', type=str, help='Model to use (must match a .py file in MODELS_PATH within then .yaml config file defined in --config_path)')
    parser.add_argument('--continue_from_epoch', type=int, help='Resume from epoch (-1: from scratch, -2: last checkpoint, int: specific epoch)')
    parser.add_argument('--num_workers', type=int, help='Dataloader workers override')
    parser.add_argument('--seed', type=int, help='Random seed override')

    return parser.parse_args()


def merge_configs(yaml_config, cli_args):
    train_config = yaml_config.get("TRAIN", {})

    merged_config = {
        # CLI overrides
        "SEED": cli_args.seed if cli_args.seed is not None else yaml_config.get("SEED", 42),
        "EXPERIMENT_NAME": cli_args.experiment_name if cli_args.experiment_name else train_config.get("EXPERIMENT_NAME", "exp_1"),
        "MODEL_NAME": cli_args.model_name if cli_args.model_name else train_config.get("NAME", "testmodel"),
        "CONTINUE_FROM_EPOCH": cli_args.continue_from_epoch if cli_args.continue_from_epoch else train_config.get("CONTINUE_FROM_EPOCH", -1),
        "NUM_WORKERS": cli_args.num_workers if cli_args.num_workers else train_config.get("NUM_WORKERS", 4),
        # YAML config
        "TRAIN_SIZE": yaml_config.get("TRAIN_SIZE", 0.8),
        "DATA_PATH": yaml_config["DATA_PATH"],
        "OBJECT_NAME": yaml_config["OBJECT_NAME"],
        "MODELS_PATH": yaml_config["MODELS_PATH"],
        "NUM_POINTS_PER_SEG_SAMPLE": yaml_config["NUM_POINTS_PER_SEG_SAMPLE"],
        "DEVICE": yaml_config.get("DEVICE", "cpu"),
        "USE_COLORS": yaml_config.get("USE_COLORS", True),
        "BATCH_SIZE": train_config.get("BATCH_SIZE", 32),
        "LR": train_config.get("LR", 0.001),
        "EPOCHS": train_config.get("EPOCHS", 100),
        "WEIGHT_DECAY": train_config.get("WEIGHT_DECAY", 0),
        "SHUFFLE": train_config.get("SHUFFLE", True),
        "NORMALIZE": train_config.get("NORMALIZE", True),
        "PIN_MEMORY": train_config.get("PIN_MEMORY", True),
    }

    available_models = [f.replace(".py", "") for f in os.listdir(merged_config["MODELS_PATH"]) if f.endswith('.py')]
    print(f"Available models: {available_models}")
    assert merged_config["MODEL_NAME"] in available_models, (
        f"Model '{merged_config['MODEL_NAME']}' not found in {merged_config['MODELS_PATH']}.\n"
        f"Available models: {available_models}"
    )

    return merged_config


def get_config():
    cli_args = parse_args()
    yaml_config = load_yaml_config(cli_args.config_path)
    return merge_configs(yaml_config, cli_args)
