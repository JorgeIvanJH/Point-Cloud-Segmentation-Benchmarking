import argparse
import yaml
import os

AVAILABLE_MODELS = ['TestModel', 'PointNet', 'PointNet2']


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

    # General
    parser.add_argument('--config_path', type=str, default="config.yaml", help='Path to YAML config file')
    parser.add_argument('--experiment_name', type=str, default="exp_1", help='Name of the experiment')
    parser.add_argument('--model_name', type=str,
                        help=f'Name of the model to use for training in MODELS_PATH in config.yaml')
    
    # Training
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--weight_decay_coefficient', type=float, help='Weight decay for optimizer')
    parser.add_argument('--continue_from_epoch', type=int, default=-1, help='Resume from epoch (use -1 to start from scratch)')
    parser.add_argument('--load_encoder_weights', type=str2bool, default=False, help='Load pretrained encoder weights')

    # System
    parser.add_argument('--num_workers', type=int, help='Dataloader workers')

    # Model

    
    # Other
    parser.add_argument('--seed', type=int, help='Random seed')

    return parser.parse_args()


def merge_configs(yaml_config, cli_args):

    train_config = yaml_config.get("TRAIN", {})
    merged_config = {
        "SEED": cli_args.seed if cli_args.seed is not None else yaml_config.get("SEED", 42),
        "TRAIN_SIZE": yaml_config.get("TRAIN_SIZE", 0.8),
        "DATA_PATH": yaml_config["DATA_PATH"],
        "OBJECT_NAME": yaml_config["OBJECT_NAME"],
        "MODELS_PATH": yaml_config["MODELS_PATH"],
        "NUM_POINTS_PER_SEG_SAMPLE": yaml_config["NUM_POINTS_PER_SEG_SAMPLE"],
        "DEVICE": yaml_config.get("DEVICE", "cpu"),
        "EXPERIMENT_NAME": cli_args.experiment_name,
        "MODEL_NAME": cli_args.model_name.lower() if cli_args.model_name is not None else train_config.get("NAME", "testmodel").lower(),
        "BATCH_SIZE": cli_args.batch_size if cli_args.batch_size is not None else train_config.get("BATCH_SIZE", 32),
        "LR": cli_args.lr if cli_args.lr is not None else train_config.get("LR", 0.001),
        "EPOCHS": cli_args.num_epochs if cli_args.num_epochs is not None else train_config.get("EPOCHS", 100),
        "WEIGHT_DECAY": cli_args.weight_decay_coefficient if cli_args.weight_decay_coefficient is not None else train_config.get("WEIGHT_DECAY", 0),
        "CONTINUE_FROM_EPOCH": cli_args.continue_from_epoch,
        "LOAD_ENCODER_WEIGHTS": cli_args.load_encoder_weights,
        "NUM_WORKERS": cli_args.num_workers if cli_args.num_workers is not None else train_config.get("NUM_WORKERS", 4),
        "SHUFFLE": train_config.get("SHUFFLE", True)
    }

    available_models = [f.replace(".py","") for f in os.listdir(merged_config["MODELS_PATH"]) if f.endswith('.py')]
    print(f"Available models: {available_models}")
    assert merged_config["MODEL_NAME"] in available_models, f"Model {merged_config['MODEL_NAME']} not found in {merged_config['MODELS_PATH']}. Available models: {available_models}"
    
    return merged_config


def get_config():
    cli_args = parse_args()
    yaml_config = load_yaml_config(cli_args.config_path)
    final_config = merge_configs(yaml_config, cli_args)
    return final_config
