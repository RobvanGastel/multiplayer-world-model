import argparse

import yaml


def merge_config(config_file: str, args: argparse.Namespace) -> argparse.Namespace:
    """Merges YAML config with argparse namespace, CLI args take priority."""
    with open(config_file, "r") as f:
        yml_config = yaml.safe_load(f)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return argparse.Namespace(**{**yml_config, **args_dict})
