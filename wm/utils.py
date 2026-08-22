import argparse
from types import SimpleNamespace

import yaml


def merge_config(config_file: str, args: argparse.Namespace) -> argparse.Namespace:
    """Merges YAML config with argparse namespace, CLI args take priority."""
    with open(config_file, "r") as f:
        yml_config = yaml.safe_load(f)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return argparse.Namespace(**{**yml_config, **args_dict})


def to_namespace(obj):
    """Recursively convert a (possibly nested) dict/list -- e.g. straight
    from yaml.safe_load or an already-parsed config dict -- into (possibly
    nested) SimpleNamespace/list, so fields are read as attributes
    (cfg.foo.bar) instead of dict keys."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def load_config(config_file: str) -> SimpleNamespace:
    """Load a YAML file as a (possibly nested) SimpleNamespace, so fields are
    read as attributes (cfg.foo.bar) instead of dict keys -- for a plain
    model config with no CLI/merge involved (see merge_config for that case)."""
    with open(config_file, "r") as f:
        return to_namespace(yaml.safe_load(f))
