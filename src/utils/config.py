"""
Configuration loading utilities.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration
    """
    merged = {}
    for config in configs:
        _deep_update(merged, config)
    return merged


def _deep_update(base: Dict, update: Dict) -> Dict:
    """
    Deep update nested dictionary.

    Args:
        base: Base dictionary
        update: Dictionary with updates

    Returns:
        Updated dictionary
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
