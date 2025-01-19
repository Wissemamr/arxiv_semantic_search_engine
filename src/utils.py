import yaml
from box import Box
import re


def load_config():
    """
    Load the YAML config
    """
    with open("config/config.yaml", "r") as f:
        return Box(yaml.safe_load(f))


def clean_text(input_str: str = None) -> str:
    """Remove punctuation, special characters and extra whitespaces from string"""
    if not input_str:
        return ""
    cleaned_str = re.sub(r"[^\w\s]", "", input_str)
    cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()
    return cleaned_str
