"""Load YAML configs with `extends` inheritance and dotted command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict with attribute access for nested keys: cfg.train.lr == cfg["train"]["lr"]."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _merge(_load_yaml(path.parent / parent), data)


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _parse_value(raw: str):
    value = yaml.safe_load(raw)
    if isinstance(value, str):          # YAML 1.1 reads "1e-4" (no dot) as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a config; overrides look like ["train.val_every=500", "dataset.max_episodes=10"]."""
    cfg = _load_yaml(Path(path))
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key, _parse_value(raw))
    return Config(cfg)
