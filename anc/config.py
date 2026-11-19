"""Preflight validation for experiment configurations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real
from typing import Any


REQUIRED_KEYS = (
    "defender_seeds",
    "attacker_seeds",
    "defender_updates",
    "attacker_updates",
    "batch_size",
    "validation_size",
    "test_size",
    "validate_every",
    "learning_rate",
    "threads",
    "defender_seconds",
    "attacker_seconds",
    "baseline_channels",
    "wider_channels",
)
POSITIVE_INTEGERS = (
    "defender_updates",
    "attacker_updates",
    "batch_size",
    "validation_size",
    "test_size",
    "validate_every",
    "threads",
)
POSITIVE_NUMBERS = ("learning_rate", "defender_seconds", "attacker_seconds")


class ConfigError(ValueError):
    """Raised when an experiment configuration is unsafe or incomplete."""


def _validate_seeds(config: dict[str, Any], key: str) -> None:
    values = config[key]
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{key} must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ConfigError(f"{key} must contain non-negative integers")
    if len(values) != len(set(values)):
        raise ConfigError(f"{key} values must be unique")


def _validate_channels(config: dict[str, Any], key: str) -> None:
    values = config[key]
    if not isinstance(values, list) or len(values) != 4:
        raise ConfigError(f"{key} must contain exactly four channel widths")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ConfigError(f"{key} must contain positive integers")
    if values[-1] != 1:
        raise ConfigError(f"{key} must end in one output channel")


def validate_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Return an independent validated config or raise an actionable error."""

    if not isinstance(config, Mapping):
        raise ConfigError("Configuration must be a mapping")
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ConfigError("Missing required config keys: " + ", ".join(missing))
    result: dict[str, Any] = deepcopy(dict(config))
    for key in POSITIVE_INTEGERS:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{key} must be a positive integer")
    for key in POSITIVE_NUMBERS:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
            raise ConfigError(f"{key} must be a positive number")
        if not math.isfinite(float(value)):
            raise ConfigError(f"{key} must be a finite positive number")
    _validate_seeds(result, "defender_seeds")
    _validate_seeds(result, "attacker_seeds")
    if set(result["defender_seeds"]) & set(result["attacker_seeds"]):
        raise ConfigError("defender_seeds and attacker_seeds must be disjoint")
    _validate_channels(result, "baseline_channels")
    _validate_channels(result, "wider_channels")
    return result
