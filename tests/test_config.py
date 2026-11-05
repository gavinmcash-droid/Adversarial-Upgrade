import json
from pathlib import Path
import subprocess
import sys

import pytest

from anc.config import ConfigError, validate_config


ROOT = Path(__file__).parents[1]


def _valid() -> dict:
    return json.loads((ROOT / "configs/smoke.json").read_text())


def test_existing_configs_validate_and_return_independent_copies():
    for path in sorted((ROOT / "configs").glob("*.json")):
        original = json.loads(path.read_text())
        validated = validate_config(original)
        assert validated == original
        assert validated is not original
        assert validated["defender_seeds"] is not original["defender_seeds"]


def test_missing_keys_are_reported_together():
    with pytest.raises(ConfigError) as error:
        validate_config({"defender_seeds": [1]})

    message = str(error.value)
    assert "Missing required config keys" in message
    assert "attacker_seeds" in message
    assert "learning_rate" in message


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("defender_updates", 0, "positive integer"),
        ("attacker_updates", True, "positive integer"),
        ("batch_size", -1, "positive integer"),
        ("validation_size", 1.5, "positive integer"),
        ("test_size", 0, "positive integer"),
        ("validate_every", 0, "positive integer"),
        ("threads", 0, "positive integer"),
        ("defender_seconds", 0, "positive number"),
        ("attacker_seconds", float("inf"), "finite positive number"),
        ("learning_rate", -0.1, "positive number"),
    ],
)
def test_numeric_constraints_are_actionable(key, value, message):
    config = _valid()
    config[key] = value

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("defender_seeds", [], "non-empty list"),
        ("attacker_seeds", [201, 201], "unique"),
        ("attacker_seeds", [101], "disjoint"),
        ("defender_seeds", [-1], "non-negative integers"),
        ("attacker_seeds", [True], "non-negative integers"),
    ],
)
def test_seed_constraints_are_actionable(key, value, message):
    config = _valid()
    config[key] = value

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("baseline_channels", [2, 4, 1], "exactly four"),
        ("wider_channels", [4, 8, 0, 1], "positive integers"),
        ("baseline_channels", [2, 4, 4, 2], "end in one output channel"),
    ],
)
def test_channel_constraints_are_actionable(key, value, message):
    config = _valid()
    config[key] = value

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_run_cli_rejects_invalid_config_before_creating_output(tmp_path):
    config = tmp_path / "invalid.json"
    root = tmp_path / "must-not-exist"
    config.write_text(json.dumps({"defender_seeds": [1]}))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "anc.run",
            "defender",
            "--config",
            str(config),
            "--root",
            str(root),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "Missing required config keys" in completed.stderr
    assert not root.exists()
