import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from anc.core import BASE, WIDE, Eve, Keyed, generator, sample, stream_seed
from anc.security import (
    ciphertext_sensitivity,
    determinism_diagnostic,
    diagnostic_sample,
    evaluate_security,
    random_guess,
    reconstruction_diagnostics,
)


def test_diagnostic_sample_is_reproducible_float32_and_not_final_test_stream():
    plaintext, key = diagnostic_sample(77, 64)
    repeated_plaintext, repeated_key = diagnostic_sample(77, 64)
    final_plaintext, final_key = sample(generator(77, "final_test"), 64)

    assert plaintext.shape == key.shape == (64, 16)
    assert plaintext.dtype == key.dtype == torch.float32
    assert set(plaintext.unique().tolist()) == {-1.0, 1.0}
    assert torch.equal(plaintext, repeated_plaintext)
    assert torch.equal(key, repeated_key)
    assert not torch.equal(plaintext, final_plaintext)
    assert not torch.equal(key, final_key)


def test_reconstruction_diagnostics_has_hand_calculated_bitwise_errors():
    plaintext = torch.ones((2, 16))
    prediction = plaintext.clone()
    prediction[0, 0] = -1
    prediction[1, 1] = -1

    result = reconstruction_diagnostics(prediction, plaintext)

    assert result["bit_error_rate"] == 2 / 32
    assert result["message_accuracy"] == 0.0
    assert result["normalized_l1"] == 2 / 32
    assert result["per_bit_error_rate"][:3] == [0.5, 0.5, 0.0]
    assert result["bitwise_accuracy"][:3] == [0.5, 0.5, 1.0]
    assert len(result["per_bit_error_rate"]) == 16


def test_independent_random_guess_is_reproducible_and_near_chance():
    plaintext, _ = diagnostic_sample(9, 10_000)
    guess = random_guess(9, len(plaintext))

    result = reconstruction_diagnostics(guess, plaintext)

    assert torch.equal(guess, random_guess(9, len(plaintext)))
    assert not torch.equal(guess, plaintext)
    assert abs(result["bit_error_rate"] - 0.5) < 0.01
    assert result["normalized_l1"] == result["bit_error_rate"]


def test_plaintext_flip_sensitivity_is_hand_calculated_for_identity_cipher():
    plaintext, key = diagnostic_sample(12, 5)

    result = ciphertext_sensitivity(lambda message, secret: message, plaintext, key, flip="plaintext")

    assert result["examples"] == 5
    assert result["input_bits"] == result["ciphertext_dimensions"] == 16
    assert result["mean_normalized_l1"] == pytest.approx(1 / 16)
    assert result["mean_sign_change_rate"] == pytest.approx(1 / 16)
    assert result["per_input_bit_normalized_l1"] == pytest.approx([1 / 16] * 16)
    assert result["per_input_bit_sign_change_rate"] == pytest.approx([1 / 16] * 16)
    matrix = torch.tensor(result["normalized_l1_matrix"])
    torch.testing.assert_close(matrix, torch.eye(16), rtol=0, atol=0)


def test_key_flip_sensitivity_distinguishes_unused_and_used_key():
    plaintext, key = diagnostic_sample(13, 4)

    unused = ciphertext_sensitivity(lambda message, secret: message, plaintext, key, flip="key")
    used = ciphertext_sensitivity(lambda message, secret: secret, plaintext, key, flip="key")

    assert unused["mean_normalized_l1"] == 0.0
    assert unused["mean_sign_change_rate"] == 0.0
    assert used["mean_normalized_l1"] == pytest.approx(1 / 16)
    assert used["mean_sign_change_rate"] == pytest.approx(1 / 16)


def test_determinism_diagnostic_detects_repeat_changes():
    plaintext, key = diagnostic_sample(14, 3)
    stable = determinism_diagnostic(lambda message, secret: message * secret, plaintext, key)
    calls = 0

    def changing(message, secret):
        nonlocal calls
        calls += 1
        return message + calls * 0.25

    unstable = determinism_diagnostic(changing, plaintext, key)

    assert stable == {"exact_match": True, "max_absolute_difference": 0.0, "mean_absolute_difference": 0.0}
    assert unstable["exact_match"] is False
    assert unstable["max_absolute_difference"] == pytest.approx(0.25)
    assert unstable["mean_absolute_difference"] == pytest.approx(0.25)


@pytest.mark.parametrize("flip", ["message", "secret", "", None])
def test_sensitivity_rejects_unknown_flip_target(flip):
    plaintext, key = diagnostic_sample(15, 2)

    with pytest.raises(ValueError, match="plaintext or key"):
        ciphertext_sensitivity(lambda message, secret: message, plaintext, key, flip=flip)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def _checkpoint_run(root: Path) -> Path:
    torch.manual_seed(123)
    defender = root / "defender_31"
    defender.mkdir(parents=True)
    config = {
        "attacker_seeds": [41],
        "attacker_updates": 5,
        "baseline_channels": list(BASE),
        "wider_channels": list(WIDE),
    }
    _write_json(defender / "metadata.json", {"seed": 31, "stage": "defender", "config": config})
    _write_json(defender / "status.json", {"status": "complete", "updates": 5})
    alice, bob = Keyed(), Keyed()
    torch.save({"alice": alice.state_dict(), "bob": bob.state_dict(), "step": 5}, defender / "best.pt")
    for architecture, channels in (("baseline", BASE), ("wider", WIDE)):
        attack = defender / f"attack_{architecture}_41"
        attack.mkdir()
        _write_json(attack / "metadata.json", {"seed": 41, "stage": "attack", "config": config})
        _write_json(attack / "status.json", {"status": "complete", "updates": 5})
        _write_json(attack / "parent.json", {"channels": list(channels), "checkpoint_sha256": "test"})
        eve = Eve(channels)
        torch.save({"eve": eve.state_dict(), "step": 5}, attack / "best.pt")
    _write_json(defender / "test_results.json", {"reserved": "must remain unchanged"})
    return defender


def test_evaluate_security_loads_checkpoints_writes_diagnostics_and_preserves_final_test(tmp_path):
    defender = _checkpoint_run(tmp_path / "study")
    output = tmp_path / "security"
    test_result = defender / "test_results.json"
    before = (test_result.read_bytes(), test_result.stat().st_mtime_ns)

    report = evaluate_security(defender, output, samples=8)

    assert (test_result.read_bytes(), test_result.stat().st_mtime_ns) == before
    assert report["status"] == "complete"
    assert report["samples"] == 8
    assert report["diagnostic_stream_seed"] == stream_seed(31, "security_diagnostic")
    assert set(report["reconstruction"]) == {
        "bob_with_key",
        "attack_baseline_41",
        "attack_wider_41",
        "random_guess",
    }
    assert len(report["reconstruction"]["attack_baseline_41"]["per_bit_error_rate"]) == 16
    assert report["plaintext_sensitivity"]["input_bits"] == 16
    assert report["key_sensitivity"]["ciphertext_dimensions"] == 16
    assert set(report["key_diversity"]) == {
        "mean_normalized_l1",
        "mean_sign_change_rate",
    }
    assert report["determinism"]["exact_match"] is True
    assert "not a proof" in report["interpretation"].lower()
    expected = {
        "security_report.json",
        "security_reconstruction.png",
        "ciphertext_sensitivity.png",
        "per_bit_recovery.png",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        evaluate_security(defender, output, samples=8)


def test_missing_defender_checkpoint_writes_structured_unavailable_report(tmp_path):
    defender = tmp_path / "defender_2"
    defender.mkdir()
    _write_json(defender / "metadata.json", {"seed": 2, "stage": "defender", "config": {}})
    final_test = defender / "test_results.json"
    _write_json(final_test, {"reserved": True})
    before = final_test.read_bytes()

    report = evaluate_security(defender, tmp_path / "security", samples=4)

    assert final_test.read_bytes() == before
    assert report["status"] == "unavailable"
    assert any("best.pt" in limitation for limitation in report["limitations"])
    assert json.loads((tmp_path / "security/security_report.json").read_text()) == report
    assert list((tmp_path / "security").iterdir()) == [tmp_path / "security/security_report.json"]


def test_security_module_cli_reports_unavailable_checkpoint(tmp_path):
    defender = tmp_path / "defender_3"
    defender.mkdir()
    _write_json(defender / "metadata.json", {"seed": 3, "stage": "defender", "config": {}})
    output = tmp_path / "security"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "anc.security",
            "--defender",
            str(defender),
            "--output",
            str(output),
            "--samples",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "unavailable" in completed.stdout.lower()
    assert output.joinpath("security_report.json").is_file()
