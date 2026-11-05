import pytest
import torch

from anc.core import generator, sample
from anc.security import (
    ciphertext_sensitivity,
    determinism_diagnostic,
    diagnostic_sample,
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
