"""Empirical ciphertext-leakage diagnostics on non-final-test data streams.

These measurements describe learned neural-network behavior. They are not proofs
of semantic security, computational security, or cryptographic confidentiality.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .core import generator, metrics, sample


Encryptor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _positive_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Diagnostic sample size must be a positive integer")


def diagnostic_sample(seed: int, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reproducible plaintext/key tensors from a diagnostic-only stream."""

    _positive_size(size)
    return sample(generator(seed, "security_diagnostic"), size)


def random_guess(seed: int, size: int) -> torch.Tensor:
    """Return independent ±1 plaintext guesses from a diagnostic-only stream."""

    _positive_size(size)
    guesses, _ = sample(generator(seed, "security_random_guess"), size)
    return guesses


def _matching_messages(prediction: torch.Tensor, plaintext: torch.Tensor) -> None:
    if prediction.ndim != 2 or plaintext.ndim != 2 or prediction.shape != plaintext.shape:
        raise ValueError("Prediction and plaintext must have the same two-dimensional shape")
    if prediction.shape[0] == 0 or prediction.shape[1] == 0:
        raise ValueError("Prediction and plaintext must be non-empty")
    if not torch.isfinite(prediction).all() or not torch.isfinite(plaintext).all():
        raise ValueError("Prediction and plaintext must contain only finite values")


def reconstruction_diagnostics(prediction: torch.Tensor, plaintext: torch.Tensor) -> dict[str, Any]:
    """Report aggregate and per-bit plaintext reconstruction behavior."""

    _matching_messages(prediction, plaintext)
    wrong = (prediction >= 0) != (plaintext >= 0)
    result: dict[str, Any] = metrics(prediction, plaintext)
    per_bit_error = wrong.float().mean(dim=0)
    result.update(
        {
            "examples": int(plaintext.shape[0]),
            "bits_per_message": int(plaintext.shape[1]),
            "per_bit_error_rate": [float(value) for value in per_bit_error.tolist()],
            "bitwise_accuracy": [float(value) for value in (1 - per_bit_error).tolist()],
        }
    )
    return result


def _ciphertext(encrypt: Encryptor, plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    value = encrypt(plaintext, key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != plaintext.shape[0]:
        raise ValueError("Encryptor must return a two-dimensional tensor with the same batch size")
    if value.shape[1] == 0 or not torch.isfinite(value).all():
        raise ValueError("Ciphertext must be non-empty and finite")
    return value


def ciphertext_sensitivity(
    encrypt: Encryptor,
    plaintext: torch.Tensor,
    key: torch.Tensor,
    flip: str,
) -> dict[str, Any]:
    """Measure ciphertext displacement after flipping each plaintext or key bit."""

    _matching_messages(plaintext, key)
    if flip not in {"plaintext", "key"}:
        raise ValueError("flip must be plaintext or key")
    source = plaintext if flip == "plaintext" else key
    with torch.no_grad():
        baseline = _ciphertext(encrypt, plaintext, key)
        displacement_rows = []
        sign_rows = []
        for bit in range(source.shape[1]):
            changed_plaintext = plaintext.clone()
            changed_key = key.clone()
            target = changed_plaintext if flip == "plaintext" else changed_key
            target[:, bit] *= -1
            changed = _ciphertext(encrypt, changed_plaintext, changed_key)
            if changed.shape != baseline.shape:
                raise ValueError("Encryptor returned inconsistent ciphertext shapes")
            displacement_rows.append(((changed - baseline).abs() / 2).mean(dim=0))
            sign_rows.append(((changed >= 0) != (baseline >= 0)).float().mean(dim=0))
    displacement = torch.stack(displacement_rows)
    sign_change = torch.stack(sign_rows)
    return {
        "flip": flip,
        "examples": int(plaintext.shape[0]),
        "input_bits": int(source.shape[1]),
        "ciphertext_dimensions": int(baseline.shape[1]),
        "mean_normalized_l1": float(displacement.mean().item()),
        "mean_sign_change_rate": float(sign_change.mean().item()),
        "per_input_bit_normalized_l1": [float(value) for value in displacement.mean(dim=1).tolist()],
        "per_input_bit_sign_change_rate": [float(value) for value in sign_change.mean(dim=1).tolist()],
        "normalized_l1_matrix": [[float(value) for value in row] for row in displacement.tolist()],
        "sign_change_matrix": [[float(value) for value in row] for row in sign_change.tolist()],
    }


def determinism_diagnostic(
    encrypt: Encryptor,
    plaintext: torch.Tensor,
    key: torch.Tensor,
) -> dict[str, float | bool]:
    """Repeat identical inputs and quantify any ciphertext change."""

    _matching_messages(plaintext, key)
    with torch.no_grad():
        first = _ciphertext(encrypt, plaintext, key)
        second = _ciphertext(encrypt, plaintext, key)
    if first.shape != second.shape:
        raise ValueError("Encryptor returned inconsistent ciphertext shapes")
    difference = (second - first).abs()
    return {
        "exact_match": bool(torch.equal(first, second)),
        "max_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
    }
