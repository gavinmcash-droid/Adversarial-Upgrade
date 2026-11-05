"""Empirical ciphertext-leakage diagnostics on non-final-test data streams.

These measurements describe learned neural-network behavior. They are not proofs
of semantic security, computational security, or cryptographic confidentiality.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from .core import Eve, Keyed, freeze, generator, metrics, sample, stream_seed


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _alternative_key(seed: int, size: int) -> torch.Tensor:
    _, key = sample(generator(seed, "security_key_alternative"), size)
    return key


def _load_defender_checkpoint(path: Path) -> tuple[Keyed, Keyed]:
    state = torch.load(path, weights_only=True)
    if not isinstance(state, dict) or "alice" not in state or "bob" not in state:
        raise ValueError("Defender checkpoint must contain alice and bob state dictionaries")
    alice, bob = Keyed(), Keyed()
    alice.load_state_dict(state["alice"])
    bob.load_state_dict(state["bob"])
    freeze(alice)
    freeze(bob)
    return alice, bob


def _load_attackers(
    defender: Path,
    defender_hash: str,
    limitations: list[str],
) -> list[tuple[str, Eve]]:
    loaded: list[tuple[str, Eve]] = []
    for path in sorted(defender.glob("attack_*")):
        if not path.is_dir():
            continue
        checkpoint = path / "best.pt"
        try:
            status = _read_json(path / "status.json")
            parent = _read_json(path / "parent.json")
            if status.get("status") != "complete":
                limitations.append(f"Skipped {path.name}: status is not complete")
                continue
            if parent.get("checkpoint_sha256") != defender_hash:
                limitations.append(f"Skipped {path.name}: trained against a different defender checkpoint")
                continue
            if not checkpoint.is_file():
                limitations.append(f"Skipped {path.name}: missing {checkpoint.name}")
                continue
            channels = parent.get("channels")
            if not isinstance(channels, list) or len(channels) != 4:
                raise ValueError("parent.json has invalid channels")
            state = torch.load(checkpoint, weights_only=True)
            if not isinstance(state, dict) or "eve" not in state:
                raise ValueError("checkpoint has no Eve state dictionary")
            eve = Eve(tuple(int(value) for value in channels))
            eve.load_state_dict(state["eve"])
            freeze(eve)
            loaded.append((path.name, eve))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as error:
            limitations.append(f"Skipped {path.name}: {error}")
    return loaded


def _plot_reconstruction(report: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reconstruction = report["reconstruction"]
    labels = list(reconstruction)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("bit_error_rate", "normalized_l1")):
        axis.bar(labels, [reconstruction[label][metric] for label in labels], color="#2878B5")
        axis.set(title=metric.replace("_", " ").title(), ylabel="Diagnostic value")
        axis.tick_params(axis="x", rotation=35)
    fig.suptitle("Diagnostic-stream plaintext reconstruction")
    fig.savefig(output / "security_reconstruction.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for label in labels:
        axis.plot(range(16), reconstruction[label]["per_bit_error_rate"], marker="o", markersize=2.5, label=label)
    axis.set(
        title="Per-bit diagnostic reconstruction error",
        xlabel="Plaintext bit index",
        ylabel="Bit error rate",
        xticks=range(16),
        ylim=(0, 1),
    )
    axis.legend(fontsize=7, frameon=False)
    fig.savefig(output / "per_bit_recovery.png", dpi=300)
    plt.close(fig)


def _plot_sensitivity(report: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    images = []
    for axis, key, title in (
        (axes[0], "plaintext_sensitivity", "Plaintext-bit flips"),
        (axes[1], "key_sensitivity", "Key-bit flips"),
    ):
        image = axis.imshow(report[key]["normalized_l1_matrix"], vmin=0, vmax=1, cmap="viridis", aspect="auto")
        images.append(image)
        axis.set(title=title, xlabel="Ciphertext dimension", ylabel="Input bit")
    fig.colorbar(images[-1], ax=axes, label="Mean |ciphertext change| / 2", shrink=0.85)
    fig.suptitle("Ciphertext sensitivity on diagnostic data")
    fig.savefig(output / "ciphertext_sensitivity.png", dpi=300)
    plt.close(fig)


def _finish_output(temporary: Path, output: Path, report: dict[str, Any]) -> dict[str, Any]:
    (temporary / "security_report.json").write_text(json.dumps(report, indent=2) + "\n")
    os.rename(temporary, output)
    return report


def evaluate_security(defender: Path, output: Path, samples: int = 1024) -> dict[str, Any]:
    """Evaluate checkpoints on new diagnostic streams and write a new report directory.

    The function never reads or writes ``test_results.json`` and refuses to overwrite
    an existing output directory.
    """

    _positive_size(samples)
    defender, output = Path(defender), Path(output)
    if output.exists():
        raise FileExistsError(f"Security output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    interpretation = (
        "These empirical neural-network diagnostics are not a proof of semantic security, "
        "computational security, or cryptographic confidentiality."
    )
    try:
        limitations: list[str] = []
        try:
            metadata = _read_json(defender / "metadata.json")
            seed = metadata.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("metadata seed must be an integer")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            report = {
                "status": "unavailable",
                "samples": samples,
                "limitations": [f"Cannot load defender metadata: {error}"],
                "interpretation": interpretation,
            }
            return _finish_output(temporary, output, report)

        checkpoint = defender / "best.pt"
        if not checkpoint.is_file():
            report = {
                "status": "unavailable",
                "defender_seed": seed,
                "samples": samples,
                "limitations": [f"Missing trained defender checkpoint: {checkpoint}"],
                "interpretation": interpretation,
            }
            return _finish_output(temporary, output, report)
        try:
            alice, bob = _load_defender_checkpoint(checkpoint)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            report = {
                "status": "unavailable",
                "defender_seed": seed,
                "samples": samples,
                "limitations": [f"Cannot load trained defender checkpoint {checkpoint}: {error}"],
                "interpretation": interpretation,
            }
            return _finish_output(temporary, output, report)

        defender_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        attackers = _load_attackers(defender, defender_hash, limitations)
        if not attackers:
            limitations.append("No complete attacker checkpoints were available for keyless plaintext recovery")
        plaintext, key = diagnostic_sample(seed, samples)
        alternative_key = _alternative_key(seed, samples)
        with torch.no_grad():
            ciphertext = alice(plaintext, key)
            alternative_ciphertext = alice(plaintext, alternative_key)
            reconstruction = {
                "bob_with_key": reconstruction_diagnostics(bob(ciphertext, key), plaintext),
                **{
                    name: reconstruction_diagnostics(eve(ciphertext), plaintext)
                    for name, eve in attackers
                },
                "random_guess": reconstruction_diagnostics(random_guess(seed, samples), plaintext),
            }
        key_difference = (alternative_ciphertext - ciphertext).abs() / 2
        key_diversity = {
            "mean_normalized_l1": float(key_difference.mean().item()),
            "mean_sign_change_rate": float(
                ((alternative_ciphertext >= 0) != (ciphertext >= 0)).float().mean().item()
            ),
        }
        report = {
            "status": "complete" if not limitations else "partial",
            "defender_seed": seed,
            "samples": samples,
            "diagnostic_stream_seed": stream_seed(seed, "security_diagnostic"),
            "alternative_key_stream_seed": stream_seed(seed, "security_key_alternative"),
            "random_guess_stream_seed": stream_seed(seed, "security_random_guess"),
            "reconstruction": reconstruction,
            "plaintext_sensitivity": ciphertext_sensitivity(alice, plaintext, key, flip="plaintext"),
            "key_sensitivity": ciphertext_sensitivity(alice, plaintext, key, flip="key"),
            "key_diversity": key_diversity,
            "determinism": determinism_diagnostic(alice, plaintext, key),
            "limitations": limitations,
            "interpretation": interpretation,
        }
        _plot_reconstruction(report, temporary)
        _plot_sensitivity(report, temporary)
        return _finish_output(temporary, output, report)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defender", required=True, type=Path, help="Saved defender_<seed> directory")
    parser.add_argument("--output", required=True, type=Path, help="New directory for diagnostic outputs")
    parser.add_argument("--samples", type=int, default=1024, help="Number of newly generated diagnostic examples")
    args = parser.parse_args()
    report = evaluate_security(args.defender, args.output, args.samples)
    print(f"Security diagnostics {report['status']}: {args.output}")
    for limitation in report.get("limitations", []):
        print(f"- {limitation}")


if __name__ == "__main__":
    main()
