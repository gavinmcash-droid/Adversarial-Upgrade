"""Filesystem boundary checks shared by read-only reporting commands."""

from pathlib import Path


def require_external_output(input_path: Path, output_path: Path) -> None:
    """Reject an output equal to or nested beneath the protected input tree."""

    protected = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if destination == protected or protected in destination.parents:
        raise ValueError(f"Output must not be inside the input tree: {destination}")
