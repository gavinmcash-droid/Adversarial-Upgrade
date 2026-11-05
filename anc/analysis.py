"""Read-only discovery and paired analysis of saved ANC experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from statistics import fmean
from typing import Any


METRIC_NAMES = ("normalized_l1", "bit_error_rate", "message_accuracy")
DIFFERENCE_NAMES = (*METRIC_NAMES, "seconds", "parameters")
ATTACK_PATTERN = re.compile(r"attack_(baseline|wider)_(\d+)$")
DEFENDER_PATTERN = re.compile(r"defender_(\d+)$")


@dataclass(frozen=True)
class AnalysisIssue:
    """A path-specific reason an artifact or comparison is unusable."""

    code: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass
class AttackerRun:
    path: Path
    defender_seed: int
    attacker_seed: int
    architecture: str
    config: dict[str, Any]
    status: dict[str, Any]
    parent: dict[str, Any]
    validation_history: list[dict[str, Any]]
    final_metrics: dict[str, float]
    valid: bool = True

    @property
    def seconds(self) -> float:
        return float(self.status["seconds"])

    @property
    def parameters(self) -> int:
        return int(self.status["parameters"])


@dataclass
class DefenderRun:
    path: Path
    seed: int
    config: dict[str, Any]
    status: dict[str, Any]
    validation_history: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    attackers: list[AttackerRun] = field(default_factory=list)
    valid: bool = True


@dataclass
class AnalysisInventory:
    root: Path
    defenders: list[DefenderRun]
    issues: list[AnalysisIssue]


@dataclass(frozen=True)
class ComparisonPair:
    defender_seed: int
    attacker_seed: int
    baseline: AttackerRun
    wider: AttackerRun

    @property
    def differences(self) -> dict[str, float | int]:
        """Return wider-minus-baseline differences; negative errors favor wider Eve."""

        values: dict[str, float | int] = {
            name: self.wider.final_metrics[name] - self.baseline.final_metrics[name]
            for name in METRIC_NAMES
        }
        values["seconds"] = self.wider.seconds - self.baseline.seconds
        values["parameters"] = self.wider.parameters - self.baseline.parameters
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "defender_seed": self.defender_seed,
            "attacker_seed": self.attacker_seed,
            "baseline": {
                **self.baseline.final_metrics,
                "seconds": self.baseline.seconds,
                "parameters": self.baseline.parameters,
            },
            "wider": {
                **self.wider.final_metrics,
                "seconds": self.wider.seconds,
                "parameters": self.wider.parameters,
            },
            "differences": self.differences,
        }


def _issue(issues: list[AnalysisIssue], code: str, path: Path, message: str) -> None:
    issues.append(AnalysisIssue(code, str(path), message))


def _read_json(path: Path, issues: list[AnalysisIssue]) -> dict[str, Any] | None:
    if not path.is_file():
        _issue(issues, "missing_file", path, "Required JSON file is missing")
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _issue(issues, "malformed_json", path, f"Cannot read JSON: {error}")
        return None
    if not isinstance(value, dict):
        _issue(issues, "malformed_json", path, "Top-level JSON value must be an object")
        return None
    return value


def _read_jsonl(path: Path, issues: list[AnalysisIssue]) -> list[dict[str, Any]]:
    if not path.is_file():
        _issue(issues, "missing_file", path, "Required validation history is missing")
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError) as error:
        _issue(issues, "malformed_jsonl", path, f"Cannot read JSONL: {error}")
        return []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            _issue(issues, "malformed_jsonl", path, f"Line {line_number}: {error}")
            return []
        if not isinstance(row, dict):
            _issue(issues, "malformed_jsonl", path, f"Line {line_number} is not an object")
            return []
        rows.append(row)
    return rows


def _metrics(value: Any, path: Path, issues: list[AnalysisIssue]) -> dict[str, float]:
    if not isinstance(value, dict):
        _issue(issues, "missing_final_metrics", path, "Final-test metrics are missing")
        return {}
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        metric = value.get(name)
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            _issue(issues, "invalid_final_metrics", path, f"Metric {name!r} must be numeric")
            return {}
        result[name] = float(metric)
    return result


def _discover_attacker(
    path: Path,
    defender: DefenderRun,
    architecture: str,
    seed: int,
    issues: list[AnalysisIssue],
) -> AttackerRun:
    metadata = _read_json(path / "metadata.json", issues) or {}
    status = _read_json(path / "status.json", issues) or {}
    parent = _read_json(path / "parent.json", issues) or {}
    history = _read_jsonl(path / "history.jsonl", issues)
    metrics_path = defender.path / "test_results.json"
    final = _metrics(defender.final_metrics.get(path.name), metrics_path, issues)
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    valid = bool(metadata and status and parent and history and final)

    if metadata and metadata.get("stage") != "attack":
        _issue(issues, "stage_mismatch", path / "metadata.json", "Expected attack-stage metadata")
        valid = False
    if config != defender.config:
        _issue(issues, "config_mismatch", path / "metadata.json", "Attacker config differs from defender config")
        valid = False
    if status:
        if status.get("status") != "complete":
            _issue(issues, "incomplete_attacker", path / "status.json", "Attacker status is not complete")
            valid = False
        expected = defender.config.get("attacker_updates")
        if status.get("updates") != expected:
            _issue(issues, "budget_mismatch", path / "status.json", "Attacker did not complete the configured update budget")
            valid = False
        if not isinstance(status.get("seconds"), (int, float)) or not isinstance(status.get("parameters"), int):
            _issue(issues, "invalid_status", path / "status.json", "Runtime and parameter count are required")
            valid = False
    expected_channels = defender.config.get(f"{architecture}_channels")
    if parent and parent.get("channels") != expected_channels:
        _issue(issues, "channel_mismatch", path / "parent.json", "Saved channels do not match the architecture config")
        valid = False
    expected_hash = defender.final_metrics.get("_defender_hash")
    if parent and expected_hash is not None and parent.get("checkpoint_sha256") != expected_hash:
        _issue(issues, "checkpoint_mismatch", path / "parent.json", "Attacker used a different defender checkpoint")
        valid = False

    return AttackerRun(
        path=path,
        defender_seed=defender.seed,
        attacker_seed=seed,
        architecture=architecture,
        config=config,
        status=status,
        parent=parent,
        validation_history=history,
        final_metrics=final,
        valid=valid,
    )


def discover_runs(root: Path) -> AnalysisInventory:
    """Discover saved defenders and attackers without modifying ``root``."""

    root = Path(root)
    issues: list[AnalysisIssue] = []
    defenders: list[DefenderRun] = []
    for path in sorted(root.glob("defender_*")):
        match = DEFENDER_PATTERN.fullmatch(path.name)
        if not path.is_dir() or match is None:
            continue
        seed = int(match.group(1))
        metadata = _read_json(path / "metadata.json", issues) or {}
        status = _read_json(path / "status.json", issues) or {}
        history = _read_jsonl(path / "history.jsonl", issues)
        test_result = _read_json(path / "test_results.json", issues) or {}
        config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
        raw_metrics = test_result.get("metrics") if isinstance(test_result.get("metrics"), dict) else {}
        final_metrics = dict(raw_metrics)
        final_metrics["_defender_hash"] = test_result.get("defender_hash")
        valid = bool(metadata and status and history and raw_metrics)
        if metadata and (metadata.get("stage") != "defender" or metadata.get("seed") != seed):
            _issue(issues, "defender_metadata_mismatch", path / "metadata.json", "Defender identity or stage is inconsistent")
            valid = False
        if status:
            if status.get("status") != "complete":
                _issue(issues, "incomplete_defender", path / "status.json", "Defender status is not complete")
                valid = False
            if status.get("updates") != config.get("defender_updates"):
                _issue(issues, "defender_budget_mismatch", path / "status.json", "Defender budget is incomplete")
                valid = False
        defender = DefenderRun(path, seed, config, status, history, final_metrics, valid=valid)
        for attack_path in sorted(path.glob("attack_*")):
            attack_match = ATTACK_PATTERN.fullmatch(attack_path.name)
            if attack_path.is_dir() and attack_match:
                architecture, attacker_seed = attack_match.groups()
                defender.attackers.append(
                    _discover_attacker(attack_path, defender, architecture, int(attacker_seed), issues)
                )

        expected_seeds = config.get("attacker_seeds", [])
        if isinstance(expected_seeds, list):
            existing = {(run.architecture, run.attacker_seed) for run in defender.attackers}
            for attacker_seed in expected_seeds:
                for architecture in ("baseline", "wider"):
                    if (architecture, attacker_seed) not in existing:
                        _issue(
                            issues,
                            "missing_attacker",
                            path / f"attack_{architecture}_{attacker_seed}",
                            "Configured attacker run is missing",
                        )

        by_seed: dict[int, dict[str, AttackerRun]] = {}
        for run in defender.attackers:
            by_seed.setdefault(run.attacker_seed, {})[run.architecture] = run
        for runs in by_seed.values():
            baseline, wider = runs.get("baseline"), runs.get("wider")
            if baseline is None or wider is None:
                continue
            if baseline.parent.get("checkpoint_sha256") != wider.parent.get("checkpoint_sha256"):
                _issue(issues, "checkpoint_mismatch", wider.path / "parent.json", "Paired attackers used different checkpoints")
                baseline.valid = wider.valid = False
            if baseline.parent.get("data_stream_seed") != wider.parent.get("data_stream_seed"):
                _issue(issues, "stream_mismatch", wider.path / "parent.json", "Paired attackers used different training streams")
                baseline.valid = wider.valid = False
            if baseline.status.get("updates") != wider.status.get("updates"):
                _issue(issues, "budget_mismatch", wider.path / "status.json", "Paired attackers have unequal update budgets")
                baseline.valid = wider.valid = False
        defenders.append(defender)

    if not defenders:
        _issue(issues, "no_defenders", root, "No defender_<seed> directories were found")
    return AnalysisInventory(root=root, defenders=defenders, issues=issues)


def pair_attackers(inventory: AnalysisInventory) -> list[ComparisonPair]:
    """Return only complete, provenance-matched baseline/wider pairs."""

    pairs: list[ComparisonPair] = []
    for defender in inventory.defenders:
        if not defender.valid:
            continue
        grouped: dict[int, dict[str, AttackerRun]] = {}
        for attacker in defender.attackers:
            grouped.setdefault(attacker.attacker_seed, {})[attacker.architecture] = attacker
        for attacker_seed, runs in sorted(grouped.items()):
            baseline, wider = runs.get("baseline"), runs.get("wider")
            if baseline is not None and wider is not None and baseline.valid and wider.valid:
                pairs.append(ComparisonPair(defender.seed, attacker_seed, baseline, wider))
    return pairs


def summarize_pairs(pairs: list[ComparisonPair]) -> dict[str, Any]:
    """Aggregate paired differences within defenders, then across defenders."""

    grouped: dict[int, list[ComparisonPair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.defender_seed, []).append(pair)
    defender_aggregates = []
    for defender_seed, defender_pairs in sorted(grouped.items()):
        means = {
            name: fmean(float(pair.differences[name]) for pair in defender_pairs)
            for name in DIFFERENCE_NAMES
        }
        defender_aggregates.append(
            {"defender_seed": defender_seed, "pair_count": len(defender_pairs), "mean_differences": means}
        )
    overall: dict[str, dict[str, float]] = {}
    for name in DIFFERENCE_NAMES:
        values = [float(row["mean_differences"][name]) for row in defender_aggregates]
        if values:
            overall[name] = {"mean": fmean(values), "min": min(values), "max": max(values)}
    defender_count = len(defender_aggregates)
    caveat = (
        f"Descriptive comparison across {defender_count} defender seed(s); "
        "do not claim statistical significance, and do not treat attacker initializations as independent defenders."
    )
    return {
        "per_seed": [pair.as_dict() for pair in pairs],
        "defender_aggregates": defender_aggregates,
        "overall": overall,
        "caveat": caveat,
    }
