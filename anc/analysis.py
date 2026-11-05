"""Read-only discovery and paired analysis of saved ANC experiment artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
from statistics import fmean
import tempfile
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


def _comparison_rows(pairs: list[ComparisonPair]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        row: dict[str, Any] = {
            "defender_seed": pair.defender_seed,
            "attacker_seed": pair.attacker_seed,
        }
        for architecture, run in (("baseline", pair.baseline), ("wider", pair.wider)):
            for name in METRIC_NAMES:
                row[f"{architecture}_{name}"] = run.final_metrics[name]
            row[f"{architecture}_seconds"] = run.seconds
            row[f"{architecture}_parameters"] = run.parameters
        for name, value in pair.differences.items():
            row[f"difference_{name}"] = value
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headings = [
        "Defender",
        "Attacker seed",
        "Baseline BER",
        "Wider BER",
        "BER difference",
        "Baseline L1",
        "Wider L1",
        "Message-accuracy difference",
        "Runtime difference (s)",
        "Parameter difference",
    ]
    lines = [
        "# Baseline and wider Eve final-test comparison",
        "",
        "Differences are **Wider minus baseline**. Negative BER/L1 differences favor wider Eve; positive message-accuracy differences favor wider Eve.",
        "",
        "| " + " | ".join(headings) + " |",
        "|" + "|".join("---" for _ in headings) + "|",
    ]
    for row in rows:
        values = [
            str(row["defender_seed"]),
            str(row["attacker_seed"]),
            f'{row["baseline_bit_error_rate"]:.6f}',
            f'{row["wider_bit_error_rate"]:.6f}',
            f'{row["difference_bit_error_rate"]:.6f}',
            f'{row["baseline_normalized_l1"]:.6f}',
            f'{row["wider_normalized_l1"]:.6f}',
            f'{row["difference_message_accuracy"]:.6f}',
            f'{row["difference_seconds"]:.6f}',
            str(row["difference_parameters"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Validation histories are intentionally excluded from this final-test table.",
            "These descriptive comparisons do not establish statistical significance or cryptographic security.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _plot_final_metrics(pairs: list[ComparisonPair], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"D{pair.defender_seed}/A{pair.attacker_seed}" for pair in pairs]
    x = list(range(len(pairs)))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, metric in zip(axes, METRIC_NAMES):
        axis.bar([value - width / 2 for value in x], [pair.baseline.final_metrics[metric] for pair in pairs], width, label="Baseline Eve")
        axis.bar([value + width / 2 for value in x], [pair.wider.final_metrics[metric] for pair in pairs], width, label="Wider Eve")
        axis.set(title=metric.replace("_", " ").title(), xticks=x, xticklabels=labels, ylabel="Final-test value")
        axis.tick_params(axis="x", rotation=35)
    axes[0].legend(frameon=False)
    fig.suptitle("Final-test reconstruction (held-out data)")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_differences(pairs: list[ComparisonPair], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"D{pair.defender_seed}/A{pair.attacker_seed}" for pair in pairs]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, metric in zip(axes, METRIC_NAMES):
        values = [pair.differences[metric] for pair in pairs]
        colors = ["#2878B5" if value < 0 else "#D95319" for value in values]
        axis.bar(labels, values, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=metric.replace("_", " ").title(), ylabel="Wider minus baseline")
        axis.tick_params(axis="x", rotation=35)
    fig.suptitle("Paired final-test differences")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_validation(inventory: AnalysisInventory, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: list[tuple[str, list[dict[str, Any]], str]] = []
    for defender in inventory.defenders:
        series.append((f"defender_{defender.seed}/bob", defender.validation_history, "bob"))
        series.append((f"defender_{defender.seed}/training_eve", defender.validation_history, "training_eve"))
        for attacker in defender.attackers:
            series.append((f"D{defender.seed}/{attacker.path.name}", attacker.validation_history, "eve"))
    if not any(history for _, history, _ in series):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for label, history, model_key in series:
        usable = [row for row in history if isinstance(row.get(model_key), dict)]
        for axis, metric in zip(axes, METRIC_NAMES):
            axis.plot(
                [row.get("step") for row in usable],
                [row[model_key].get(metric) for row in usable],
                label=label,
                linewidth=1.4,
            )
            axis.set(title=metric.replace("_", " ").title(), xlabel="Saved validation step", ylabel="Validation value")
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Validation histories only (not final-test metrics)")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return True


def write_analysis(root: Path, output: Path) -> dict[str, Any]:
    """Write analysis artifacts to a new directory while leaving ``root`` untouched."""

    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Analysis output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        inventory = discover_runs(root)
        pairs = pair_attackers(inventory)
        summary = summarize_pairs(pairs)
        result = {
            "input_root": str(root.resolve()),
            "defender_count": len(inventory.defenders),
            "attacker_count": sum(len(defender.attackers) for defender in inventory.defenders),
            "pair_count": len(pairs),
            "issues": [issue.as_dict() for issue in inventory.issues],
            "summary": summary,
        }
        (temporary / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        rows = _comparison_rows(pairs)
        comparison_fields = list(rows[0]) if rows else ["defender_seed", "attacker_seed"]
        _write_csv(temporary / "comparison.csv", rows, comparison_fields)
        _write_markdown(temporary / "comparison.md", rows)
        aggregate_rows = [
            {
                "defender_seed": row["defender_seed"],
                "pair_count": row["pair_count"],
                **{f"mean_difference_{name}": value for name, value in row["mean_differences"].items()},
            }
            for row in summary["defender_aggregates"]
        ]
        aggregate_fields = list(aggregate_rows[0]) if aggregate_rows else ["defender_seed", "pair_count"]
        _write_csv(temporary / "defender_aggregates.csv", aggregate_rows, aggregate_fields)
        if pairs:
            _plot_final_metrics(pairs, temporary / "final_test_comparison.png")
            _plot_differences(pairs, temporary / "paired_differences.png")
        _plot_validation(inventory, temporary / "validation_curves.png")
        os.rename(temporary, output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Saved experiment root containing defender_<seed> directories")
    parser.add_argument("--output", required=True, type=Path, help="New directory for generated analysis artifacts")
    args = parser.parse_args()
    result = write_analysis(args.root, args.output)
    noun = "pair" if result["pair_count"] == 1 else "pairs"
    print(f"Wrote {result['pair_count']} valid baseline/wider {noun} to {args.output}")
    if result["issues"]:
        print(f"Recorded {len(result['issues'])} incomplete or invalid artifact issue(s).")


if __name__ == "__main__":
    main()
