import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from anc.analysis import discover_runs, pair_attackers, summarize_pairs, write_analysis
from anc.core import stream_seed


METRICS = ("normalized_l1", "bit_error_rate", "message_accuracy")


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def _config(attacker_seeds=(11, 12)) -> dict:
    return {
        "defender_seeds": [1],
        "attacker_seeds": list(attacker_seeds),
        "defender_updates": 20,
        "attacker_updates": 10,
        "batch_size": 8,
        "validation_size": 16,
        "test_size": 32,
        "validate_every": 5,
        "learning_rate": 0.001,
        "threads": 1,
        "defender_seconds": 60,
        "attacker_seconds": 30,
        "baseline_channels": [2, 4, 4, 1],
        "wider_channels": [4, 8, 8, 1],
    }


def _make_study(root: Path, attacker_seeds=(11, 12)) -> Path:
    cfg = _config(attacker_seeds)
    defender_hash = "d" * 64
    defender = root / "defender_1"
    defender.mkdir(parents=True)
    _write(defender / "metadata.json", {"config": cfg, "seed": 1, "stage": "defender"})
    _write(defender / "status.json", {"status": "complete", "updates": 20, "seconds": 5.0})
    _write(
        defender / "history.jsonl",
        {
            "step": 20,
            "seconds": 5.0,
            "bob": {"normalized_l1": 0.01, "bit_error_rate": 0.02, "message_accuracy": 0.75},
            "training_eve": {"normalized_l1": 0.5, "bit_error_rate": 0.5, "message_accuracy": 0.0},
        },
    )
    final_metrics = {
        "bob": {"normalized_l1": 0.02, "bit_error_rate": 0.03, "message_accuracy": 0.70},
        "random_guess": {"normalized_l1": 0.5, "bit_error_rate": 0.5, "message_accuracy": 0.0},
    }
    values = {
        ("baseline", 11): (0.45, 0.40, 0.01, 10.0, 100),
        ("wider", 11): (0.35, 0.30, 0.03, 12.0, 120),
        ("baseline", 12): (0.25, 0.20, 0.05, 9.0, 100),
        ("wider", 12): (0.30, 0.25, 0.04, 11.0, 120),
    }
    for architecture in ("baseline", "wider"):
        for seed in attacker_seeds:
            l1, ber, accuracy, seconds, parameters = values[(architecture, seed)]
            name = f"attack_{architecture}_{seed}"
            attack = defender / name
            attack.mkdir()
            _write(attack / "metadata.json", {"config": cfg, "seed": seed, "stage": "attack"})
            _write(
                attack / "status.json",
                {"status": "complete", "updates": 10, "seconds": seconds, "parameters": parameters},
            )
            _write(
                attack / "parent.json",
                {
                    "checkpoint_sha256": defender_hash,
                    "channels": cfg[f"{architecture}_channels"],
                    "data_stream_seed": stream_seed(1, "attacker_train"),
                },
            )
            _write(
                attack / "history.jsonl",
                {
                    "step": 10,
                    "seconds": seconds,
                    "eve": {"normalized_l1": 0.99, "bit_error_rate": 0.99, "message_accuracy": 0.0},
                },
            )
            final_metrics[name] = {
                "normalized_l1": l1,
                "bit_error_rate": ber,
                "message_accuracy": accuracy,
            }
    _write(
        defender / "test_results.json",
        {
            "metrics": final_metrics,
            "examples": 32,
            "test_seed": stream_seed(1, "final_test"),
            "checkpoint_hashes": {
                f"attack_{architecture}_{seed}": ("a" if architecture == "baseline" else "b") * 64
                for architecture in ("baseline", "wider")
                for seed in attacker_seeds
            },
            "defender_hash": defender_hash,
        },
    )
    return defender


def test_pairs_complete_runs_and_computes_literal_differences(tmp_path):
    _make_study(tmp_path)

    inventory = discover_runs(tmp_path)
    pairs = pair_attackers(inventory)

    assert inventory.issues == []
    assert [(pair.defender_seed, pair.attacker_seed) for pair in pairs] == [(1, 11), (1, 12)]
    assert pairs[0].differences == {
        "normalized_l1": pytest.approx(-0.10),
        "bit_error_rate": pytest.approx(-0.10),
        "message_accuracy": pytest.approx(0.02),
        "seconds": pytest.approx(2.0),
        "parameters": 20,
    }
    assert pairs[0].baseline.final_metrics["bit_error_rate"] == 0.40
    assert pairs[0].baseline.validation_history[-1]["eve"]["bit_error_rate"] == 0.99


def test_aggregates_by_defender_and_warns_single_defender_is_descriptive(tmp_path):
    _make_study(tmp_path)

    summary = summarize_pairs(pair_attackers(discover_runs(tmp_path)))

    defender = summary["defender_aggregates"][0]
    assert defender["defender_seed"] == 1
    assert defender["pair_count"] == 2
    assert defender["mean_differences"]["bit_error_rate"] == pytest.approx(-0.025)
    assert summary["overall"]["bit_error_rate"] == {
        "mean": pytest.approx(-0.025),
        "min": pytest.approx(-0.025),
        "max": pytest.approx(-0.025),
    }
    assert "statistical significance" in summary["caveat"]


@pytest.mark.parametrize(
    ("relative_path", "replacement", "issue_code"),
    [
        ("defender_1/attack_wider_11/status.json", {"status": "failed", "updates": 10}, "incomplete_attacker"),
        ("defender_1/attack_wider_11/metadata.json", {"config": {}, "seed": 11, "stage": "attack"}, "config_mismatch"),
        (
            "defender_1/attack_wider_11/parent.json",
            {"checkpoint_sha256": "other", "channels": [4, 8, 8, 1], "data_stream_seed": 12345},
            "checkpoint_mismatch",
        ),
        (
            "defender_1/attack_wider_11/parent.json",
            {"checkpoint_sha256": "defender-hash", "channels": [4, 8, 8, 1], "data_stream_seed": 999},
            "stream_mismatch",
        ),
        ("defender_1/attack_wider_11/status.json", {"status": "complete", "updates": 9}, "budget_mismatch"),
    ],
)
def test_invalid_comparisons_are_reported_and_excluded(tmp_path, relative_path, replacement, issue_code):
    _make_study(tmp_path)
    _write(tmp_path / relative_path, replacement)

    inventory = discover_runs(tmp_path)
    pairs = pair_attackers(inventory)

    assert issue_code in {issue.code for issue in inventory.issues}
    assert [(pair.defender_seed, pair.attacker_seed) for pair in pairs] == [(1, 12)]


def test_missing_and_malformed_files_are_path_specific_issues(tmp_path):
    _make_study(tmp_path, attacker_seeds=(11,))
    (tmp_path / "defender_1/attack_wider_11/metadata.json").unlink()
    (tmp_path / "defender_1/attack_baseline_11/history.jsonl").write_text("not-json\n")

    inventory = discover_runs(tmp_path)

    by_code = {issue.code: issue.path for issue in inventory.issues}
    assert by_code["missing_file"].endswith("attack_wider_11/metadata.json")
    assert by_code["malformed_jsonl"].endswith("attack_baseline_11/history.jsonl")
    assert pair_attackers(inventory) == []


def test_missing_expected_architecture_is_reported(tmp_path):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    wider = defender / "attack_wider_11"
    for child in wider.iterdir():
        child.unlink()
    wider.rmdir()

    inventory = discover_runs(tmp_path)

    assert "missing_attacker" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


def test_empty_history_is_reported_instead_of_silently_excluded(tmp_path):
    _make_study(tmp_path, attacker_seeds=(11,))
    (tmp_path / "defender_1/attack_baseline_11/history.jsonl").write_text("")

    inventory = discover_runs(tmp_path)

    assert "empty_history" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


def test_attacker_metadata_seed_must_match_directory_seed(tmp_path):
    _make_study(tmp_path)
    metadata = json.loads((tmp_path / "defender_1/attack_wider_11/metadata.json").read_text())
    metadata["seed"] = 999
    _write(tmp_path / "defender_1/attack_wider_11/metadata.json", metadata)

    inventory = discover_runs(tmp_path)

    assert "attacker_metadata_mismatch" in {issue.code for issue in inventory.issues}
    assert [(pair.defender_seed, pair.attacker_seed) for pair in pair_attackers(inventory)] == [(1, 12)]


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -0.01, 1.01, 10**400])
def test_nonfinite_or_out_of_range_final_metric_is_rejected(tmp_path, metric):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    results_path = defender / "test_results.json"
    results = json.loads(results_path.read_text())
    results["metrics"]["attack_baseline_11"]["bit_error_rate"] = metric
    _write(results_path, results)

    inventory = discover_runs(tmp_path)

    assert "invalid_final_metrics" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


def test_structurally_invalid_history_row_is_rejected(tmp_path):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    (defender / "attack_baseline_11/history.jsonl").write_text("{}\n")

    inventory = discover_runs(tmp_path)

    assert "invalid_history" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


def test_unconfigured_attacker_seed_is_reported_and_excluded(tmp_path):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    results_path = defender / "test_results.json"
    results = json.loads(results_path.read_text())
    for architecture in ("baseline", "wider"):
        source = defender / f"attack_{architecture}_11"
        target = defender / f"attack_{architecture}_99"
        shutil.copytree(source, target)
        metadata_path = target / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["seed"] = 99
        _write(metadata_path, metadata)
        results["metrics"][target.name] = dict(results["metrics"][source.name])
        results["checkpoint_hashes"][target.name] = "9" * 64
    _write(results_path, results)

    inventory = discover_runs(tmp_path)

    assert "unconfigured_attacker" in {issue.code for issue in inventory.issues}
    assert [(pair.defender_seed, pair.attacker_seed) for pair in pair_attackers(inventory)] == [(1, 11)]


def test_attackers_must_use_expected_defender_training_stream(tmp_path):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    for architecture in ("baseline", "wider"):
        parent_path = defender / f"attack_{architecture}_11/parent.json"
        parent = json.loads(parent_path.read_text())
        parent["data_stream_seed"] = 999
        _write(parent_path, parent)

    inventory = discover_runs(tmp_path)

    assert "stream_mismatch" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("examples", 31), ("test_seed", 999)],
)
def test_final_test_size_and_seed_must_match_defender_config(tmp_path, field, value):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    results_path = defender / "test_results.json"
    results = json.loads(results_path.read_text())
    results[field] = value
    _write(results_path, results)

    inventory = discover_runs(tmp_path)

    assert "test_provenance_mismatch" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


def test_missing_config_and_result_sample_size_cannot_match_as_none(tmp_path):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    for metadata_path in [defender / "metadata.json", *defender.glob("attack_*/metadata.json")]:
        metadata = json.loads(metadata_path.read_text())
        metadata["config"].pop("test_size")
        _write(metadata_path, metadata)
    results_path = defender / "test_results.json"
    results = json.loads(results_path.read_text())
    results.pop("examples")
    _write(results_path, results)

    inventory = discover_runs(tmp_path)

    assert "invalid_configuration" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("defender_hash", None), ("defender_hash", "not-a-sha256")],
)
def test_defender_hash_is_required_and_well_formed(tmp_path, field, value):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    results_path = defender / "test_results.json"
    results = json.loads(results_path.read_text())
    if value is None:
        results.pop(field)
    else:
        results[field] = value
    _write(results_path, results)

    inventory = discover_runs(tmp_path)

    assert "invalid_checkpoint_hashes" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


def test_checkpoint_hash_keys_must_cover_exactly_the_saved_attack_metrics(tmp_path):
    defender = _make_study(tmp_path, attacker_seeds=(11,))
    results_path = defender / "test_results.json"
    results = json.loads(results_path.read_text())
    results["checkpoint_hashes"].pop("attack_wider_11")
    _write(results_path, results)

    inventory = discover_runs(tmp_path)

    assert "checkpoint_hash_mismatch" in {issue.code for issue in inventory.issues}
    assert pair_attackers(inventory) == []


def _snapshot_tree(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _snapshot_structure(root: Path) -> list[tuple[str, bool, bytes | None, int | None]]:
    return [
        (
            str(path.relative_to(root)),
            path.is_dir(),
            None if path.is_dir() else path.read_bytes(),
            None if path.is_dir() else path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    ]


def test_write_analysis_creates_tables_plots_and_keeps_input_read_only(tmp_path):
    study = tmp_path / "study"
    output = tmp_path / "analysis"
    _make_study(study)
    before = _snapshot_tree(study)

    result = write_analysis(study, output)

    assert _snapshot_tree(study) == before
    assert result["pair_count"] == 2
    assert result["issues"] == []
    expected = {
        "analysis_summary.json",
        "comparison.csv",
        "comparison.md",
        "defender_aggregates.csv",
        "final_test_comparison.png",
        "paired_differences.png",
        "validation_curves.png",
    }
    assert expected <= {path.name for path in output.iterdir()}
    saved = json.loads((output / "analysis_summary.json").read_text())
    assert saved["summary"]["per_seed"][0]["baseline"]["bit_error_rate"] == 0.4
    assert "validation" not in saved["summary"]["per_seed"][0]["baseline"]
    csv_text = (output / "comparison.csv").read_text()
    assert "difference_bit_error_rate" in csv_text
    assert "-0.1" in csv_text
    assert b"\r\n" not in (output / "comparison.csv").read_bytes()
    assert b"\r\n" not in (output / "defender_aggregates.csv").read_bytes()
    assert "Wider minus baseline" in (output / "comparison.md").read_text()
    for name in ("final_test_comparison.png", "paired_differences.png", "validation_curves.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_write_analysis_preserves_relative_input_label_in_saved_summary(tmp_path, monkeypatch):
    _make_study(tmp_path / "reports" / "pilot", attacker_seeds=(11,))
    monkeypatch.chdir(tmp_path)

    write_analysis(Path("reports/pilot"), Path("analysis/pilot"))

    saved = json.loads(Path("analysis/pilot/analysis_summary.json").read_text())
    assert saved["input_root"] == "reports/pilot"


def test_write_analysis_refuses_to_overwrite_output(tmp_path):
    study = tmp_path / "study"
    output = tmp_path / "analysis"
    _make_study(study)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError):
        write_analysis(study, output)

    assert marker.read_text() == "keep"


def test_write_analysis_rejects_output_nested_inside_input_before_mutation(tmp_path):
    study = tmp_path / "study"
    _make_study(study, attacker_seeds=(11,))
    output = study / "derived" / "analysis"
    before = _snapshot_structure(study)

    with pytest.raises(ValueError, match="inside the input"):
        write_analysis(study, output)

    assert _snapshot_structure(study) == before
    assert not output.parent.exists()


def test_write_analysis_excludes_rejected_history_rows_from_plots(tmp_path):
    study = tmp_path / "study"
    output = tmp_path / "analysis"
    defender = _make_study(study)
    _write(
        defender / "attack_baseline_11/history.jsonl",
        {
            "step": {"invalid": True},
            "seconds": 0.0,
            "eve": {
                "normalized_l1": {"invalid": True},
                "bit_error_rate": 0.5,
                "message_accuracy": 0.0,
            },
        },
    )

    result = write_analysis(study, output)

    assert result["pair_count"] == 1
    assert "invalid_history" in {issue["code"] for issue in result["issues"]}
    assert (output / "validation_curves.png").is_file()


def test_analysis_module_cli_reports_destination(tmp_path):
    study = tmp_path / "study"
    output = tmp_path / "analysis"
    _make_study(study, attacker_seeds=(11,))

    completed = subprocess.run(
        [sys.executable, "-m", "anc.analysis", "--root", str(study), "--output", str(output)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert str(output) in completed.stdout
    assert "1 valid baseline/wider pair" in completed.stdout
