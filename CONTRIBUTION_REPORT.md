# Contribution report: analysis and security evaluation

## Scope and repository state

The repository already implemented the Alice, Bob, baseline Eve, and wider Eve networks; deterministic named random streams; separated defender and fresh-attacker training; validation checkpoint selection; one-time final evaluation; metadata and status records; checkpoint provenance; overwrite safeguards; a smoke workflow; and a completed single-seed pilot. The existing automated suite had 11 tests in this checkout and passed before development began.

The audit confirmed that the planned cross-run analysis and ciphertext security diagnostics were unfinished. The archived pilot in `reports/pilot/` contains metrics, histories, metadata, and statuses but no `.pt` checkpoints. Therefore, its final-test comparison can be analyzed, but checkpoint-dependent security diagnostics cannot be measured from the archive.

All pre-existing tracked-file modifications were line-ending-only (`git diff --ignore-space-at-eol` was empty). Work was performed on `feature/analysis-security-evaluation`. The archived pilot input tree had composite SHA-256 `1a5918dd581c539e81ee023a8ab79033baaeae2cdecf4af0fc161fd63b326de1` before and after analysis, verifying read-only behavior for this execution.

## Implemented functionality

### Experiment analysis

`anc.analysis` now:

- discovers defender and baseline/wider attacker runs from saved directories;
- validates JSON/JSONL files, stages, statuses, configured budgets, channels, configurations, checkpoint identities, and shared attacker data streams;
- reports missing, malformed, incomplete, and invalid artifacts with path-specific issue codes;
- pairs attackers only by defender seed and attacker initialization seed;
- calculates wider-minus-baseline differences for normalized L1, BER, whole-message accuracy, runtime, and parameter count;
- averages attacker initializations within each defender before reporting means and observed ranges across defenders;
- keeps validation histories separate from final-test metrics;
- writes JSON, CSV, Markdown, and 300-DPI figures to a new output directory; and
- refuses to overwrite an existing analysis directory.

The generated pilot outputs are in `analysis/pilot/`. They include the final-test table, defender aggregate table, final-test comparison plot, paired-difference plot, and validation-only curves.

### Security diagnostics

`anc.security` now provides deterministic diagnostic-only data streams and quantitative functions for:

- plaintext recovery metrics without passing the secret key to Eve;
- independent random-guess controls;
- bitwise accuracy and per-bit error distributions;
- ciphertext displacement and sign changes after each plaintext-bit flip;
- ciphertext displacement and sign changes after each key-bit flip;
- ciphertext diversity for an independently sampled alternative key; and
- repeated-input determinism.

For checkpoint-bearing runs, the CLI loads the saved Alice/Bob and all complete Eve checkpoints, writes `security_report.json`, and creates reconstruction, per-bit, and sensitivity figures. It uses `security_diagnostic`, `security_key_alternative`, and `security_random_guess` streams; it never reads or writes `test_results.json`.

Running it against the archived pilot produced a structured `unavailable` report because `reports/pilot/defender_101/best.pt` is absent. No sensitivity or new reconstruction values were invented. These diagnostics describe empirical neural-network behavior and do not prove semantic security, computational security, or cryptographic confidentiality.

### Reproducibility safeguards

`anc.config` validates all required fields before an output directory is created. It rejects missing fields, non-positive budgets and sizes, non-finite numeric values, duplicate or overlapping defender/attacker seeds, and invalid four-layer channel specifications. Both `anc.run` and `study.py` now perform this preflight validation. Existing checkpoint and experiment formats are unchanged.

## Files added or modified

Added:

- `anc/analysis.py`
- `anc/security.py`
- `anc/config.py`
- `tests/test_analysis.py`
- `tests/test_security.py`
- `tests/test_config.py`
- `analysis/pilot/` generated analysis, figures, benchmark evidence, and security limitation report
- `CONTRIBUTION_REPORT.md`
- `RESULTS_ANALYSIS.md`
- `docs/superpowers/specs/2026-11-04-analysis-security-design.md`
- `docs/superpowers/plans/2026-11-04-analysis-security-evaluation.md`

Modified with focused changes:

- `anc/run.py`: validate configuration before setup/training.
- `study.py`: validate configuration before creating a run root.
- `tests/test_workflow.py`: use a complete valid synthetic workflow configuration.

No saved pilot metric, history, metadata, status, or final-test file was modified.

## Automated tests

Baseline: `11 passed in 6.66s`.

Final verification command:

```bash
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m pytest -q
```

The final suite contains 57 tests. New coverage includes:

- literal metric and paired-difference calculations;
- defender-level aggregation and the single-defender significance caveat;
- missing/malformed JSON and JSONL;
- incomplete, mismatched, and unequal-budget attackers;
- read-only analysis and output overwrite protection;
- deterministic diagnostic streams distinct from `final_test`;
- tensor shapes, dtypes, per-bit errors, random guessing, sensitivity matrices, and determinism;
- real checkpoint load/evaluate integration and missing-checkpoint reporting;
- preservation of reserved `test_results.json`; and
- configuration schema, ranges, seeds, channels, and preflight ordering.

This session used the same locked dependencies in `/tmp/anc-upgrade-venv` because unpacking the PyTorch wheel into the Windows-mounted repository `.venv` stalled. That environment-location change does not affect source or scientific behavior.

## Actual experimental findings

The reusable analyzer found one valid pair and no artifact issues in the archived pilot:

| Quantity | Baseline Eve | Wider Eve | Wider − baseline |
|---|---:|---:|---:|
| Normalized L1 | 0.227982 | 0.197520 | −0.030462 |
| BER | 20.9198% | 18.9041% | −2.0157 percentage points |
| Whole-message accuracy | 1.9775% | 3.1494% | +1.1719 percentage points |
| Runtime | 3.1180 s | 3.5300 s | +0.4120 s |
| Parameters | 599 | 717 | +118 |

These reproduce the already saved pilot findings; they are not a new independent experiment. One defender and one attacker initialization do not support a statistical-significance claim. Both Eves reconstructed far better than the independent random control (50.2731% BER), demonstrating leakage to these trained attackers. Chance-level behavior would not itself prove confidentiality.

No new checkpoint-backed cryptographic diagnostic result was obtained because the archived checkpoints are missing.

## Full-experiment readiness

At the pilot batch size 256, measured pilot rates imply about 4.0 hours of timed training phases for the proposed three-defender/eighteen-attacker update counts. That estimate does not apply directly to `configs/full.json`, whose batch size is 4096.

A separate development benchmark used seeds 9001/9002, batch 4096, 100 defender cycles, and 100 updates per attacker. It did not run final evaluation:

| Phase | Measured time | Rate |
|---|---:|---:|
| Defender | 9.0816 s / 100 cycles | 90.816 ms/cycle |
| Baseline Eve | 2.6859 s / 100 updates | 26.859 ms/update |
| Wider Eve | 3.7507 s / 100 updates | 37.507 ms/update |

Linear extrapolation to the configured update counts and all seeds gives approximately 11.35 defender-hours, 16.79 baseline-attacker-hours, and 23.44 wider-attacker-hours: **51.58 timed CPU-hours total** if throughput remains stable. This is a short-run throughput estimate, not a convergence prediction or guaranteed wall-clock duration. The configured per-phase time caps allow as much as 504 hours across 21 sequential phases. The full study is therefore technically runnable but not appropriate for an unattended coding session; resources and budgets should be approved before launch.

## Reproduction commands

Install and verify:

```bash
python3 study.py setup
.venv/bin/python -m pytest -q
```

Regenerate analysis in a new directory (outputs intentionally cannot be overwritten):

```bash
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.analysis --root reports/pilot --output analysis/pilot-repeat
```

Attempt archived-pilot diagnostics, which will accurately report the missing checkpoint:

```bash
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.security --defender reports/pilot/defender_101 --output analysis/pilot-security-repeat --samples 1024
```

For a checkpoint-bearing run, replace the defender path with `runs/<study>/defender_<seed>`.

Reproduce the bounded full-batch benchmark without final evaluation:

```bash
.venv/bin/python -m anc.run defender --config analysis/pilot/benchmark_config.json --root runs/fullbatch_benchmark_repeat
.venv/bin/python -m anc.run attack --config analysis/pilot/benchmark_config.json --root runs/fullbatch_benchmark_repeat
cat runs/fullbatch_benchmark_repeat/defender_9001/status.json
cat runs/fullbatch_benchmark_repeat/defender_9001/attack_*/status.json
```

Deliberately launch the full study only after resource approval:

```bash
python3 study.py run --profile full --root runs/full_approved_01
```

## Remaining work

- Restore or retain trained pilot checkpoints if real sensitivity and determinism measurements are required for that learned cryptosystem.
- Approve a feasible full-study compute budget and complete all three defender seeds plus all 18 matched attackers.
- Analyze those results at defender level and report failures/time limits without dropping seeds.
- Consider stronger attackers and formal leakage measures; empirical Eve reconstruction alone cannot establish cryptographic security.
- Independently review the loss scaling, SAME-padding interpretation, and scientific protocol before presenting this as a reproduction.

Commits already created before this report: `9c4fc21`, `566f257`, `7a97530`, `b2eb04c`, `e2b9727`, and `5903d82`. Use `git log --oneline main..feature/analysis-security-evaluation` for the final documentation commit as well.
