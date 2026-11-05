# Analysis and Security Evaluation Design

## Intent

Extend the existing adversarial neural cryptography study with reproducible, read-only analysis and diagnostic security tooling. The work is for an undergraduate cryptography group that needs transparent evidence, honest limitations, and reusable commands without changing the completed pilot or its reserved final-test results.

## Goals

- Discover defender and attacker runs from saved artifacts and validate their provenance, status, budgets, histories, and final metrics.
- Pair baseline and wider Eve runs by defender and attacker seed, then report individual outcomes, defender-level aggregates, mean paired differences, and observed ranges.
- Keep validation histories distinct from final-test metrics and generate tables plus publication-quality figures under a separate output directory.
- Quantify ciphertext sensitivity, Eve reconstruction, per-bit leakage, key variation, random-guess behavior, and repeated-input determinism on new named diagnostic streams.
- Validate experiment configurations before expensive work starts and preserve existing checkpoint and result formats.
- Test malformed/incomplete artifacts, pairing, metric math, deterministic diagnostic data, input immutability, and final-test protection with small fixtures.
- Document what was measured, what could not be measured, full-study feasibility estimates, and exact reproduction commands.

## Non-goals and anti-goals

- Do not retrain models or rerun final evaluation to populate analysis.
- Do not modify, delete, or overwrite saved pilot metrics or checkpoints.
- Do not claim semantic security, computational security, confidentiality, or statistical significance.
- Do not treat attacker initializations sharing one defender as independent defender replications.
- Do not launch the proposed full experiment or tune from the exposed pilot final test.
- Do not substantially refactor the working training architecture.

## Constraints

- Python and the repository's existing PyTorch, NumPy, Matplotlib, and pytest dependencies only.
- Existing JSON, JSONL, status, metadata, parent, and checkpoint formats remain readable without migration.
- Input run directories are read-only from the analysis tools' perspective; generated artifacts go to a caller-selected output directory.
- Security diagnostics derive samples from new stream labels and never use `final_test`.
- Archived report data contain metrics and histories but no `.pt` checkpoints, so real security diagnostics must report that limitation until a checkpoint-bearing run is supplied.

## Approaches considered

### 1. Extend `anc.run` with analysis and security commands

This would reuse its parser but would couple read-only reporting to training/final evaluation and make accidental scientific-state changes easier. It would also grow an already multi-responsibility module.

### 2. Separate analysis and security modules with independent CLIs (chosen)

`anc.analysis` owns artifact discovery, validation, pairing, aggregation, tables, and plots. `anc.security` owns diagnostic streams, sensitivity calculations, checkpoint-backed evaluation, and diagnostic plots. A small `anc.config` module centralizes preflight validation used by the workflow. This preserves training behavior while giving each responsibility a narrow, testable boundary.

### 3. A notebook-only analysis

A notebook is convenient for exploration but difficult to validate automatically, easy to run out of order, and weak at proving read-only behavior. It can be added later as a presentation layer over the reusable modules.

## Architecture

`anc.analysis` reads defender directories and returns explicit records plus structured issues rather than silently dropping invalid runs. Pairing requires matching defender seed, attacker seed, configuration, checkpoint identity, data stream, completion state, and update budget. Final-test comparisons come only from `test_results.json`; validation curves come only from `history.jsonl`. The CLI writes JSON, CSV/Markdown tables, and PNG figures without touching its input root.

`anc.security` exposes pure tensor metric functions and a checkpoint-backed evaluator. Diagnostic messages and keys use SHA256-derived stream labels `security_diagnostic`, `security_key_alternative`, and `security_random_guess`. Sensitivity flips one plaintext or key bit at a time and measures continuous ciphertext displacement and sign changes. Reconstruction is reported for each available Eve and an independent random control, including per-bit error. Determinism repeats identical inputs. The report explicitly describes these as empirical neural-network diagnostics, not proofs of cryptographic security.

`anc.config` checks schema, numeric ranges, seed uniqueness, channel shapes, time/update budgets, and disjoint defender/attacker seeds. Existing workflow entry points call it before creating output directories or training.

## Outputs

- `analysis_summary.json`: machine-readable inventory, issues, pairs, aggregates, and caveats.
- `comparison.csv` and `comparison.md`: per-seed final-test metrics, runtimes, parameters, and wider-minus-baseline differences.
- `defender_aggregates.csv`: defender-level paired summaries and overall observed ranges.
- `final_test_comparison.png`, `paired_differences.png`, and `validation_curves.png` when valid inputs exist.
- `security_report.json` and diagnostic PNG figures for checkpoint-bearing runs; a structured unavailable report when checkpoints are absent.
- `CONTRIBUTION_REPORT.md` and `RESULTS_ANALYSIS.md` for human readers.

## Acceptance criteria

- Synthetic tests prove metric calculations, pairing, validation, missing/malformed handling, diagnostic reproducibility, tensor shapes/dtypes, sensitivity math, read-only inputs, and final-test preservation.
- The existing pilot report directory can be analyzed without retraining or final evaluation.
- Security analysis of the archived pilot clearly reports missing checkpoints and does not fabricate results.
- The complete test suite passes in the project environment.
- Documentation names actual commands, actual pilot results, estimates as estimates, and remaining limitations.
