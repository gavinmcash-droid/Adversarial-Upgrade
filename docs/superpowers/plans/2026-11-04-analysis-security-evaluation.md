# Analysis and Security Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only experiment comparison, checkpoint-backed security diagnostics, preflight configuration validation, tests, generated pilot analysis, and academically cautious reports.

**Architecture:** Keep training unchanged except for a small preflight validator. Implement artifact analysis and security diagnostics as separate modules with independent CLIs, pure computation functions, structured issue reporting, and caller-selected output directories.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Matplotlib, pytest, JSON/JSONL, CSV, Markdown

**Spec:** `docs/superpowers/specs/2026-11-04-analysis-security-design.md`

## Global Constraints

- Never modify or rerun existing final-test results.
- Never overwrite an existing analysis output directory.
- Use only existing project dependencies.
- Keep validation histories separate from final-test metrics.
- Use newly named diagnostic RNG streams, not `final_test`.
- Treat missing checkpoints as a reported limitation, never as permission to fabricate results.
- Preserve checkpoint formats and the existing scientific methodology.

## Review Focus

- A malformed JSON/JSONL file must become a path-specific issue, not silently disappear or crash an unrelated defender.
- An attacker missing its matching architecture/seed, using another checkpoint/data stream/config, or completing a different budget must not form a valid pair.
- A single-defender result must display descriptive ranges and a no-significance caveat.
- Diagnostic evaluation must be deterministic for a seed while remaining disjoint from final-test and training streams.
- Analysis/security commands must leave every byte and timestamp in the input tree unchanged.

---

### Task 1: Experiment artifact discovery and pairing

**Files:**
- Create: `anc/analysis.py`
- Create: `tests/test_analysis.py`

**Interfaces:**
- Produces: `discover_runs(root: Path) -> AnalysisInventory`, `pair_attackers(inventory: AnalysisInventory) -> list[ComparisonPair]`, and `summarize_pairs(pairs) -> dict`.

- [ ] Write synthetic fixture tests for complete pairing, literal wider-minus-baseline metric/runtime/parameter differences, defender aggregation, and single-seed caveats.
- [ ] Run `python -m pytest tests/test_analysis.py -q` and verify failure because `anc.analysis` is missing.
- [ ] Implement typed records, JSON/JSONL loading, architecture/seed parsing, validity checks, pairing, and aggregation.
- [ ] Add tests for missing/malformed results, incomplete status, mismatched config/checkpoint/data stream/budget, and preservation of validation/final-test separation.
- [ ] Run the analysis tests and then the complete suite.

### Task 2: Tables, plots, and read-only analysis CLI

**Files:**
- Modify: `anc/analysis.py`
- Modify: `tests/test_analysis.py`

**Interfaces:**
- Produces: `write_analysis(root: Path, output: Path) -> dict` and `python -m anc.analysis --root ... --output ...`.

- [ ] Write failing tests for CSV/Markdown/JSON outputs, non-overwrite behavior, plot creation, and byte-for-byte unchanged inputs.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement atomic output-directory creation, tables, final-test figures, paired-difference figures, validation-only curves, and CLI messages.
- [ ] Run focused and complete tests.

### Task 3: Pure security diagnostic metrics

**Files:**
- Create: `anc/security.py`
- Create: `tests/test_security.py`

**Interfaces:**
- Produces: `diagnostic_sample(seed: int, size: int)`, `reconstruction_diagnostics(prediction, plaintext)`, `ciphertext_sensitivity(encrypt, plaintext, key, flip: str)`, and `determinism_diagnostic(encrypt, plaintext, key)`.

- [ ] Write failing hand-calculated tests for bitwise reconstruction, per-bit errors, random guessing, plaintext/key flip sensitivity, deterministic outputs, dtype, and shape.
- [ ] Run focused tests and verify failure because `anc.security` is missing.
- [ ] Implement the pure functions with named diagnostic streams and no final-test access.
- [ ] Run focused and complete tests.

### Task 4: Checkpoint-backed security report and missing-checkpoint handling

**Files:**
- Modify: `anc/security.py`
- Modify: `tests/test_security.py`

**Interfaces:**
- Produces: `evaluate_security(defender: Path, output: Path, samples: int) -> dict` and `python -m anc.security --defender ... --output ...`.

- [ ] Write failing integration tests using tiny saved real models and an archived-style run without checkpoints.
- [ ] Verify failures, then implement trusted local checkpoint loading, reconstruction/random controls, bit-flip/key diagnostics, JSON output, plots, and structured unavailable reports.
- [ ] Assert diagnostic commands never create or change `test_results.json`, then run focused and complete tests.

### Task 5: Configuration preflight validation

**Files:**
- Create: `anc/config.py`
- Modify: `anc/run.py`
- Modify: `study.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `validate_config(config: Mapping[str, object]) -> dict` returning a normalized copy or raising `ConfigError` with actionable messages.

- [ ] Write failing tests for required keys, positive integer/range constraints, unique/disjoint seeds, four positive channel widths ending in one output channel, and a valid existing config.
- [ ] Run focused tests and verify failure.
- [ ] Implement validation and call it before output directory creation in both CLIs.
- [ ] Run focused and complete tests.

### Task 6: Analyze archived pilot and document findings/readiness

**Files:**
- Create: `analysis/pilot/*`
- Create: `CONTRIBUTION_REPORT.md`
- Create: `RESULTS_ANALYSIS.md`

**Interfaces:**
- Consumes: Tasks 1-5 CLIs and `reports/pilot` artifacts.
- Produces: reproducible pilot summaries/figures and student-readable reports.

- [ ] Run analysis on `reports/pilot` without changing input files and record output hashes/commands.
- [ ] Run security evaluation against the archived pilot defender and record the verified missing-checkpoint limitation.
- [ ] Estimate full-study same-batch runtime from measured pilot phases; label the batch-4096 scaling as an unvalidated bound and do not launch training.
- [ ] Write the two requested reports with actual results, cryptographic caveats, changed files, test evidence, commands, and remaining work.
- [ ] Run the complete test suite and CLI smoke checks.
- [ ] Review `git diff --ignore-space-at-eol`, stage only intended files, and create logically separated commits if repository permissions allow.

## Self-review

- Spec coverage: every requested phase is represented; real security measurements remain conditional on checkpoints, and expensive training is explicitly excluded.
- Type consistency: analysis records feed pairing/reporting; security pure functions feed checkpoint evaluation; configuration validation returns a normal dictionary accepted by existing code.
- Failure coverage: the five review-focus cases are assigned to Tasks 1-5 tests.
- Proportion: tasks define observable interfaces and test cycles without transcribing implementation bodies.
