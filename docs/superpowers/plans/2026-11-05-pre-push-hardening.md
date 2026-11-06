# Pre-Push Provenance Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the pre-push provenance, semantic-validation, read-only-output, portability, and documentation gaps found in the final review.

**Architecture:** Keep training and saved experiment formats unchanged. Harden `anc.analysis` with explicit artifact-schema and provenance checks, harden `anc.security` with the same fail-closed attacker identity checks while continuing to report limitations, and reject output paths nested inside input trees. Regenerate only derived analysis artifacts and documentation; never retrain or reevaluate the pilot.

**Tech Stack:** Python 3.12, PyTorch, Matplotlib, pytest, JSON/JSONL, Markdown

**Spec:** `docs/superpowers/specs/2026-11-04-analysis-security-design.md`

## Global Constraints

- Never modify or rerun existing final-test results.
- Never overwrite an existing analysis or security output directory.
- Keep validation histories separate from final-test metrics.
- Preserve checkpoint formats and training behavior.
- Archived checkpoints are absent, so validate available hash structure and key coverage without claiming cryptographic authenticity.
- Use the locked project dependencies and run the complete suite before committing.

## Review Focus

- Non-finite or out-of-range scientific metrics must invalidate the affected run.
- Empty or structurally invalid validation rows must invalidate the affected run.
- Foreign seeds, streams, test sizes, test seeds, and hash-key sets must not form valid comparisons.
- Security diagnostics must skip attackers whose identity, budget, stream, channels, or checkpoint step is inconsistent.
- Output paths inside input experiment trees must be rejected before any directory is created.

---

### Task 1: Strict analysis artifact validation

**Files:**
- Modify: `anc/analysis.py`
- Modify: `tests/test_analysis.py`

**Interfaces:**
- Consumes: existing defender, attacker, status, history, parent, and final-test JSON formats.
- Produces: path-specific issues and only provenance-complete `ComparisonPair` objects.

- [ ] Add failing tests for non-finite/out-of-range metrics, malformed history rows, unconfigured attackers, wrong expected stream seed, wrong final-test seed/size, missing or malformed hashes, and checkpoint-hash key mismatch.
- [ ] Run focused tests and confirm failures represent missing validation.
- [ ] Implement finite/range, history-schema, configured-seed, derived-stream/test-seed, result-size, and hash-schema/key validation.
- [ ] Run focused tests and the complete suite.

### Task 2: Strict security attacker provenance

**Files:**
- Modify: `anc/security.py`
- Modify: `tests/test_security.py`

**Interfaces:**
- Consumes: defender metadata/config/checkpoint plus attacker metadata/status/parent/checkpoint.
- Produces: diagnostics only for provenance-valid attackers and precise limitations for skipped attackers.

- [ ] Add failing parameterized integration tests for wrong metadata seed/stage/config, unconfigured attacker, wrong channels/stream, incomplete budget, and checkpoint-step mismatch.
- [ ] Run focused tests and confirm invalid attackers are currently included.
- [ ] Implement strict identity, configuration, budget, stream, channel, and checkpoint-step checks.
- [ ] Run focused tests and the complete suite.

### Task 3: Enforce read-only output boundaries

**Files:**
- Modify: `anc/analysis.py`
- Modify: `anc/security.py`
- Modify: `tests/test_analysis.py`
- Modify: `tests/test_security.py`

**Interfaces:**
- Produces: rejection of outputs equal to or nested beneath the input tree before filesystem mutation.

- [ ] Add failing tests using nested non-existent output paths and full tree snapshots.
- [ ] Run focused tests and confirm input trees are currently mutated.
- [ ] Add a shared path-containment guard and invoke it before creating output parents.
- [ ] Run focused tests and the complete suite.

### Task 4: Portable artifacts and release documentation

**Files:**
- Modify: `anc/analysis.py`
- Modify: `tests/test_analysis.py`
- Modify: `README.md`
- Modify: `TESTING_PROTOCOL.md`
- Modify: `CONTRIBUTION_REPORT.md`
- Regenerate: `analysis/pilot/analysis_summary.json`
- Create: `analysis/pilot/SHA256SUMS`
- Create: `.gitattributes`

**Interfaces:**
- Produces: portable `input_root` labels, discoverable analysis/security commands, current protocol text, generated-output hashes, and stable LF text policy.

- [ ] Add a failing behavior test proving a relative input root remains relative in the saved summary.
- [ ] Implement portable input-root serialization and run the focused test.
- [ ] Regenerate the pilot analysis into a temporary directory and copy only the derived summary after verifying the metrics are unchanged.
- [ ] Update README/protocol/report text and create a SHA-256 manifest for committed generated pilot outputs.
- [ ] Add `.gitattributes` with LF normalization for text and explicit binary treatment for PNG/PT files.
- [ ] Run the complete suite, CLI smoke checks, whitespace checks, secret/path scans, and final code review.
- [ ] Commit the complete hardening change as one reviewed commit.

## Self-Review

- Spec coverage: every pre-push finding maps to a task; no training or final-test behavior changes.
- Type consistency: analysis remains structured-issue based; security remains limitation based; both use the same path-boundary contract.
- Review focus: each listed failure mode has an owning regression test.
- Proportion: four independently testable tasks cover code, then derived artifacts/documentation.
