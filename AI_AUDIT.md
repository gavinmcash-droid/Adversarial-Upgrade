# AI assistance audit

Date: 2026-10-26. This project was generated and executed with Codex assistance. The user specified the research question, architectures, controls, verification requirements and stopping point. No subagents were used. Human scientific review remains necessary.

## Adopted AI outputs

AI inspected the paper and reference source before implementing, selected CPU PyTorch based on the detected machine, wrote the model/training/evaluation/plot/demo code, configurations, tests and documentation, and ran tests, a smoke workflow and one bounded pilot. The project is an original small implementation of the described experiment, not a validated reproduction of the paper. All numerical findings in the pilot report come from saved executed results; no wider-Eve benefit is presumed.

Source identity: arXiv 1610.06918v1 PDF SHA256 `6dd591c37b93a5a9f64ba3ac5d1ec1efa55b8ec2e6cbfb48c36605385778d860`; reference commit `30efea2b4d170c862703ff16f0fd5ad4ab76ddc7`. Local downloaded copies are under `sources/`, ignored by Git. The reference's incorrect Eve update is at `neural_encryption.py:80`; dense input shape/global-mixing issue is in `net.py`. The obsolete reference TensorFlow dependencies were not installed.

## Issues found and actions taken

1. Sandboxed network lookup failed; source download and local environment package installation succeeded using approved network access. Neither requested source remained inaccessible.
2. System Python lacked pip/ensurepip. A standard venv attempt failed. Bootstrapped pip inside the local `.venv` only; no global Python packages or drivers were modified. Pip reported a non-writable global cache and disabled caching; this did not fail installation/training.
3. First verification had one failure and seven passes: AI's manually expected parameter counts were wrong. Actual tensor counts and independent layer arithmetic established 1111/599/717. Corrected tests and prose, not model widths. Original failure output is retained in `reports/tests-first.txt`; corrected run in `reports/tests.txt`.
4. Paper's raw-L1 definition vs N/2 chance target is ambiguous for −1/+1 representation. Chose half-L1 message error and per-example quadratic penalty, explicitly documented in the plan. SAME padding is a documented inference.
5. Full-study seed values are reserved separately (1101–1103, 1201–1203) to avoid reusing pilot final-test streams. No training hyperparameters were tuned from pilot test results; the pilot has one fixed configuration and no retries.

## Verification evidence

Eight automated tests passed before training the pilot. They cover global FC connectivity, shapes, widths/parameter counts, data representation, independent RNG streams, Eve's ciphertext-only API, optimizer isolation, finite gradients and changed intended weights, frozen defenders during both fresh attacks, prediction-preserving state-dict roundtrips (absolute tolerance 1e-7), hand-calculated metrics/objective, and a seeded random-guess sanity control.

The complete smoke workflow exercised defender training, two fresh attacks, validation checkpoint selection, final evaluation, plotting and loading for a terminal demonstration. Smoke predictions were near chance; this was execution evidence only. Smoke and pilot directories retain their original raw histories, configs, source hashes/snapshots, versions, selected/final checkpoints, status and timings. Small evidence copies are in `reports/`. See `reports/PILOT.md` for the measured pilot outcome and explicit limitations.

No local project Git commit was available (the mounted .git is an empty read-only directory); each run uses code hashes and a source snapshot for revision provenance. Reference repository commit is available and recorded. Generated run data are ignored but retained on disk; someone sharing this project must separately archive them.

## Remaining human checks

- Read sections 2.4–2.5 and independently approve the loss-scale interpretation, padding and initialization departures before treating this as a reproduction study.
- Review optimizer boundaries, RNG/data separation, validation selection and final-test discipline in code; automated tests are evidence, not a formal proof.
- Decide whether the pilot defender quality/confidentiality is suitable and choose realistic full budgets using resources available. Do not tune against the exposed final test; use the reserved full streams or new seeds.
- Run the planned replication set only after that decision, report all failures, and analyze paired differences at defender level. One pilot seed cannot answer the general capacity question.
- Check course/institution rules for disclosing AI assistance and cite the paper/reference appropriately. No novelty or cryptographic security claim is justified.

## Streamlined interface (2026-10-28)

Added `study.py` setup/run/view commands to orchestrate existing scientific stages. Updated README and testing protocol with the relocated project path. Added tests for incomplete-budget rejection, stopping before final evaluation, preservation of failed runs, and view-only behavior. Existing experiment code and saved results are unchanged. Verification evidence is recorded in `reports/streamlining-verification.txt`; a new smoke workflow is used to exercise the wrapper. No full study is launched by this change.
