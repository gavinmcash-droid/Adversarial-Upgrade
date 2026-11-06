# Testing protocol: attacker capacity in adversarial neural cryptography

## Three-command workflow

From the repository root, the streamlined interface is:

```bash
python3 study.py setup
python3 study.py run --profile pilot --root runs/pilot_easy_01
python3 study.py view --root runs/pilot_easy_01
```

`setup` installs locked dependencies only in `.venv`; skip it if already installed. `run` runs verification, trains defenders, trains fresh baseline/wider attackers, checks that all configured budgets completed, then evaluates once. `view` generates validation plots and displays saved-model examples and final metrics without retraining or reevaluating. Open the printed `validation.png` paths to see the plots; this is not a live dashboard.

For a tiny execution check, replace `pilot` with `smoke`. The `full` profile is explicit and expensive; inspect its budget before using it. Use a new root each time. A custom configuration is supported with `--config configs/my_study.json` instead of `--profile`. These wrappers retain the original stage commands below.

The wrapper copies the configuration to the run root, saves its own source, writes stage logs and `workflow_status.json`, and stops on errors or incomplete budgets. Failed runs remain intact; automatic resume is not implemented. Inspect `verification.log`, `defender.log`, `attack.log`, `evaluate.log` and any `workflow_error.txt` there. Scientific outputs remain in each `defender_<seed>/` directory. The `view` shortcut expects a wrapper-created run; use the detailed commands for older runs.

The extra original commands reflected both added research controls and an unnecessarily exposed internal workflow. This interface keeps the controls while reducing manual orchestration. It does not change architectures, objectives, data streams or checkpoints from existing experiments.


This guide explains where the project is, what to run, and how to turn the saved results into a student study. Commands are for Bash on the computer where the project was created. Run each command separately and inspect its outcome before continuing. Writing this guide does not launch any training.

## 1. Understand the comparison

The question is: **does a wider, independently trained Eve recover more plaintext information than baseline Eve when both attack the same frozen Alice/Bob pair?**

Alice receives plaintext and a secret key and produces ciphertext. Bob receives ciphertext and the key and tries to recover the plaintext. Eve receives ciphertext only. A training Eve helps train Alice/Bob in stage 1. Stage 2 discards that attacker for the comparison and trains fresh baseline and wider Eves against the saved, frozen defenders.

Both fresh attackers must have the same examples, batch size, optimizer settings and update budget. Only their convolution channel widths differ. Baseline Eve has 599 parameters; wider Eve has 717. Their runtimes are recorded separately.

## 2. Open the project and check the environment

Open a terminal and run:

```bash
cd /path/to/Adversarial-Upgrade
pwd
.venv/bin/python --version
.venv/bin/python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())'
```

The project already has a local `.venv` environment. Every Python command below uses it explicitly, so activation is unnecessary. The previously tested setup was Python 3.12.3, PyTorch 2.14.1+cpu, on an Intel CPU. Training commands currently use CPU even if a later machine has a GPU. Do not install global packages or GPU drivers for these instructions.

If `.venv/bin/python` is missing, follow the environment installation commands in `README.md`. Exact tested dependency versions are in `requirements.lock.txt`.

## 3. Know where the files are

All paths below are relative to the repository root.

| Location | What it contains |
|---|---|
| `anc/core.py` | Networks, random data generation, objectives, metrics and optimizer steps |
| `anc/run.py` | Training, evaluation, plotting and demonstration commands |
| `configs/smoke.json` | Tiny execution check |
| `configs/pilot.json` | Bounded one-defender pilot |
| `configs/full.json` | Proposed three-defender, eighteen-attacker study |
| `tests/test_core.py` | Automated verification tests |
| `EXPERIMENT_PLAN.md` | Controls, architecture details, source review and limitations |
| `reports/PILOT.md` | Already completed pilot findings and runtime measurements |
| `AI_AUDIT.md` | AI assistance, verification evidence and remaining human checks |
| `runs/` | Generated checkpoints, metadata, metrics and plots |

Read `EXPERIMENT_PLAN.md` before choosing the study budget. In particular, it documents the paper's L1 scaling ambiguity and the implementation's explicit interpretation.

## 4. Run software verification

```bash
.venv/bin/python -m pytest -q
```

The previous result was **8 passed**. The tests check shapes, global input mixing, parameter counts, ±1 data, separate random streams, Eve's lack of a key input, optimizer isolation, frozen defenders, finite gradients, save/load predictions and hand-calculated metrics.

If a test fails, stop before training, keep the output, investigate and document the correction. Passing tests establishes software properties; it does not establish learning or confidentiality.

## 5. Run a complete smoke test

Choose a new output directory. This example uses `runs/smoke_protocol_01`. If it already exists, choose another name and use that name consistently in every command. Do not delete an old run to reuse its name.

### Train the defenders

```bash
.venv/bin/python -m anc.run defender --config configs/smoke.json --root runs/smoke_protocol_01
cat runs/smoke_protocol_01/defender_101/status.json
```

This runs four Alice/Bob updates and eight training-Eve updates. Check for `"status": "complete"` and `"updates": 4`.

### Train both fresh attackers

```bash
.venv/bin/python -m anc.run attack --config configs/smoke.json --root runs/smoke_protocol_01
cat runs/smoke_protocol_01/defender_101/attack_baseline_201/status.json
cat runs/smoke_protocol_01/defender_101/attack_wider_201/status.json
```

Each attacker should complete four updates. The command reloads the selected defenders and freezes them automatically. You do not need to move or rename checkpoints.

### Evaluate, plot and demonstrate

```bash
.venv/bin/python -m anc.run evaluate --config configs/smoke.json --root runs/smoke_protocol_01
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.run plot --config configs/smoke.json --root runs/smoke_protocol_01
.venv/bin/python -m anc.run demo --config configs/smoke.json --root runs/smoke_protocol_01
```

Look for `test_results.json` and `validation.png` inside `runs/smoke_protocol_01/defender_101/`. The demonstration prints three plaintexts and thresholded model predictions. These are illustrative examples, not aggregate evidence. Poor smoke predictions are expected; four updates are only an execution check.

## 6. Review or reproduce the pilot

A pilot already exists. Inspect it without retraining:

```bash
cat reports/PILOT.md
cat runs/pilot/defender_101/test_results.json
.venv/bin/python -m anc.run demo --config configs/pilot.json --root runs/pilot
```

Its timed training phases totaled approximately 36.6 seconds. Bob and the fresh baseline Eve learned to reconstruct information; wider Eve had lower error in that single run. This does not settle the study question.

If you need to reproduce the pilot, use a new root:

```bash
.venv/bin/python -m anc.run defender --config configs/pilot.json --root runs/pilot_protocol_01
cat runs/pilot_protocol_01/defender_101/status.json
.venv/bin/python -m anc.run attack --config configs/pilot.json --root runs/pilot_protocol_01
cat runs/pilot_protocol_01/defender_101/attack_baseline_201/status.json
cat runs/pilot_protocol_01/defender_101/attack_wider_201/status.json
.venv/bin/python -m anc.run evaluate --config configs/pilot.json --root runs/pilot_protocol_01
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.run plot --config configs/pilot.json --root runs/pilot_protocol_01
.venv/bin/python -m anc.run demo --config configs/pilot.json --root runs/pilot_protocol_01
```

Before running, note the limits: 3000 defender cycles, 1500 updates per fresh attacker, batch 256; time caps of 480 seconds for the defender and 150 seconds per attacker. Inspect each status before continuing. Repeating the same seeds is a reproducibility check, not an additional independent replication. Do not tune from the pilot's exposed final-test results.

## 7. Fix the full-study design before starting

The existing `configs/full.json` proposes:

| Setting | Value |
|---|---:|
| Defender seeds | 1101, 1102, 1103 |
| Fresh attacker initialization seeds | 1201, 1202, 1203 |
| Fresh attackers per defender | 3 baseline + 3 wider |
| Total fresh attackers | 18 |
| Defender cycles per seed | 150000 (each contains 1 AB + 2 Eve updates) |
| Updates per fresh attacker | 250000 |
| Batch size | 4096 |
| Validation/test messages per defender | 8192 / 32768 |
| Validation interval | 1000 updates/cycles |
| Learning rate | 0.0008 |
| Time cap per defender or individual attacker | 86400 seconds (24 hours) |

**These are proposed large budgets, not a short test.** The time caps apply separately to each model run, not to the whole command. The full batch size has not been benchmarked. The approximately four-hour estimate in the pilot report applies to these update counts at batch **256**, not batch 4096.

Choose a feasible budget and record it before starting. If you need a different configuration, copy `configs/full.json` to a clearly named file, edit it before any study training, and substitute that filename in every command below. Preserve the three-by-three replication structure if feasible. Keep the two attacker architectures' training settings matched. Do not edit the configuration between stages: commands require it to match the configuration saved with the defender.

Reserve full-study final test examples until the design is fixed. The full defender seeds are different from the pilot seeds, so they produce separate test streams. Any further exploratory tuning should use separate development seeds and validation data, not the reserved final test.

## 8. Execute the full study when ready

The following commands are provided for deliberate execution; the full study has not been run as part of this guide. Use a new root, here `runs/full_protocol_01`.

### Stage 1: train three defender pairs

```bash
.venv/bin/python -m anc.run defender --config configs/full.json --root runs/full_protocol_01
```

This command processes all three configured defender seeds sequentially. It can run for a long time with little terminal output. Each defender writes `history.jsonl` at validation intervals. From a second terminal, you can inspect progress, for example:

```bash
cd /path/to/Adversarial-Upgrade
tail -n 3 runs/full_protocol_01/defender_1101/history.jsonl
```

After the command finishes, inspect all statuses:

```bash
cat runs/full_protocol_01/defender_*/status.json
```

For an intact full budget, each must say `complete` with 150000 updates. A time-limited defender is a partial run, not a completed full-budget result. The program may still permit attacks against it; you must report that distinction.

### Stage 2: train eighteen fresh attackers

```bash
.venv/bin/python -m anc.run attack --config configs/full.json --root runs/full_protocol_01
cat runs/full_protocol_01/defender_*/attack_*/status.json
```

This runs the six attackers for each defender sequentially. Every fresh attacker uses that defender's same `best.pt`, identical training data stream and matched budget. Confirm all eighteen statuses are complete, with 250000 updates each. Keep runtime values even if one architecture is slower.

### Final test evaluation: run once after training

```bash
.venv/bin/python -m anc.run evaluate --config configs/full.json --root runs/full_protocol_01
```

This evaluates the validation-selected checkpoints on each defender's held-out test set. Baseline and wider Eves intentionally share the same test examples within a defender. The command rejects incomplete attackers, unequal update counts, and existing test-result files. Do not change the models or select different checkpoints after looking at these results.

### Generate plots and examples

```bash
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.run plot --config configs/full.json --root runs/full_protocol_01
.venv/bin/python -m anc.run demo --config configs/full.json --root runs/full_protocol_01
```

Each defender directory receives a `validation.png`. Plots read saved validation histories; they do not retrain. Defender plot steps are adversarial cycles, while fresh-attacker plot steps are individual optimizer updates. The axes do not represent equal computation between stages.

## 9. Find and preserve the evidence

For example, the first full-study defender would have:

```text
runs/full_protocol_01/defender_1101/
    metadata.json             Configuration, seeds, versions, code hashes
    source/                   Executable source snapshot
    status.json               Completion state, actual updates and timing
    history.jsonl             Validation metrics over training
    best.pt                   Validation-selected defender checkpoint
    last.pt                   Final defender checkpoint
    training_state.pt         Additional optimizer/RNG state
    test_results.json         Final Bob, attackers and random-control metrics
    validation.png            Plot generated from saved histories
    attack_baseline_1201/     First fresh baseline attacker
    attack_wider_1201/        First fresh wider attacker
    ...                       Remaining attacker initializations
```

Attacker directories contain their own metadata, status, history, selected/final checkpoints and `parent.json` identifying the defender checkpoint and data stream. A checkpoint is a saved set of learned weights. `best.pt` means best by the predeclared validation rule, not best on the final test.

`runs/` is excluded by `.gitignore`. Back up the entire study root separately if you need to share or retain trained models. New commands do not automatically copy results into `reports/`; the existing report copies were made for the initial pilot. Keep raw JSON/JSONL evidence alongside any new written report.

## 10. Analyze the results

Inspect the result files:

```bash
cat runs/full_protocol_01/defender_*/test_results.json
```

Each contains three separate reconstruction metrics:

| Metric | Meaning | Better recovery |
|---|---|---|
| `normalized_l1` | Mean absolute continuous prediction error divided by 2 | Lower |
| `bit_error_rate` | Fraction of incorrect predicted signs; zero predicts +1 | Lower |
| `message_accuracy` | Fraction of 16-bit messages recovered completely | Higher |

For the main comparison:

1. For each defender and attacker seed, subtract baseline BER from wider BER. Negative differences favor wider Eve. Multiply by 100 to express differences in percentage points.
2. Average those three differences within each defender.
3. Report all three defender-level averages, their mean and their range. Also show every individual run rather than only the best attacker.
4. Report continuous L1, whole-message accuracy, Bob's quality, actual budgets and each attacker's runtime separately.

Generate the predeclared paired summaries, tables and plots with the dedicated read-only analyzer. The output directory must be new and outside the saved study tree:

```bash
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.analysis \
  --root runs/full_protocol_01 --output analysis/full_protocol_01
```

The analyzer validates artifact structure, configured seeds and budgets, expected stream/test seeds, metric ranges, and recorded checkpoint-hash format, key coverage and internal consistency before forming pairs. It does not hash checkpoint files or establish their trusted origin; retain full run directories and verify checkpoint bytes separately when that guarantee matters. Inspect `analysis_summary.json` for every excluded or malformed artifact. Do not treat eighteen attackers as eighteen independent defender replications; attackers within a defender share the same learned communication scheme. Three defenders provide limited evidence and weak uncertainty estimates.

For each checkpoint-bearing defender, run the separate diagnostic command into another new directory:

```bash
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.security \
  --defender runs/full_protocol_01/defender_1101 \
  --output analysis/full_protocol_01-defender-1101-security --samples 1024
```

This command verifies saved attacker identity, configuration, budget, stream, channels and checkpoint step before including it. Its reconstruction and sensitivity measurements use diagnostic-only streams and do not establish semantic or computational security.

Random guesses should have BER near 50%, with finite-sample fluctuations. Chance-level BER does not prove that no information leaks. Continuous loss is not measured discrete bit error. Better Bob reconstruction does not by itself establish confidentiality. A result where wider Eve wins on one seed and loses on another should be reported as such.

## 11. Handle interruptions and failures honestly

If a command raises an error, stop and inspect its traceback and available `status.json` files. A handled training exception records `failed`; an update-boundary deadline records `time_limit`. An abrupt process kill may leave `running` even though the process has ended. Metadata/setup failures can occur before a detailed status is written.

There is no automatic resume command. Repeating a training command against an existing directory will fail rather than overwrite it. Keep interrupted runs, document what happened, and use a new root for a deliberate restart. Do not silently remove a failed seed or increase only one attacker's budget.

Final evaluation writes one result file per defender sequentially. If evaluation stops partway through, preserve any completed files; blindly rerunning the whole command will encounter the overwrite guard. Resolve the specific failure before planning a targeted recovery.

Finish the study report with the question, fixed configuration, sources and departures, verification evidence, every seed's outcomes, paired comparisons, runtimes, failures and limitations. Disclose AI assistance using `AI_AUDIT.md`. Describe the outcome as a controlled reconstruction experiment, not a novel cryptosystem or proof of security.
