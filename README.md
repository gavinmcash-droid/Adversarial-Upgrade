# Attacker capacity in adversarial neural cryptography

## Three-command workflow

From `/home/gavinator6454/Projects/Adversarial-Upgrade`, the streamlined interface is:

```bash
python3 study.py setup
python3 study.py run --profile pilot --root runs/pilot_easy_01
python3 study.py view --root runs/pilot_easy_01
```

`setup` installs locked dependencies only in `.venv`; skip it if already installed. `run` runs verification, trains defenders, trains fresh baseline/wider attackers, checks that all configured budgets completed, then evaluates once. `view` generates validation plots and displays saved-model examples and final metrics without retraining or reevaluating. Open the printed `validation.png` paths to see the plots; this is not a live dashboard.

For a tiny execution check, replace `pilot` with `smoke`. The `full` profile is explicit and expensive; inspect its budget before using it. Use a new root each time. A custom configuration is supported with `--config configs/my_study.json` instead of `--profile`. These wrappers retain the original stage commands below.

The wrapper copies the configuration to the run root, saves its own source, writes stage logs and `workflow_status.json`, and stops on errors or incomplete budgets. Failed runs remain intact; automatic resume is not implemented. Inspect `verification.log`, `defender.log`, `attack.log`, `evaluate.log` and any `workflow_error.txt` there. Scientific outputs remain in each `defender_<seed>/` directory. The `view` shortcut expects a wrapper-created run; use the detailed commands for older runs.

The extra original commands reflected both added research controls and an unnecessarily exposed internal workflow. This interface keeps the controls while reducing manual orchestration. It does not change architectures, objectives, data streams or checkpoints from existing experiments.


A small, controlled Python/PyTorch student project: train Alice and Bob with a training Eve, freeze the selected defender pair, then compare fresh baseline and wider Eves. No claim of cryptographic security, novelty, or improvement over the paper. Start with [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) and [reports/PILOT.md](reports/PILOT.md).

## Detected environment and installation

Tested on Linux x86_64, Intel Core 7 150U (12 logical CPUs), 15 GiB RAM, Python 3.12.3. No `nvidia-smi`, NVIDIA device, or available CUDA backend detected. CPU only, one PyTorch thread, deterministic algorithms requested. No global packages or GPU drivers installed. PyTorch 2.14.1+cpu, NumPy 2.5.4, Matplotlib 3.11.2, pytest 9.1.1; all exact installed versions in `requirements.lock.txt`.

The project `.venv` already exists. For a new checkout on the same system, copy/paste from this folder:

```bash
cd /home/gavinator6454/Projects/Adversarial-Upgrade
python3 -m venv --without-pip .venv
curl -L --fail https://bootstrap.pypa.io/get-pip.py -o /tmp/anc-get-pip.py
.venv/bin/python /tmp/anc-get-pip.py
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
```

The normal `python3 -m venv .venv` failed here because system ensurepip is absent; the above bootstrap installs pip only inside `.venv`. Internet is required for installation. The lock pins versions, not distribution hashes; exact bitwise reproducibility across platforms/versions is not promised.

## Verification and smoke workflow

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m anc.run defender --config configs/smoke.json --root runs/smoke_repeat
.venv/bin/python -m anc.run attack --config configs/smoke.json --root runs/smoke_repeat
.venv/bin/python -m anc.run evaluate --config configs/smoke.json --root runs/smoke_repeat
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.run plot --config configs/smoke.json --root runs/smoke_repeat
.venv/bin/python -m anc.run demo --config configs/smoke.json --root runs/smoke_repeat
```

Use a new root on each rerun; existing runs are never overwritten. Smoke checks execution, **not learning or confidentiality**. The terminal demo displays three example plaintexts and thresholded predictions from Bob and both saved Eves; examples are not aggregate evidence.

## Pilot commands

The initial pilot has already been run; read its results before spending more compute. To reproduce it deliberately with the same fixed settings:

```bash
.venv/bin/python -m anc.run defender --config configs/pilot.json --root runs/pilot_repeat
.venv/bin/python -m anc.run attack --config configs/pilot.json --root runs/pilot_repeat
.venv/bin/python -m anc.run evaluate --config configs/pilot.json --root runs/pilot_repeat
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.run plot --config configs/pilot.json --root runs/pilot_repeat
.venv/bin/python -m anc.run demo --config configs/pilot.json --root runs/pilot_repeat
```

Inspect the existing pilot without retraining or reevaluating:

```bash
cat reports/PILOT.md
cat runs/pilot/defender_101/test_results.json
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.run plot --config configs/pilot.json --root runs/pilot
.venv/bin/python -m anc.run demo --config configs/pilot.json --root runs/pilot
```

`configs/full.json` proposes three defender seeds and three attacker initializations per architecture per defender (18 fresh Eves). **The full experiment has not been launched.** Its budgets need a resource decision first; see the runtime estimate and limitations in the pilot report. Each command requires the same configuration saved with the defender. Do not change settings after inspecting the final test set. New exploratory tuning must reserve new test seeds.

## A student's guide to the files

- `anc/core.py`: a neural network is a trainable function. Alice transforms plaintext plus key into 16 real ciphertext values. Bob uses ciphertext plus key to reconstruct plaintext. Eve gets only ciphertext. Dense layers mix all inputs; convolutions apply learned local rules. The optimizer adjusts weights using error gradients. This file also defines sampling, objectives and metrics.
- `anc/run.py`: command-line training, validation selection, loading, final evaluation, plotting and demo. The two training stages are separate. Fresh Eves cannot update the frozen defenders.
- `configs/*.json`: seed lists, update counts, batches, time limits and independently configured Eve widths. An update is one optimizer step; a defender cycle includes one AB and two Eve steps.
- `tests/test_core.py`: shape, mixing, parameter-count, stream, metric, gradient, isolation, freeze and save/load checks. Prediction roundtrip tolerance is absolute 1e-7, relative 0.
- `EXPERIMENT_PLAN.md`: controls, source findings, architecture decisions, loss scaling, full-study analysis and limitations.
- `AI_AUDIT.md`: AI contributions, failures, verification evidence and human checks still needed.
- `requirements.lock.txt`: tested package versions.

Each run saves metadata (config, seeds, runtime framework/platform, reference revision, code hashes, package versions), an executable-source snapshot, validation history, status and checkpoints. Attacker provenance includes the frozen defender checkpoint hash and data seed. `best.pt` is selected only by validation; `last.pt` preserves the final state. `history.jsonl` is raw validation evidence; `test_results.json` is a separate one-time final test result. Status records actual updates and runtime, including time limits/failures. Checkpoints use trusted local files only. The folder has no usable local Git commit; source hashes/snapshots provide revision identity instead.

Generated checkpoints, plots and downloaded sources live under ignored `runs/` and `sources/`. Small result evidence is copied to `reports/` for review. Archive generated runs separately if sharing/reproducing the exact trained models. Failure directories are preserved; there is no automatic resume command.

## Reading metrics

All three are reported separately, on a 0–1 scale:

| Metric | Meaning | Better reconstruction |
|---|---|---|
| `normalized_l1` | Mean absolute continuous prediction error divided by 2 | Lower |
| `bit_error_rate` | Fraction of predicted signs wrong; zero maps to +1 | Lower |
| `message_accuracy` | Fraction of complete 16-bit messages correct | Higher |

A random independent bit guess has expected BER 50%, with sampling fluctuations. Continuous error is not discrete BER. Chance-level BER does not prove absence of information leakage. A wider model may perform better, worse, or similarly under a finite optimization budget.
