# Initial pilot: completed, then stopped

All eight verification tests passed. The four-cycle smoke workflow completed training, fresh attacks, final evaluation, plot generation and saved-model demonstration. It established execution only. No full experiment or extra tuning run was launched.

## Actual pilot budget and outcome

One defender seed (101), one fresh attacker initialization (201) per architecture. Batch 256, learning rate .0008. Completed 3000 Alice/Bob updates plus 6000 training-Eve updates, then 1500 updates for **each** fresh attacker. Validation size 2048; independent shared final test size 4096 messages (65536 bits). Selected checkpoints: defender step 3000; both fresh Eves step 1500. Selection used validation continuous L1 only.

| Model/control | Continuous normalized L1 | Discrete BER | Whole-message accuracy |
|---|---:|---:|---:|
| Bob | 0.009626 | 0.0824% | 98.7549% |
| Fresh baseline Eve | 0.227982 | 20.9198% | 1.9775% |
| Fresh wider Eve | 0.197520 | 18.9041% | 3.1494% |
| Independent random guesses | 0.502731 | 50.2731% | 0% |

Raw values: `pilot/defender_101/test_results.json` (copied from the unchanged generated run). These metrics are distinct: the continuous L1 value is not measured bit error rate. Random guesses fluctuate around 50%; the expected complete-message success probability is only 1/65536.

There is evidence of baseline learning. Bob's validation BER fell from 50.0732% to 0.1099%; fresh baseline Eve's fell from 50.0732% to 21.1060%. Both also reconstruct substantially better than chance on the independent final test. This is a learned communication system with substantial demonstrated leakage to these attackers, not a successful confidentiality result.

Wider Eve's test BER is 2.0157 percentage points lower than baseline in this one run, with lower continuous error and higher whole-message recovery. This is a descriptive observation only. One defender and one initialization cannot establish a general capacity effect or statistical significance. Wider has 717 parameters vs baseline's 599; differing optimization speed could explain a finite-budget advantage. Do not claim improvement over the paper or cryptographic security.

## Measured runtime and estimates

| Phase | Actual updates | Measured seconds | Rate used for extrapolation |
|---|---:|---:|---:|
| Defender | 3000 cycles, each 1 AB + 2 Eve | 29.9160 | 9.972 ms/cycle |
| Baseline fresh Eve | 1500 | 3.1180 | 2.079 ms/update |
| Wider fresh Eve | 1500 | 3.5300 | 2.353 ms/update |

Sum of timed pilot phases: **36.5640 seconds**. Timers include model setup/loading, training, periodic validation and checkpoint writes inside each phase; exclude Python process startup, metadata generation, final test evaluation, plots, demonstration and environment installation. All phases completed before their caps (480 + 150 + 150 seconds). Smoke was also tiny; unattended training stayed far below the initial 15-minute ceiling.

**Estimates, not measurements:** at the *same batch size 256*, linear scaling predicts about 24.9 minutes for 150000 defender cycles; 8.7 minutes for 250000 baseline updates; 9.8 minutes for 250000 wider updates. Three defenders plus nine attackers per architecture would take about **4.0 hours** of timed phases. This assumes stable throughput and ignores startup, variable checkpoint/validation overhead and thermal effects. It is not a prediction of convergence.

`configs/full.json` uses batch 4096, which is 16 times larger; that configuration's runtime was **not measured** and cannot be reliably inferred from update count alone. Benchmark the chosen full batch under a separate agreed compute budget before committing to it. Full seeds [1101,1102,1103] and attacker seeds [1201,1202,1203] reserve streams distinct from the exposed pilot final test. No full run was started.

## Verification, failures and limits

- `tests.txt`: 8 passed. `tests-first.txt`: retained initial 1-failed/7-passed run. That failure exposed incorrect hand-counted parameter expectations; corrected to actual tensor counts. No model change was needed.
- Standard venv setup initially failed because ensurepip was missing; local pip bootstrap resolved it. No global installations. Initial sandbox DNS failure resolved through approved network access.
- No smoke/pilot training failures, nonfinite losses, timeouts or unequal attacker budgets occurred. All status files say complete.
- Save/load predictions were tested at absolute tolerance 1e-7. Optimizer isolation and frozen defenders were tested for both widths. The source's erroneous Eve update is not reproduced.
- Paper loss-scaling ambiguity, inferred SAME padding, PyTorch default initialization, smaller batches, validation selection policy and short budgets are documented departures/interpretations. These remain limitations, not hidden fixes.
- Three curves' update counters have different meanings: defender histories use adversarial cycles, fresh-Eve histories use single optimizer steps. They are not equal-compute comparisons. The saved plot explicitly labels this and contains validation data only.

## Exact next commands (inspection, no training)

From `/home/gavinator6454/Projects/Adversarial`:

```bash
.venv/bin/python -m pytest -q
cat runs/pilot/defender_101/test_results.json
MPLCONFIGDIR=/tmp/anc-mpl .venv/bin/python -m anc.run plot --config configs/pilot.json --root runs/pilot
.venv/bin/python -m anc.run demo --config configs/pilot.json --root runs/pilot
```

The README contains exact environment, smoke and pilot reproduction commands using new output directories. Do not rerun final evaluation for an existing run; it intentionally refuses overwrites. Before a longer study, read the plan, choose resources/budgets and retain independent held-out test streams. Leave this pilot's results unchanged.
