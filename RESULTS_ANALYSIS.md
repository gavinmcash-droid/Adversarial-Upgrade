# Results analysis: attacker capacity and learned-ciphertext leakage

## 1. Research question

The study asks whether a wider, independently trained Eve can recover more plaintext information than the baseline Eve when both attack the same frozen Alice/Bob communication system. The comparison concerns reconstruction ability under a fixed finite training budget. It is not a proof that the learned scheme is secure or insecure in a formal cryptographic sense.

## 2. Experimental methodology

Alice maps a 16-bit plaintext and independent 16-bit key to 16 real ciphertext values. Bob receives the ciphertext and key; Eve receives only the ciphertext. After adversarial Alice/Bob training, the selected defenders are frozen. Fresh baseline and wider Eves then receive identical training examples, optimizer settings, update counts, validation examples, and final-test examples for a defender.

The baseline Eve has 599 parameters and the wider Eve has 717. Final results report normalized continuous L1, discrete bit error rate (BER), and exact 16-bit message accuracy. Lower L1 and BER mean better reconstruction; higher message accuracy means better reconstruction. Comparisons use wider minus baseline, paired by defender and attacker initialization seed. Validation histories and final-test measurements remain separate.

## 3. Pilot results

The completed pilot used defender seed 101 and attacker seed 201. It trained 3,000 defender cycles and 1,500 updates for each fresh Eve, with batch size 256. The held-out final test contained 4,096 messages.

| Model/control | Normalized L1 | BER | Whole-message accuracy |
|---|---:|---:|---:|
| Bob | 0.009626 | 0.0824% | 98.7549% |
| Baseline Eve | 0.227982 | 20.9198% | 1.9775% |
| Wider Eve | 0.197520 | 18.9041% | 3.1494% |
| Independent random guess | 0.502731 | 50.2731% | 0% |

Bob reconstructed almost all messages. Both trained Eves substantially outperformed independent random guessing, so the learned ciphertext leaked recoverable plaintext information to these attacker models.

## 4. New measured results

The new analyzer discovered one valid archived comparison and no missing, malformed, incomplete, or provenance-mismatched artifact in the pilot metrics archive. It reproduced the saved values without retraining or rerunning final evaluation. The archived input-tree SHA-256 remained unchanged before and after analysis.

A development-only throughput benchmark measured the proposed full-study batch size 4096 using new seeds 9001 and 9002. One hundred defender cycles took 9.0816 seconds; 100 baseline-Eve updates took 2.6859 seconds; and 100 wider-Eve updates took 3.7507 seconds. No final evaluation was run, so these measurements provide compute evidence only, not learning or security evidence.

Linear update-count extrapolation estimates about 51.58 timed CPU-hours for the three defenders and 18 attackers in `configs/full.json`. The short benchmark may not represent long-run throughput, and it does not predict whether the models will converge.

## 5. Baseline versus wider Eve

For the single pilot pair, wider minus baseline was:

- normalized L1: −0.030462;
- BER: −0.020157, or 2.0157 percentage points lower for wider Eve;
- whole-message accuracy: +0.011719, or 1.1719 percentage points higher;
- runtime: +0.4120 seconds; and
- parameter count: +118.

All three reconstruction metrics favored wider Eve in this run. However, there is only one learned defender and one attacker initialization. The observed range therefore collapses to the same single difference, and no statistical-significance claim is justified. The outcome may reflect capacity, optimization speed, or a seed-specific local optimum.

## 6. Information leakage and cryptographic interpretation

The final-test BERs of 20.92% and 18.90% are far below the 50% random-guess control. This is direct empirical evidence that these trained Eves can recover substantial bit-level information without the key. It does not quantify all possible leakage and does not show that a chance-level Eve would imply security.

The new security module can also measure plaintext-bit and key-bit ciphertext sensitivity, per-bit Eve errors, alternative-key ciphertext diversity, and repeated-input determinism on newly designated diagnostic streams. Those tests require trained checkpoints. The archived `reports/pilot/` directory does not include `best.pt`, so no checkpoint-backed sensitivity or determinism result is reported here. The generated report records this limitation rather than substituting fabricated data.

Neural reconstruction performance is not equivalent to semantic security, indistinguishability, computational security, or a formal confidentiality guarantee. No such property has been proved.

## 7. Limitations

- The pilot has one defender seed and one attacker initialization.
- The pilot final test is already exposed and must not guide tuning.
- Wider Eve has more parameters and a slower measured update, so equal updates are not equal computation.
- Finite optimization can make architecture comparisons reflect training dynamics rather than representational capacity.
- The archived metrics omit checkpoints needed for the new diagnostic security experiments.
- The full-batch benchmark is only 100 steps per phase and excludes final evaluation, plotting, and long-run system effects.
- Three defenders in the proposed full study would still provide only weak uncertainty estimates.
- Empirical attacks cannot replace formal cryptographic analysis.

## 8. Future improvements

The next defensible step is to approve resources, preserve the reserved full-study seeds, and run all configured defenders and matched attackers without dropping failures. The results should be aggregated within defender before comparing across defenders. Checkpoint-bearing runs should then receive the new diagnostic security evaluation using separate diagnostic streams. Future work could add stronger attacker architectures, larger initialization sets, mutual-information or distinguishability-oriented diagnostics, and independent scientific review of the loss scaling and architecture assumptions.
