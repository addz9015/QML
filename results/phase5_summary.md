# Phase 5 - real IBM Quantum hardware

15 held-out test images, one Estimator job, depolarizing noise at p=0.1 as the matched simulated-noise comparison point.

| backend | RMSE of <Z> vs ideal | accuracy |
|---|---|---|
| ideal simulator | 0.000 | 1.000 |
| noisy simulator (p=0.1 depolarizing) | 0.3265 | 1.000 |
| real hardware | 0.1126 | 1.000 |

Real hardware tracked the ideal simulator more closely than the matched simulated-noise condition did - the idealized depolarizing channel used in Phases 3-4 is a pessimistic stand-in for this circuit on this backend.

## Gate-folding ZNE on real hardware

| condition | RMSE of <Z> vs ideal |
|---|---|
| fold=0 (raw hardware) | 0.1126 |
| fold=1 (3x noise) | 0.0820 |
| fold=2 (5x noise) | 0.0806 |
| ZNE, Richardson-extrapolated to zero | 0.1597 |

Folded (deliberately noisier) circuits came out closer to ideal than the unfolded one, the opposite of what gate-folding ZNE assumes - extrapolating that reversed trend overshoots and gives a worse estimate than not mitigating at all. See the README for the full discussion and a plausible explanation (default gate twirling / measurement-error mitigation on the real device).
