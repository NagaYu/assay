| Scenario | Assay detect (req) | BB 10% interleaved | BB dedicated | Distribution BB | AUC |
|---|---|---|---|---|---|
| Full substitution | 1 | 30 | 2 | 5 | 1.000 |
| Quantization | 1 | **not detected** | 181 | **not detected** | 1.000 |
| Randomized p=0.9 | 11 | **not detected** | 278 | **not detected** | 1.000 |
| Audit evasion | 1 | **not detected** | **not detected** | 5 | 1.000 |
| Receipt forgery (M' text, M receipt) | 3 | 31 | 2 | 5 | 0.214 |

- Honest control (3 trials, 240 requests): Assay sequential false alarms 0%, black box 0%, per-request FPR(z>3.09) = 0.000
- Verifier cost ratio: layer-token 0.74 / wall 0.93 (full recompute = 1.0)
- Coverage: Assay 100% / BB 10%
- Config: pair=gpt2, n=80/scenario, trials=3, alpha=0.01 (CPU)
