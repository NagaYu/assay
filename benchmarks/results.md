| Scenario | Assay detect (req) | BB 10% interleaved | BB dedicated | Distribution BB | AUC |
|---|---|---|---|---|---|
| Full substitution | 1 | 30 | 2 | 5 | 1.000 |
| Quantization | 1 | 19 | 2 | 10 | 1.000 |
| Randomized p=0.9 | 11 | **not detected** | **not detected** | **not detected** | 1.000 |
| Audit evasion | 1 | **not detected** | **not detected** | 5 | 1.000 |

- Honest operation: sequential false alarm = no, per-request FPR(z>3.09) = 0.000
- Verifier cost ratio: layer-token 0.63 / wall 0.72 (full recompute = 1.0)
- Coverage: Assay 100% / BB 10%
- Config: pair=gpt2, n=80/scenario, trials=3, alpha=0.01 (CPU)
