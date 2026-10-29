<!-- fontes: measures_replication -->

**Table 4a. Verdict per measure and seed (S sustains, P hyperparameter proxy, C contradicts, 0 null; seeds 0/1/2), the 3/3 replication rule, and the fixed-effect combination of the three seeds for partial ∣ val / partial ∣ val, 5 hp.**

| Scenario | Tr(H²) M=5 | λmax | Tr(H²) M=200 | Tr(H²) M=200, 2nd batch | sharpness ρ=0.1 | sharpness ρ=0.5 |
|:--|:--|:--|:--|:--|:--|:--|
| NSL-KDD | C → C; FE −0.43 / +0.08 | C → C; FE −0.49 / +0.10 | C → C; FE −0.49 / +0.09 | C → C; FE −0.52 / +0.13 | C → C; FE −0.21 / +0.04 | C → C; FE −0.28 / −0.01 |
| KronoDroid | 0 0 0 → 0 in 3/3; FE −0.07 / +0.02 | 0 0 0 → 0 in 3/3; FE −0.05 / +0.07 | 0 0 0 → 0 in 3/3; FE −0.05 / +0.07 | 0 0 0 → 0 in 3/3; FE −0.05 / +0.07 | S 0 C → not replicated; FE +0.02 / +0.08 | 0 0 C → not replicated; FE −0.02 / +0.05 |
| DREBIN cluster, fold 0 | P S P → not replicated; FE +0.52 / +0.10 | P S S → not replicated; FE +0.72 / +0.12 | P S P → not replicated; FE +0.71 / +0.10 | P P P → P in 3/3; FE +0.71 / +0.08 | 0 0 0 → 0 in 3/3; FE −0.01 / −0.02 | 0 0 0 → 0 in 3/3; FE +0.06 / −0.05 |
| DREBIN cluster, fold 1 | 0 P 0 → not replicated; FE +0.09 / −0.02 | 0 P 0 → not replicated; FE +0.13 / −0.01 | 0 P 0 → not replicated; FE +0.12 / −0.04 | 0 P 0 → not replicated; FE +0.13 / −0.01 | P 0 0 → not replicated; FE +0.12 / +0.01 | P 0 0 → not replicated; FE +0.16 / −0.01 |
| DREBIN cluster, fold 2 | 0 0 0 → 0 in 3/3; FE −0.11 / −0.00 | 0 0 0 → 0 in 3/3; FE −0.10 / +0.02 | 0 0 0 → 0 in 3/3; FE −0.09 / +0.04 | 0 0 0 → 0 in 3/3; FE −0.10 / +0.01 | C C 0 → not replicated; FE −0.32 / −0.06 | 0 C 0 → not replicated; FE −0.33 / −0.06 |
| DREBIN cluster, fold 3 | P S 0 → not replicated; FE +0.23 / +0.12 | P S 0 → not replicated; FE +0.26 / +0.20 | P S 0 → not replicated; FE +0.27 / +0.17 | P S 0 → not replicated; FE +0.26 / +0.19 | 0 C 0 → not replicated; FE −0.22 / +0.17 | 0 0 0 → 0 in 3/3; FE −0.11 / +0.18 |
| DREBIN cluster, fold 4 | 0 0 0 → 0 in 3/3; FE −0.00 / +0.04 | 0 0 0 → 0 in 3/3; FE −0.05 / −0.05 | 0 0 0 → 0 in 3/3; FE −0.05 / −0.04 | 0 0 0 → 0 in 3/3; FE −0.04 / −0.06 | 0 0 0 → 0 in 3/3; FE −0.03 / −0.03 | 0 0 0 → 0 in 3/3; FE −0.07 / −0.06 |
