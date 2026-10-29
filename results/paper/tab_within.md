<!-- fontes: e0_within_config (-p200) -->

**Table 5. Same configuration, different training seeds (15 configurations × 20 seeds, Tr(H²) with 200 probes): within-configuration partial correlations, between-configuration contrast and the seed-to-seed spread of the shifted target versus validation.**

| Scenario | measure | ICC | partial within ∣ val | CI95 (seed bootstrap) | Fisher RE | CI95 | I² | between (n=15) | verdict |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| KronoDroid | Tr(H²) M=200 (primary) | 0.98 | −0.01 | [−0.12; +0.10] | −0.06 | [−0.20; +0.10] | 30% | +0.03 | null |
| KronoDroid | Tr(H²) M=200 (2nd estimate) | 0.98 | −0.02 | [−0.14; +0.09] | −0.05 | [−0.19; +0.10] | 23% | +0.03 | null |
| KronoDroid | λmax | 0.95 | −0.01 | [−0.12; +0.12] | −0.05 | [−0.18; +0.07] | 0% | +0.03 | null |
| KronoDroid | sharpness ρ=0.1 | 0.36 | +0.08 | [−0.04; +0.20] | +0.05 | [−0.08; +0.17] | 4% | −0.01 | null |
| KronoDroid | sharpness ρ=0.5 | 0.53 | +0.06 | [−0.06; +0.18] | +0.02 | [−0.10; +0.15] | 0% | +0.07 | null |
| KronoDroid | ρ within(val, target); sd within of target / val |  | +0.10 |  |  |  |  | 0.110 / 0.021 |  |
| DREBIN cluster, fold 0 | Tr(H²) M=200 (primary) | 0.96 | +0.12 | [+0.00; +0.24] | +0.12 | [−0.01; +0.24] | 0% | +0.75 | sustains |
| DREBIN cluster, fold 0 | Tr(H²) M=200 (2nd estimate) | 0.95 | +0.09 | [−0.03; +0.21] | +0.11 | [−0.02; +0.23] | 0% | +0.74 | null |
| DREBIN cluster, fold 0 | λmax | 0.98 | +0.16 | [+0.03; +0.28] | +0.19 | [+0.07; +0.31] | 0% | +0.75 | sustains |
| DREBIN cluster, fold 0 | sharpness ρ=0.1 | 0.84 | −0.11 | [−0.24; +0.01] | −0.13 | [−0.25; −0.01] | 0% | +0.21 | null |
| DREBIN cluster, fold 0 | sharpness ρ=0.5 | 0.77 | −0.17 | [−0.28; −0.05] | −0.18 | [−0.30; −0.06] | 0% | +0.28 | contradicts |
| DREBIN cluster, fold 0 | ρ within(val, target); sd within of target / val |  | −0.08 |  |  |  |  | 0.106 / 0.036 |  |
| DREBIN cluster, fold 3 | Tr(H²) M=200 (primary) | 0.97 | +0.06 | [−0.07; +0.17] | +0.04 | [−0.09; +0.16] | 0% | +0.54 | null |
| DREBIN cluster, fold 3 | Tr(H²) M=200 (2nd estimate) | 0.97 | +0.06 | [−0.06; +0.18] | +0.07 | [−0.06; +0.19] | 0% | +0.50 | null |
| DREBIN cluster, fold 3 | λmax | 0.98 | +0.04 | [−0.07; +0.17] | +0.07 | [−0.06; +0.19] | 0% | +0.51 | null |
| DREBIN cluster, fold 3 | sharpness ρ=0.1 | 0.71 | +0.26 | [+0.15; +0.37] | +0.23 | [−0.00; +0.45] | 73% | −0.50 | sustains |
| DREBIN cluster, fold 3 | sharpness ρ=0.5 | 0.63 | +0.26 | [+0.15; +0.37] | +0.24 | [+0.02; +0.44] | 69% | −0.14 | sustains |
| DREBIN cluster, fold 3 | ρ within(val, target); sd within of target / val |  | +0.09 |  |  |  |  | 0.422 / 0.019 |  |
