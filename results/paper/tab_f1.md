<!-- fontes: f1_val_size -->

**Table 9. Validation scarcity: partial correlation of log Tr(H²) with the shifted target given a validation CVaR computed on n_v samples (mean over 200 subsamples; two-level bootstrap CI), and the same given depth and width.**

| Scenario | statistic | n_v = 50 | n_v = 100 | n_v = 200 | full validation | verdict |
|:--|:--|:--|:--|:--|:--|:--|
| CIFAR-10.1 (CVaR), log Tr(H²) | P | +0.17 [−0.12; +0.44] | +0.11 [−0.19; +0.42] | +0.05 [−0.26; +0.36] | −0.22 [−0.32; −0.09] (n_v=5000) | does not sustain (CI contains zero at some n_v ≤ 200) |
| CIFAR-10.1, P_arch (| val, depth, width) | P_arch | +0.15 [−0.15; +0.44] | +0.12 [−0.20; +0.41] | +0.08 [−0.24; +0.39] | −0.07 [−0.19; +0.07] (n_v=5000) | does not sustain (CI contains zero at some n_v ≤ 200) |
| DREBIN cluster (RE over 5 folds) | P | +0.23 [+0.07; +0.38] | +0.22 [+0.05; +0.38] | +0.20 [+0.03; +0.35] | +0.14 [−0.06; +0.32] (n_v=1945) | scarcity sustains (CI above zero at all n_v ≤ 200, P non-increasing) |
| DREBIN cluster, P_arch | P_arch | −0.05 [−0.24; +0.14] | −0.05 [−0.23; +0.13] | −0.04 [−0.21; +0.13] | +0.00 [−0.12; +0.12] (n_v=1945) | scarcity sustains (CI above zero at all n_v ≤ 200, P non-increasing) |
| NSL-KDD (RE over 3 seeds) | P | −0.29 [−0.45; −0.10] | −0.30 [−0.47; −0.12] | −0.33 [−0.48; −0.17] | −0.42 [−0.55; −0.27] (n_v=5000) | contradicts (CI below zero at all n_v ≤ 200) |
| NSL-KDD, P_arch | P_arch | −0.14 [−0.21; −0.07] | −0.13 [−0.20; −0.06] | −0.12 [−0.18; −0.05] | +0.01 [−0.06; +0.07] (n_v=5000) | contradicts (CI below zero at all n_v ≤ 200) |
