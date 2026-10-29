<!-- fontes: e1_offline -->

**Table 8. Offline tuner criterion (budget b=30, w=1): change in the shifted-target CVaR of the chosen configuration relative to the validation-only criterion, per seed, and the 3/3 rule; regret = validation-only choice minus the oracle.**

| Scenario | ρ(val, target) s0/s1/s2 | regret of V | V+Tr(H²): Δ s0/s1/s2 → rule | V+λmax | V+sharpness | V+placebo |
|:--|:--|:--|:--|:--|:--|:--|
| NSL-KDD | +0.33/+0.45/+0.16 | 0.32/0.48/0.54 | +0.08/+0.11/+0.12 → hurts in 3/3 | +0.07 → hurts | +0.07 → hurts | +0.07/+0.10/+0.04 → mixed |
| KronoDroid | +0.93/+0.92/+0.90 | 0.04/0.03/0.04 | +0.07/+0.16/+0.10 → hurts in 3/3 | +0.08/+0.24/+0.12 → mixed | +0.08/+0.15/+0.07 → hurts in 3/3 | +0.13/+0.20/+0.14 → hurts in 3/3 |
| DREBIN cluster, fold 0 | +0.75/+0.79/+0.70 | 0.07/0.06/0.08 | −0.01/+0.02/+0.00 → null in 3/3 | −0.02/−0.01/−0.04 → null in 3/3 | +0.14/+0.17/+0.14 → hurts in 3/3 | +0.06/+0.07/+0.11 → hurts in 3/3 |
| DREBIN cluster, fold 1 | −0.01/−0.06/+0.01 | 0.14/0.19/0.18 | +0.01/−0.01/+0.01 → null in 3/3 | +0.00/−0.01/−0.00 → null in 3/3 | −0.02/−0.03/−0.01 → mixed | +0.01/−0.01/+0.00 → null in 3/3 |
| DREBIN cluster, fold 2 | +0.82/+0.81/+0.86 | 0.07/0.04/0.08 | +0.03/+0.07/+0.02 → mixed | +0.03/+0.05/+0.03 → mixed | +0.12/+0.14/+0.07 → hurts in 3/3 | +0.09/+0.13/+0.08 → hurts in 3/3 |
| DREBIN cluster, fold 3 | +0.51/+0.49/+0.47 | 0.50/0.28/0.54 | +0.04/+0.16/+0.08 → null in 3/3 | +0.06/+0.13/+0.02 → null in 3/3 | +0.38/+0.73/+0.47 → hurts in 3/3 | +0.29/+0.50/+0.28 → hurts in 3/3 |
| DREBIN cluster, fold 4 | +0.79/+0.76/+0.75 | 0.23/0.17/0.12 | +0.05/+0.06/+0.03 → mixed | +0.05/+0.05/+0.03 → mixed | +0.15/+0.12/+0.18 → hurts in 3/3 | +0.08/+0.06/+0.16 → hurts in 3/3 |
