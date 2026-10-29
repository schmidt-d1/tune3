<!-- fontes: D2 JSONs, d2_followup -->

**Table 10. CIFAR-10 grid (3⁵ = 243 configurations, 3 seeds): (A) replication of the generalization-gap correlation (Kendall τ, granulated Ψ) and (B) the Tune3 question on CIFAR-10.1.**

| seed | n | τ(gap) [CI95] | Ψ | factors with ψ > 0 (p < 0.05) | (A) | (B) partial ∣ val, arch [CI95] | partial ∣ val | partial ∣ val, 5 hp [CI95] | ref: τ of val error | ref: τ of dropout |
|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 0 | 225 | +0.46 [+0.38; +0.53] | +0.31 | n_layers, learning_rate, dropout | positive | −0.07 [−0.19; +0.07] | −0.22 | −0.11 [−0.23; +0.03] | +0.91 | +0.57 |
| 1 | 225 | +0.48 [+0.42; +0.55] | +0.30 | n_layers, learning_rate, dropout | positive | −0.08 [−0.20; +0.05] | −0.15 | −0.02 [−0.14; +0.10] |  |  |
| 2 | 224 | +0.54 [+0.48; +0.60] | +0.38 | n_layers, learning_rate, weight_decay, dropout | positive | −0.28 [−0.39; −0.15] | −0.32 | +0.08 [−0.06; +0.20] |  |  |
