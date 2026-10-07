# tune3/experiments/generalization_measures.py
"""
Analises da etapa D2 (controle positivo no CIFAR-10), no formato de Jiang et al. (ICLR 2020).

  kendall_tau(m, g)                  tau de Kendall entre a medida m e o gap de generalizacao g.
  granulated_kendall(rows, ...)      "Kendall granulado" (Psi de Jiang et al.): para cada
                                     hiperparametro i, media do tau calculado DENTRO dos grupos em
                                     que so' i varia (todos os outros fixos); Psi = media dos psi_i.
                                     Como cada tau compara modelos que diferem em UM fator, a medida
                                     nao pode "ganhar" so' por acompanhar outro fator (ex.: a
                                     profundidade, que dominou a curvatura nos tres cenarios da D1).
  replication_positive(tau_ci, gk)   regra da analise (A) da D2, fixada ANTES de rodar: IC95 do tau
                                     acima de zero, Psi > 0 E pelo menos 3 dos 5 fatores com psi_i
                                     SIGNIFICATIVAMENTE positivo (teste do sinal unilateral sobre os
                                     grupos: mais grupos com tau > 0 que com tau < 0, p < 0,05).
                                     A terceira condicao existe porque uma medida que so' acompanha a
                                     profundidade passa nas duas primeiras (tau alto, Psi ~ psi_prof/5 > 0)
                                     sem prever nada do que os outros fatores fazem com o gap.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import kendalltau, binom


def kendall_tau(m, g) -> float:
    m = np.asarray(m, float); g = np.asarray(g, float)
    ok = np.isfinite(m) & np.isfinite(g)
    if ok.sum() < 3:
        return float("nan")
    t = kendalltau(m[ok], g[ok]).statistic
    return float(t) if np.isfinite(t) else float("nan")


def granulated_kendall(rows: Sequence[Dict], measure: str, target: str,
                       factors: Sequence[str]) -> Dict:
    """`rows`: dicionarios com rows[k]["hparams"][f] para cada fator f, e os campos `measure` e
    `target`. Grupos com menos de 2 modelos validos sao ignorados; grupos em que a medida ou o alvo
    nao variam contribuem com tau indefinido e tambem sao ignorados."""
    psi = {}
    for f in factors:
        others = [o for o in factors if o != f]
        groups: Dict[tuple, List] = defaultdict(list)
        for r in rows:
            key = tuple(r["hparams"][o] for o in others)
            groups[key].append((r[measure], r[target]))
        taus = []
        for pts in groups.values():
            if len(pts) < 2:
                continue
            m = np.array([p[0] for p in pts], float); g = np.array([p[1] for p in pts], float)
            if np.ptp(m) == 0 or np.ptp(g) == 0:
                continue
            t = kendalltau(m, g).statistic
            if np.isfinite(t):
                taus.append(float(t))
        npos = sum(t > 0 for t in taus); nneg = sum(t < 0 for t in taus)
        p_sign = float(binom.sf(npos - 1, npos + nneg, 0.5)) if (npos + nneg) else float("nan")
        psi[f] = {"psi": float(np.mean(taus)) if taus else float("nan"), "n_groups": len(taus),
                  "n_pos": int(npos), "n_neg": int(nneg), "p_sign": p_sign}
    vals = [v["psi"] for v in psi.values() if np.isfinite(v["psi"])]
    return {"Psi": float(np.mean(vals)) if vals else float("nan"), "per_factor": psi}


def replication_positive(tau_ci, gk: Dict, min_factors: int = 3, alpha: float = 0.05) -> bool:
    n_pos = sum(1 for v in gk["per_factor"].values()
                if np.isfinite(v.get("p_sign", np.nan)) and v["p_sign"] < alpha and v["psi"] > 0)
    return bool(np.isfinite(tau_ci[0]) and tau_ci[0] > 0 and np.isfinite(gk["Psi"]) and gk["Psi"] > 0
                and n_pos >= min_factors)
