# tune3/experiments/stats.py
"""
Protocolo estatistico do Tune3 (manual, secao de analise estatistica).

Comparacoes PAREADAS por seed: cada metodo e' rodado nas MESMAS seeds, e
comparamos Tune3 vs cada baseline par a par.

Componentes:
  - Wilcoxon signed-rank (pareado, nao-parametrico): nao assume normalidade.
  - Correcao Bonferroni-Holm: controla o erro familiar (FWER) sobre a familia
    de comparacoes (Tune3 vs B1, vs B2, vs B3, ...).
  - d_z de Cohen pareado: tamanho de efeito = media(dif)/desvio(dif).
  - IC bootstrap da diferenca media.

ATENCAO SOBRE PODER (importante para o piloto):
  Wilcoxon two-sided com n pares tem p-minimo = 2 / 2^n (todos com mesmo sinal).
  n=3 -> p_min = 0.25;  n=5 -> 0.0625;  n=6 -> 0.03125.
  Ou seja, com < 6 seeds e' IMPOSSIVEL atingir p < 0.05. O piloto de 3 seeds
  valida o PIPELINE, nao produz significancia. Por isso o protocolo usa N=20
  nas comparacoes primarias (ver analise de poder do artigo).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def cohens_dz(diffs) -> float:
    """d_z de Cohen pareado = media(dif) / desvio-padrao(dif) (ddof=1)."""
    d = np.asarray(diffs, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 1e-12 else 0.0


def bootstrap_ci(diffs, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    """IC (1-alpha) bootstrap percentil para a media das diferencas."""
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.array([rng.choice(d, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def min_achievable_p(n_pairs: int) -> float:
    """p-valor minimo teorico do Wilcoxon two-sided com n pares."""
    if n_pairs < 1:
        return 1.0
    return min(1.0, 2.0 / (2 ** n_pairs))


def compare_paired(
    tune3_scores: List[float],
    baseline_scores: Dict[str, List[float]],
    alpha: float = 0.05,
    lower_is_better: bool = True,
) -> Dict:
    """
    Compara Tune3 vs cada baseline (pareado por seed).

    tune3_scores: lista de scores por seed (ex.: CVaR no teste).
    baseline_scores: {nome_baseline: lista de scores por seed}.
    lower_is_better: True para CVaR/loss (menor e' melhor).

    Retorna, por baseline: diff_mean, dz, ci, p_raw, p_holm, significant, win.
    Convencao: diff = (baseline - tune3) se lower_is_better, de modo que
    diff > 0 e dz > 0 significam TUNE3 MELHOR.
    """
    t = np.asarray(tune3_scores, dtype=float)
    n = len(t)
    names = list(baseline_scores.keys())
    raw_p, rows = [], {}

    for name in names:
        b = np.asarray(baseline_scores[name], dtype=float)
        if len(b) != n:
            raise ValueError(f"baseline '{name}' tem {len(b)} seeds, Tune3 tem {n}")
        diff = (b - t) if lower_is_better else (t - b)  # >0 => Tune3 melhor
        # Wilcoxon signed-rank (two-sided); trata diffs todas nulas
        if np.allclose(diff, 0.0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(t, b)
            except ValueError:
                p = 1.0
        raw_p.append(p)
        rows[name] = {
            "diff_mean": float(diff.mean()),
            "dz": cohens_dz(diff),
            "ci95": bootstrap_ci(diff),
            "p_raw": float(p),
            "win": bool(diff.mean() > 0),
        }

    # Bonferroni-Holm sobre a familia de comparacoes
    if raw_p:
        reject, p_corr, _, _ = multipletests(raw_p, alpha=alpha, method="holm")
        for i, name in enumerate(names):
            rows[name]["p_holm"] = float(p_corr[i])
            rows[name]["significant"] = bool(reject[i])

    return {
        "n_seeds": n,
        "min_achievable_p": min_achievable_p(n),
        "alpha": alpha,
        "comparisons": rows,
    }


def summarize(scores: List[float]) -> Dict:
    """Resumo descritivo (mediana, IQR, media, desvio) de uma lista de scores."""
    a = np.asarray(scores, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"median": float(med), "iqr": float(q3 - q1),
            "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a)}
