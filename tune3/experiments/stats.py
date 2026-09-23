# tune3/experiments/stats.py
"""
Protocolo estatistico do Tune3 (PRE-REGISTRADO -- ver PREREGISTRATION.md).

Comparacoes PAREADAS por seed: cada metodo e' rodado nas MESMAS seeds (mesmos
splits), e comparamos Tune3 vs cada baseline par a par.

Componentes:
  - Wilcoxon signed-rank (pareado, nao-parametrico): nao assume normalidade.
  - Correcao Bonferroni-Holm: controla o erro familiar (FWER) sobre a familia
    de comparacoes (Tune3 vs B1, vs B2, ...).
  - d_z de Cohen pareado: tamanho de efeito = media(dif)/desvio(dif).
  - IC 95% bootstrap **BCa** (bias-corrected and accelerated) para d_z e para
    a diferenca media. O percentilico fica disponivel (ci_method="percentile")
    apenas para comparacao com resultados antigos.
  - REGRA DE DECISAO pre-registrada: "vitoria" exige, simultaneamente,
        p_Holm < alpha   E   |d_z| >= dz_min (0.30)   E   direcao favoravel.

ATENCAO SOBRE PODER (importante para o piloto):
  Wilcoxon two-sided com n pares tem p-minimo = 2 / 2^n (todos com mesmo sinal).
  n=3 -> p_min = 0.25;  n=5 -> 0.0625;  n=6 -> 0.03125.
  Ou seja, com < 6 seeds e' IMPOSSIVEL atingir p < 0.05. O piloto de 3 seeds
  valida o PIPELINE, nao produz significancia. O protocolo usa N=20 nas
  comparacoes primarias, com caminho pre-planejado para N=40 em caso inconclusivo.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon, bootstrap as _sp_bootstrap
from statsmodels.stats.multitest import multipletests

DZ_MIN_DEFAULT = 0.30   # limiar pre-registrado de efeito pratico minimo


def cohens_dz(diffs) -> float:
    """d_z de Cohen pareado = media(dif) / desvio-padrao(dif) (ddof=1)."""
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return 0.0
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 1e-12 else 0.0


def _stat_mean(x, axis=-1):
    return np.mean(x, axis=axis)


def _stat_dz(x, axis=-1):
    x = np.asarray(x, dtype=float)
    sd = x.std(axis=axis, ddof=1)
    m = x.mean(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 1e-12, m / np.where(sd > 1e-12, sd, 1.0), 0.0)
    return out


MIN_N_BOOTSTRAP = 6   # abaixo disso o bootstrap nao e' inferencia (ver bootstrap_ci)


def bootstrap_ci(diffs, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
                 statistic: str = "mean", ci_method: str = "bca") -> Tuple[float, float]:
    """
    IC (1-alpha) bootstrap para a estatistica das diferencas pareadas.

    statistic: "mean" (diferenca media) ou "dz" (d_z de Cohen).
    ci_method: "bca" (padrao, pre-registrado) ou "percentile" (legado).

    Se o BCa for indefinido (amostra degenerada: todas as diferencas iguais, ou
    n < 3), cai para o percentilico com aviso -- nunca falha silenciosamente.
    """
    d = np.asarray(diffs, dtype=float)
    stat = _stat_mean if statistic == "mean" else _stat_dz
    if statistic not in ("mean", "dz"):
        raise ValueError("statistic deve ser 'mean' ou 'dz'")
    if d.size < MIN_N_BOOTSTRAP:
        # Com n=3 existem so' 10 reamostras distintas: o "IC" bate na fronteira e parece
        # informativo sem ser (visto nos pilotos de 23/09/2026: limite inferior 0.0000 em
        # todas as comparacoes). Mesmo limiar em que o Wilcoxon passa a poder dar p<0.05.
        warnings.warn(f"bootstrap com n={d.size} < {MIN_N_BOOTSTRAP}: IC nao reportado (NaN).",
                      RuntimeWarning)
        return float("nan"), float("nan")
    if np.allclose(d, d[0]):
        v = float(stat(d)) if d.size else 0.0
        return v, v
    method = {"bca": "BCa", "percentile": "percentile"}[ci_method]
    rng = np.random.default_rng(seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _sp_bootstrap((d,), stat, n_resamples=n_boot, confidence_level=1 - alpha,
                                method=method, random_state=rng, vectorized=True, axis=-1)
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise FloatingPointError("BCa indefinido")
        return lo, hi
    except Exception as e:                       # fallback explicito e avisado
        if method == "BCa":
            warnings.warn(f"BCa indefinido ({e}); usando percentilico.", RuntimeWarning)
            return bootstrap_ci(d, n_boot, alpha, seed, statistic, ci_method="percentile")
        raise


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
    dz_min: float = DZ_MIN_DEFAULT,
    ci_method: str = "bca",
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict:
    """
    Compara Tune3 vs cada baseline (pareado por seed).

    tune3_scores: lista de scores por seed (ex.: CVaR no teste).
    baseline_scores: {nome_baseline: lista de scores por seed}.
    lower_is_better: True para CVaR/loss (menor e' melhor).

    Retorna, por baseline:
      diff_mean, dz, ci95 (dif. media, BCa), dz_ci95 (BCa), p_raw, p_holm,
      significant (p_holm < alpha), practical (|dz| >= dz_min), win (direcao),
      prereg_win (significant AND practical AND win)  <- a regra pre-registrada.
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
        if np.allclose(diff, 0.0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(t, b)
            except ValueError:
                p = 1.0
        raw_p.append(p)
        dz = cohens_dz(diff)
        rows[name] = {
            "diff_mean": float(diff.mean()),
            "dz": dz,
            "ci95": bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed,
                                 statistic="mean", ci_method=ci_method),
            "dz_ci95": bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed,
                                    statistic="dz", ci_method=ci_method),
            "p_raw": float(p),
            "win": bool(diff.mean() > 0),
            "practical": bool(abs(dz) >= dz_min),
        }

    if raw_p:
        reject, p_corr, _, _ = multipletests(raw_p, alpha=alpha, method="holm")
        for i, name in enumerate(names):
            rows[name]["p_holm"] = float(p_corr[i])
            rows[name]["significant"] = bool(reject[i])
            rows[name]["prereg_win"] = bool(reject[i] and rows[name]["practical"] and rows[name]["win"])

    return {
        "n_seeds": n,
        "min_achievable_p": min_achievable_p(n),
        "alpha": alpha,
        "dz_min": dz_min,
        "ci_method": ci_method,
        "underpowered": bool(min_achievable_p(n) >= alpha),
        "comparisons": rows,
    }


def summarize(scores: List[float]) -> Dict:
    """Resumo descritivo (mediana, IQR, media, desvio) de uma lista de scores."""
    a = np.asarray(scores, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"median": float(med), "iqr": float(q3 - q1),
            "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a)}


def aggregate_per_seed(results_by_fold: Dict[str, Dict], method: str,
                       metric: str = "cvar_test") -> List[float]:
    """
    Agrega resultados de S2 (leave-one-cluster-out) POR SEED: media sobre os
    folds, devolvendo UM valor por seed. Evita pseudo-replicacao: os N seeds de
    um mesmo fold compartilham o mesmo conjunto de teste e NAO sao observacoes
    independentes; tratar folds x seeds como F*N pares infla o n do Wilcoxon.
    results_by_fold[fold]["by_metric"][method][metric] e' uma lista alinhada a seeds.
    """
    folds = list(results_by_fold.values())
    n_seeds = len(folds[0]["seeds"])
    out = []
    for i in range(n_seeds):
        out.append(float(np.mean([f["by_metric"][method][metric][i] for f in folds])))
    return out


def partial_spearman(y, x, controls) -> float:
    """Correlacao PARCIAL em postos entre y e x, controlando pelas colunas de `controls`.

    Residuos de OLS dos postos de y e de x sobre os postos dos controles; devolve a correlacao
    de Pearson dos residuos. Robusta a cauda pesada (so' usa ordem). E' a estatistica do E0:
    rho(log Tr(H^2), CVaR_teste | CVaR_validacao) mede se a curvatura carrega informacao sobre
    o teste que a validacao NAO carrega. A correlacao bruta nao serve para isso: se a curvatura
    so' afeta o teste ATRAVES da validacao, a bruta e' positiva e a parcial e' ~0 (redundancia);
    se ha' um efeito direto de sinal oposto, a bruta pode ser ~0 e a parcial nao (supressao).
    """
    from scipy.stats import rankdata, pearsonr
    y = np.asarray(y, float); x = np.asarray(x, float)
    ry, rx = rankdata(y), rankdata(x)
    A = np.column_stack([np.ones(len(y))] + [rankdata(np.asarray(c, float)) for c in controls])
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    if ey.std() == 0 or ex.std() == 0:
        return float("nan")
    return float(pearsonr(ey, ex)[0])
