# tune3/experiments/val_scarcity.py
"""
F1 -- a curvatura acrescenta informacao quando a validacao e' ESCASSA? (pre-registrado em 08/10/2026)

Nada aqui treina modelos: trabalha sobre as perdas por amostra gravadas (results/raw). Para cada
tamanho de validacao n_v, sorteia subamostras de validacao COMUNS a todos os modelos, calcula a
estatistica de validacao de cada modelo nessa subamostra (CVaR_0,95 -- o objetivo do Tune3 -- ou perda
media) e mede a correlacao parcial entre a medida geometrica e o alvo (teste deslocado), dada a
validacao escassa:

    P(n_v) = rho( alvo , medida | val(n_v) )           media sobre R subamostras
    IC95 por bootstrap em dois niveis: subamostra nova + reamostragem dos modelos (B vezes)

Tambem: P_arq(n_v) (controle adicional: profundidade e largura, conhecidas por quem otimiza) e, como
referencia de "folga", rho(val(n_v), alvo) e a parcial de cada hiperparametro | val(n_v).

Regra pre-registrada (scan_verdict): "escassez sustenta H_G" se em TODOS os n_v <= 200 o IC95 de P
ficar acima de zero (mais curvatura -> pior alvo) e P nao crescer com n_v; IC95 abaixo de zero nos
tres: contradiz; senao: nao sustenta.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from tune3.experiments.stats import partial_spearman

SIZES_DEFAULT = (50, 100, 200, 500, 1000, 2000)


def cvar_rows(L: np.ndarray, gamma: float = 0.95) -> np.ndarray:
    """CVaR empirico por linha (mesma definicao de tune3.core.objectives.cvar: media da cauda >= VaR)."""
    q = np.quantile(L, gamma, axis=1, keepdims=True); m = L >= q
    return (L * m).sum(1) / m.sum(1)


def val_stat(L: np.ndarray, stat: str) -> np.ndarray:
    if stat == "cvar":
        return cvar_rows(L)
    if stat == "mean":
        return L.mean(1)
    raise ValueError(stat)


def scan(V: np.ndarray, target: np.ndarray, measure: np.ndarray, arch: Optional[Sequence[np.ndarray]] = None,
         hparams: Optional[Dict[str, np.ndarray]] = None, sizes: Sequence[int] = SIZES_DEFAULT,
         stat: str = "cvar", R: int = 200, B: int = 1000, seed: int = 0) -> Dict:
    """V: (modelos x n_val) perdas por amostra da validacao; target/measure: (modelos,).
    Devolve, por tamanho: P (media de R), IC95 (dois niveis), P_arq, referencias."""
    V = np.asarray(V, float); y = np.asarray(target, float); x = np.asarray(measure, float)
    n_models, n_val = V.shape
    rng = np.random.default_rng(seed)
    sizes = [int(s) for s in sizes if s < n_val] + [n_val]        # o ultimo e' a validacao completa
    arch = list(arch) if arch else []
    hparams = hparams or {}
    out = {"stat": stat, "n_models": int(n_models), "n_val": int(n_val), "R": R, "B": B, "sizes": sizes, "per_size": []}

    for nv in sizes:
        full = nv >= n_val
        def sub():
            return V if full else V[:, rng.choice(n_val, nv, replace=False)]
        # ponto: media sobre R subamostras (1 so' se for a validacao inteira)
        reps = 1 if full else R
        P, Pa, rv, hp_p = [], [], [], {k: [] for k in hparams}
        for _ in range(reps):
            v = val_stat(sub(), stat)
            P.append(partial_spearman(y, x, [v]))
            if arch:
                Pa.append(partial_spearman(y, x, [v] + arch))
            rv.append(float(spearmanr(v, y)[0]))
            for k, h in hparams.items():
                hp_p[k].append(partial_spearman(y, h, [v]))
        # IC em dois niveis: subamostra nova + reamostragem dos modelos
        bp, bpa = [], []
        for _ in range(B):
            v = val_stat(sub(), stat); i = rng.integers(0, n_models, n_models)
            bp.append(partial_spearman(y[i], x[i], [v[i]]))
            if arch:
                bpa.append(partial_spearman(y[i], x[i], [v[i]] + [a[i] for a in arch]))
        bp = np.array(bp); bp = bp[np.isfinite(bp)]
        row = {"n_val": int(nv), "P": float(np.nanmean(P)), "P_ci95": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
               "rho_val_target": float(np.nanmean(rv)),
               "hparam_partial_given_val": {k: float(np.nanmean(v_)) for k, v_ in hp_p.items()}}
        if arch:
            bpa = np.array(bpa); bpa = bpa[np.isfinite(bpa)]
            row["P_arch"] = float(np.nanmean(Pa))
            row["P_arch_ci95"] = [float(np.percentile(bpa, 2.5)), float(np.percentile(bpa, 97.5))]
        out["per_size"].append(row)
    out["verdict"] = scan_verdict(out)
    return out


def scan_verdict(res: Dict, small: int = 200) -> str:
    rows = res["per_size"]
    sm = [r for r in rows if r["n_val"] <= small]
    if len(sm) < 3:
        return "EM ABERTO: menos de tres tamanhos <= %d" % small
    lo = [r["P_ci95"][0] for r in sm]; hi = [r["P_ci95"][1] for r in sm]
    P = [r["P"] for r in rows]; nv = [r["n_val"] for r in rows]
    trend = float(spearmanr(nv, P)[0]) if len(rows) >= 3 else float("nan")
    if all(l > 0 for l in lo) and (not np.isfinite(trend) or trend <= 0):
        return (f"ESCASSEZ SUSTENTA H_G: IC95 acima de zero em todos os n_v <= {small} e P nao cresce com n_v "
                f"(tendencia {trend:+.2f})")
    if all(h < 0 for h in hi):
        return f"CONTRADIZ H_G: IC95 abaixo de zero em todos os n_v <= {small}"
    if all(l > 0 for l in lo):
        return f"NAO SUSTENTA: IC95 acima de zero em n_v <= {small}, mas P cresce com n_v (tendencia {trend:+.2f})"
    return f"NAO SUSTENTA: IC95 contem zero em algum n_v <= {small}"


def format_scan(res: Dict, label: str) -> str:
    lines = [f"  [{label}] estatistica de validacao = {res['stat']}; {res['n_models']} modelos; R={res['R']}, B={res['B']}",
             f"  {'n_val':>6} {'P=parcial|val':>14} {'IC95':>16} {'P_arq':>7} {'IC95':>16} {'rho(val,alvo)':>14}  parciais dos hiperparametros | val"]
    for r in res["per_size"]:
        hp = "  ".join(f"{k}={v:+.2f}" for k, v in r["hparam_partial_given_val"].items())
        pa = f"{r['P_arch']:>+7.3f} [{r['P_arch_ci95'][0]:+.2f}, {r['P_arch_ci95'][1]:+.2f}]" if "P_arch" in r else f"{'':>7} {'':>16}"
        lines.append(f"  {r['n_val']:>6} {r['P']:>+14.3f} [{r['P_ci95'][0]:+.2f}, {r['P_ci95'][1]:+.2f}] {pa} {r['rho_val_target']:>+14.3f}  {hp}")
    lines.append(f"  -> {res['verdict']}")
    return "\n".join(lines)
