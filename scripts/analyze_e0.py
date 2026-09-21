#!/usr/bin/env python
# scripts/analyze_e0.py
"""
E0 -- Reanalise de custo zero: correlacao entre CVaR e log Tr(H^2) DENTRO dos folds
de H3 (zero-day), com a correlacao PARCIAL controlando log(lr) e log(wd) (protocolo V2).

Fonte: os JSONs de results/s2_zeroday/ gerados a partir do commit corrigido
(cada trial do Tune3 e' registrado em payload.results_by_fold[fold].per_seed[i]._telemetry.tune3
com hparams, cvar (validacao) e curvature). JSONs antigos (pre-auditoria) NAO servem:
tinham vazamento do teste.

Predicao de H_G (geometrica): correlacao parcial POSITIVA (mais curvatura, mais CVaR),
com IC BCa excluindo zero. Predicao de H_M: correlacao parcial ~0.

Uso:
    python scripts/analyze_e0.py                       # varre results/s2_zeroday/*.json
    python scripts/analyze_e0.py --file results/s2_zeroday/2026...json --tag aluno-nome
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
from scipy.stats import bootstrap, pearsonr, spearmanr

from tune3.experiments.results_io import load_results, save_result


def _trials(doc):
    """Extrai (fold, seed, log_lr, log_wd, cvar, log_curv) de todos os trials do Tune3 real."""
    rows = []
    rbf = doc["payload"]["results_by_fold"]
    for fold, res in rbf.items():
        for entry in res["per_seed"]:
            tel = (entry.get("_telemetry") or {}).get("tune3") or []
            for t in tel:
                if t.get("aborted") or not np.isfinite(t["curvature"]) or t["curvature"] <= 0:
                    continue
                hp = t["hparams"]
                rows.append((int(fold), int(entry["seed"]), math.log10(hp["learning_rate"]),
                             math.log10(max(hp["weight_decay"], 1e-12)), float(t["cvar"]),
                             math.log10(1.0 + t["curvature"])))
    return np.array(rows, dtype=float)


def partial_corr(y, x, Z):
    """Correlacao parcial de y e x controlando as colunas de Z (residuos de OLS)."""
    A = np.column_stack([np.ones(len(y)), Z])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    return float(pearsonr(ry, rx)[0]), ry, rx


def bca_ci(ry, rx, n_boot=5000, seed=0):
    def stat(a, b):
        return pearsonr(a, b)[0]
    try:
        res = bootstrap((ry, rx), stat, paired=True, n_resamples=n_boot, method="BCa",
                        random_state=np.random.default_rng(seed), vectorized=False)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time()

    docs = []
    if args.file:
        docs = [(args.file, json.load(open(args.file, encoding="utf-8")))]
    else:
        docs = list(load_results("s2_zeroday"))
    if not docs:
        raise SystemExit("nenhum JSON de S2 encontrado em results/s2_zeroday/")

    out = {}
    for path, doc in docs:
        T = _trials(doc)
        if len(T) < 10:
            print(f"[pulado] {path}: {len(T)} trials validos"); continue
        fold, seed, llr, lwd, cvar, lcurv = T.T
        r_raw = float(pearsonr(cvar, lcurv)[0]); rho = float(spearmanr(cvar, lcurv)[0])
        r_par, ry, rx = partial_corr(cvar, lcurv, np.column_stack([llr, lwd]))
        lo, hi = bca_ci(ry, rx)
        per_fold = {}
        for f in np.unique(fold):
            m = fold == f
            if m.sum() >= 8:
                rp, a, b = partial_corr(cvar[m], lcurv[m], np.column_stack([llr[m], lwd[m]]))
                per_fold[int(f)] = {"n": int(m.sum()), "r_partial": rp}
        verdict = ("consistente com H_G (r_parcial>0, IC exclui 0)" if lo > 0 else
                   "consistente com H_M (IC contem 0 ou r_parcial<=0)" if not (lo > 0) and np.isfinite(lo) else "EM ABERTO")
        out[path] = {"n_trials": int(len(T)), "r_pearson": r_raw, "rho_spearman": rho,
                     "r_partial_ctrl_lr_wd": r_par, "r_partial_ci95_bca": [lo, hi],
                     "per_fold": per_fold, "verdict": verdict,
                     "git_commit_of_source": doc.get("meta", {}).get("git", {}).get("commit_short")}
        print(f"\n{path}\n  trials={len(T)}  r={r_raw:+.3f}  rho={rho:+.3f}  "
              f"r_parcial(ctrl log lr, log wd)={r_par:+.3f}  IC95 BCa=[{lo:+.3f},{hi:+.3f}]")
        for f, v in per_fold.items():
            print(f"    fold {f}: n={v['n']}  r_parcial={v['r_partial']:+.3f}")
        print(f"  -> {verdict}")
    save_result(out, experiment="e0_reanalysis", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
