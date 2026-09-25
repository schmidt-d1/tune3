#!/usr/bin/env python
# scripts/e0_bootstrap2.py
"""
IC em DOIS NIVEIS para a parcial do E0, a partir das perdas por amostra gravadas (--save-losses).

Por que (25/09/2026). O IC que o E0 imprime reamostra so' as CONFIGURACOES: trata o conjunto de
teste (e o de validacao) como fixo. Mas o sorteio das instancias de teste e' uma fonte de
variacao comum a todas as configuracoes -- a checagem de "metades i.i.d." mostrou valores de ate'
|0,28| em alvos pequenos, que e' exatamente essa variacao aparecendo. Aqui cada reamostragem
sorteia (i) as instancias de validacao, (ii) as instancias do alvo e (iii) as configuracoes, e
recalcula tudo (CVaR por configuracao e parcial). Nao treina nada: le results/raw/*.npz.

Uso:
    python scripts/e0_bootstrap2.py "results/e0_generalization/*dionatan-drebin-cluster-f*.json" --B 1000
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np

from tune3.experiments.stats import partial_spearman


def cvar_rows(L, gamma=0.95):
    """CVaR empirico linha a linha, mesma definicao de tune3.core.objectives.cvar (quantil + media
    da cauda >= VaR)."""
    v = np.quantile(L, gamma, axis=1, keepdims=True); m = L >= v
    return (L * m).sum(1) / m.sum(1)


def fisher(rs, ns, nc):
    rs = np.asarray(rs, float); w = np.asarray(ns, float) - 3 - nc
    z = np.arctanh(np.clip(rs, -0.9999, 0.9999)); return float(np.tanh((w * z).sum() / w.sum()))


def valid(r):
    ok = np.isfinite(r["curvature"]) and r["curvature"] > 0 and r["curvature"] != 1e3 \
        and np.isfinite(r["cvar_test"]) and r["cvar_test"] != 1e3
    if "reached" in r:                          # parada por perda de treino
        return ok and r["reached"]
    return ok and r.get("best_epoch", 2) >= 2   # early stopping: exclui estagnadas


def main():
    ap = argparse.ArgumentParser(description="IC em dois niveis (instancias + configuracoes) para o E0.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--target", default="cvar_test_novel", choices=["cvar_test", "cvar_test_novel", "cvar_test_known"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = sorted({f for pat in args.files for f in glob.glob(pat)})

    runs = []
    for f in files:
        J = json.load(open(f)); stem = os.path.splitext(os.path.basename(f))[0]
        for d in J["payload"]["datasets"]:
            npz = f"results/raw/{stem}_{d['dataset']}.npz"
            if not os.path.exists(npz):
                print(f"[aviso] sem perdas gravadas para {stem} / {d['dataset']} ({npz}); pulando"); continue
            Z = np.load(npz); ok = [r for r in d["rows"] if valid(r)]
            idx = np.array([r["cfg"] for r in ok])
            V = Z["val"][idx].astype(np.float64); T = Z["test"][idx].astype(np.float64)
            mask = {"cvar_test": np.ones(T.shape[1], bool), "cvar_test_novel": Z["test_novel_mask"],
                    "cvar_test_known": ~Z["test_novel_mask"]}[args.target]
            if not mask.any():
                continue
            h = lambda k: np.array([r["hparams"][k] for r in ok], float)
            H = [np.log10(h("learning_rate")), np.log10(h("weight_decay")), h("dropout"), h("hidden_dim"), h("n_layers")]
            lc = np.log10([r["curvature"] for r in ok]); TT = T[:, mask]
            y0, v0 = cvar_rows(TT), cvar_rows(V)
            runs.append({"name": f"{d['dataset']} fold {J['meta']['args'].get('cluster_fold')}", "n": len(ok),
                         "V": V, "T": TT, "lc": lc, "H": H,
                         "p": partial_spearman(y0, lc, [v0]), "p5": partial_spearman(y0, lc, [v0] + H)})
    if not runs:
        print("nada a analisar"); return

    rng = np.random.default_rng(args.seed)
    per = {i: {"full": [], "p5": []} for i in range(len(runs))}; comb, comb5 = [], []
    for _ in range(args.B):
        rs, r5 = [], []
        for i, R in enumerate(runs):
            nv, nt, n = R["V"].shape[1], R["T"].shape[1], R["n"]
            v = cvar_rows(R["V"][:, rng.integers(0, nv, nv)]); y = cvar_rows(R["T"][:, rng.integers(0, nt, nt)])
            ic = rng.integers(0, n, n)
            a = partial_spearman(y[ic], R["lc"][ic], [v[ic]])
            b = partial_spearman(y[ic], R["lc"][ic], [v[ic]] + [x[ic] for x in R["H"]])
            per[i]["full"].append(a); per[i]["p5"].append(b); rs.append(a); r5.append(b)
        ns = [R["n"] for R in runs]; comb.append(fisher(rs, ns, 1)); comb5.append(fisher(r5, ns, 6))
    q = lambda a: np.nanpercentile(a, [2.5, 97.5])
    print(f"[E0 dois niveis] alvo={args.target}  B={args.B} (instancias de val e do alvo + configuracoes)\n")
    print(f"  {'rodada':24s} {'n':>4} {'parcial|val':>12} {'IC95':>16} {'|val,5hp':>9} {'IC95':>16}")
    for i, R in enumerate(runs):
        a, b = q(per[i]["full"]), q(per[i]["p5"])
        print(f"  {R['name']:24s} {R['n']:>4} {R['p']:>+12.3f}  [{a[0]:+.2f}, {a[1]:+.2f}] {R['p5']:>+9.3f}  [{b[0]:+.2f}, {b[1]:+.2f}]")
    if len(runs) > 1:
        ns = [R["n"] for R in runs]; c, c5 = q(comb), q(comb5)
        print(f"\n  combinado (efeito fixo) parcial|val = {fisher([R['p'] for R in runs], ns, 1):+.3f} "
              f"[{c[0]:+.3f}, {c[1]:+.3f}];  |val,5hp = {fisher([R['p5'] for R in runs], ns, 6):+.3f} [{c5[0]:+.3f}, {c5[1]:+.3f}]")
        print("  (este IC incorpora a variacao de instancias, nao a heterogeneidade entre folds -- para isso,"
              " ver o IC de efeitos aleatorios do e0_combine)")


if __name__ == "__main__":
    main()
