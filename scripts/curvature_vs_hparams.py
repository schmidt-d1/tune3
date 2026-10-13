#!/usr/bin/env python
# scripts/curvature_vs_hparams.py
"""
Cenario 1 (out/2026): quanto da curvatura, no ponto de parada, e' FUNCAO DOS HIPERPARAMETROS?

Para cada rodada (E0 ou D2), ajusta por minimos quadrados log10 Tr(H^2) ~ 5 hiperparametros (log lr,
log wd, dropout, largura, profundidade), linear e com termos quadraticos/cruzados, e reporta o R^2 (e o
R^2 ajustado). Se for alto, explica mecanicamente o resultado do E0: pouco sobra na "geometria" que nao
seja arquitetura e regularizacao -- e esse pouco nao preve o alvo deslocado. Tambem reporta o R^2 por
hiperparametro isolado (quem explica a curvatura) e, para a sharpness adaptativa (D2, ou E0 com
--measures), o mesmo ajuste.

Uso:
    python scripts/curvature_vs_hparams.py "results/e0_generalization/*k600-stop-s*.json" "results/e0_generalization/*drebin-cluster*-f*.json" \\
        "results/e0_generalization/*krono-s*.json" "results/d2_cifar/*grid*.json" --tag dionatan
"""
from __future__ import annotations

import argparse, glob, json, os, time
import numpy as np

from tune3.experiments.results_io import save_result

MEASURES = ["curvature", "sharp_0.1", "sharp_0.5", "lmax"]


def r2(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ beta
    ss = ((y - y.mean()) ** 2).sum(); r = 1 - (res ** 2).sum() / ss if ss > 0 else float("nan")
    n, p = X.shape; adj = 1 - (1 - r) * (n - 1) / max(n - p, 1)
    return float(r), float(adj)


def design(H, quad=False):
    cols = [np.ones(len(H[0]))] + list(H)
    if quad:
        cols += [a * b for i, a in enumerate(H) for b in H[i:]]
    return np.column_stack(cols)


def rows_of(J):
    P = J["payload"]
    if "datasets" in P:                       # E0
        for d in P["datasets"]:
            ok = [r for r in d["rows"] if r.get("reached", r.get("best_epoch", 2) >= 2) and np.isfinite(r["curvature"]) and 0 < r["curvature"] != 1e3]
            a = J["meta"].get("args", {})
            name = f"{d['dataset']} fold {a.get('cluster_fold')}" if d["dataset"] == "drebin_cluster" else f"{d['dataset']} semente {a.get('seed')}"
            yield name, ok
    else:                                     # D2
        ok = [r for r in P["rows"] if r.get("reached") and np.isfinite(r["curvature"]) and r["curvature"] > 0]
        yield f"cifar semente {J['meta'].get('args', {}).get('seed')}", ok


def main():
    ap = argparse.ArgumentParser(description="R^2 de log Tr(H^2) (e outras medidas) sobre os hiperparametros, por rodada.")
    ap.add_argument("files", nargs="+"); ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time()
    files = sorted({f for pat in args.files for f in glob.glob(pat)}); out = []
    print(f"  {'rodada':28s} {'medida':12s} {'n':>4} {'R2 lin':>7} {'R2 adj':>7} {'R2 quad':>8} | R2 de cada hiperparametro sozinho")
    for f in files:
        J = json.load(open(f))
        for name, ok in rows_of(J):
            if len(ok) < 20:
                continue
            h = lambda k: np.array([r["hparams"][k] for r in ok], float)
            H = {"log_lr": np.log10(h("learning_rate")), "log_wd": np.log10(h("weight_decay") + 1e-6), "dropout": h("dropout"),
                 "largura": h("hidden_dim"), "prof.": h("n_layers")}
            for key in MEASURES:
                if key not in ok[0]:
                    continue
                m = np.array([r.get(key, np.nan) for r in ok], float); okm = np.isfinite(m) & (m > 0)
                if okm.sum() < 20:
                    continue
                y = np.log10(m[okm]); HH = [c[okm] for c in H.values()]
                rl, ra = r2(design(HH), y); rq, _ = r2(design(HH, quad=True), y)
                single = {k: r2(design([c[okm]]), y)[0] for k, c in H.items()}
                out.append({"src": os.path.basename(f), "run": name, "measure": key, "n": int(okm.sum()), "r2_linear": rl, "r2_adj": ra, "r2_quadratic": rq, "r2_single": single})
                print(f"  {name:28s} {key:12s} {int(okm.sum()):>4} {rl:>7.2f} {ra:>7.2f} {rq:>8.2f} | " + "  ".join(f"{k}={v:.2f}" for k, v in single.items()))
    save_result({"rows": out}, experiment="curvature_vs_hparams", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
