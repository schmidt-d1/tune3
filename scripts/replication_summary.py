#!/usr/bin/env python
# scripts/replication_summary.py
"""
F2 -- resumo da replicacao com varias sementes (regra pre-registrada em claude/TUNE3_PREREGISTRO_FASE_DIRECAO.md).

Le os JSONs de cada semente e aplica, POR SEMENTE, o padrao pre-registrado:
  NSL-KDD (E0, k=600, parada por perda):  parcial Tr(H^2) | val com IC95 que NAO fica inteiramente
                                          acima de zero  E  parcial | val, 5hp com IC95 (bootstrap sobre
                                          modelos, recalculado aqui a partir das linhas do JSON) contendo 0
  DREBIN por cluster (E0, 5 folds):       combinado por efeitos aleatorios (parcial | val) com IC contendo 0
                                          E  combinado | val, 5hp com IC contendo 0
  CIFAR (D2, grade):                      (A) positivo com a Tr(H^2) pela regra da D2  E  (B) IC95 contendo 0
Uma semente que viole o padrao e' reportada como NAO REPLICADO naquele cenario -- nenhuma e' excluida.

Uso:
    python scripts/replication_summary.py \\
        --nsl  results/e0_generalization/*k600-stop-s0.json results/e0_generalization/*k600-stop-s1.json results/e0_generalization/*k600-stop-s2.json \\
        --drebin-cluster "results/e0_generalization/*drebin-cluster-f*.json" "results/e0_generalization/*drebin-cluster-s1-f*.json" "results/e0_generalization/*drebin-cluster-s2-f*.json" \\
        --cifar results/d2_cifar/*dionatan-grid.json results/d2_cifar/*grid-s1.json results/d2_cifar/*grid-s2.json --tag dionatan
"""
from __future__ import annotations

import argparse, glob, json, os, time
import numpy as np

from tune3.experiments.results_io import save_result
from tune3.experiments.stats import fisher_combine, partial_spearman

CONTROLS_5HP = 5


def _rows_ok(d):
    return [r for r in d["rows"] if r.get("reached", r.get("best_epoch", 2) >= 2)
            and np.isfinite(r["curvature"]) and 0 < r["curvature"] != 1e3 and r.get("cvar_test", 0) != 1e3]


def _partial_5hp_ci(rows, target_key, B=2000, seed=0):
    y = np.array([r[target_key] for r in rows]); v = np.array([r["cvar_val"] for r in rows])
    x = np.log10([r["curvature"] for r in rows]); h = lambda k: np.array([r["hparams"][k] for r in rows], float)
    C = [v, np.log10(h("learning_rate")), np.log10(h("weight_decay") + 1e-6), h("dropout"), h("hidden_dim"), h("n_layers")]
    p = partial_spearman(y, x, C); n = len(y); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, n, n); bs.append(partial_spearman(y[i], x[i], [c[i] for c in C]))
    bs = np.array(bs); bs = bs[np.isfinite(bs)]
    return float(p), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _git(J):
    g = J["meta"].get("git", {}); return f"{g.get('commit_short', '?')}{' SUJO' if g.get('dirty') else ''}"


def nsl(paths):
    out = []
    for p in paths:
        J = json.load(open(p)); d = [x for x in J["payload"]["datasets"] if x["dataset"] == "nslkdd"][0]
        r = d["results"]["cvar_test_novel"]; rows = _rows_ok(d)
        p5, ci5 = _partial_5hp_ci(rows, "cvar_test_novel")
        ok = (not r["partial_ci95"][0] > 0) and (ci5[0] <= 0 <= ci5[1])
        out.append({"src": os.path.basename(p), "seed": J["meta"]["args"].get("seed"), "git": _git(J), "n": len(rows),
                    "partial_val": r["partial_given_val"], "ci_val": r["partial_ci95"], "partial_5hp": p5, "ci_5hp": ci5, "replica": bool(ok)})
    return out


def drebin_cluster(globs):
    out = []
    for g in globs:
        files = sorted(glob.glob(g)); rs, ns, r5, seeds, gits = [], [], [], set(), set()
        for p in files:
            J = json.load(open(p)); d = J["payload"]["datasets"][0]; r = d["results"]["cvar_test_novel"]
            rs.append(r["partial_given_val"]); ns.append(d["n_ok"]); r5.append(r.get("partial_given_val_5hp", np.nan))
            seeds.add(J["meta"]["args"].get("seed")); gits.add(_git(J))
        fc = fisher_combine(rs, ns, 1); f5 = fisher_combine(r5, ns, 1 + CONTROLS_5HP)
        ok = (fc["ci95_re"][0] <= 0 <= fc["ci95_re"][1]) and (f5["ci95_re"][0] <= 0 <= f5["ci95_re"][1])
        out.append({"glob": g, "n_folds": len(files), "seed": sorted(seeds), "git": sorted(gits), "per_fold": rs,
                    "re": fc["r_re"], "ci_re": fc["ci95_re"], "I2": fc["I2"], "re_5hp": f5["r_re"], "ci_re_5hp": f5["ci95_re"], "replica": bool(ok)})
    return out


def cifar(paths):
    out = []
    for p in paths:
        J = json.load(open(p)); m = J["payload"]["analysis"]["measures"]["log_curv"]
        okA = bool(m["A_positive"]); ciB = m["partial_B_ci95"]; okB = ciB[0] <= 0 <= ciB[1]
        out.append({"src": os.path.basename(p), "seed": J["meta"]["args"].get("seed"), "git": _git(J), "n": J["payload"]["analysis"]["n_ok"],
                    "tau_gap": m["tau_gap"], "tau_ci": m["tau_ci95"], "Psi": m["Psi"], "A_positive": okA,
                    "partial_B": m["partial_B"], "ci_B": ciB, "replica": bool(okA and okB)})
    return out


def main():
    ap = argparse.ArgumentParser(description="F2: resumo da replicacao com varias sementes.")
    ap.add_argument("--nsl", nargs="*", default=[]); ap.add_argument("--drebin-cluster", nargs="*", default=[])
    ap.add_argument("--cifar", nargs="*", default=[]); ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time(); out = {}
    if args.nsl:
        out["nsl"] = nsl(sorted({f for pat in args.nsl for f in glob.glob(pat)}))
        print("\n== NSL-KDD (E0, ataques novos) ==")
        print(f"  {'semente':>7} {'git':>10} {'n':>4} {'parcial|val':>12} {'IC95':>16} {'|val,5hp':>9} {'IC95':>16}  replica?")
        for r in out["nsl"]:
            print(f"  {str(r['seed']):>7} {r['git']:>10} {r['n']:>4} {r['partial_val']:>+12.3f} [{r['ci_val'][0]:+.2f}, {r['ci_val'][1]:+.2f}] "
                  f"{r['partial_5hp']:>+9.3f} [{r['ci_5hp'][0]:+.2f}, {r['ci_5hp'][1]:+.2f}]  {'sim' if r['replica'] else 'NAO'}")
    if args.drebin_cluster:
        out["drebin_cluster"] = drebin_cluster(args.drebin_cluster)
        print("\n== DREBIN por cluster (E0, 5 folds, efeitos aleatorios) ==")
        print(f"  {'semente':>7} {'folds':>5} {'RE parcial|val':>15} {'IC95':>16} {'I2':>4} {'RE |val,5hp':>12} {'IC95':>16}  replica?")
        for r in out["drebin_cluster"]:
            print(f"  {str(r['seed']):>7} {r['n_folds']:>5} {r['re']:>+15.3f} [{r['ci_re'][0]:+.2f}, {r['ci_re'][1]:+.2f}] {r['I2']:>4.0%} "
                  f"{r['re_5hp']:>+12.3f} [{r['ci_re_5hp'][0]:+.2f}, {r['ci_re_5hp'][1]:+.2f}]  {'sim' if r['replica'] else 'NAO'}"
                  f"   (por fold: {', '.join(f'{x:+.2f}' for x in r['per_fold'])})")
    if args.cifar:
        out["cifar"] = cifar(sorted({f for pat in args.cifar for f in glob.glob(pat)}))
        print("\n== CIFAR (D2, grade) ==")
        print(f"  {'semente':>7} {'git':>10} {'n':>4} {'tau(gap)':>9} {'IC95':>16} {'Psi':>6} {'(A)':>4} {'(B) parcial':>12} {'IC95':>16}  replica?")
        for r in out["cifar"]:
            print(f"  {str(r['seed']):>7} {r['git']:>10} {r['n']:>4} {r['tau_gap']:>+9.3f} [{r['tau_ci'][0]:+.2f}, {r['tau_ci'][1]:+.2f}] {r['Psi']:>+6.2f} "
                  f"{'pos' if r['A_positive'] else 'nao':>4} {r['partial_B']:>+12.3f} [{r['ci_B'][0]:+.2f}, {r['ci_B'][1]:+.2f}]  {'sim' if r['replica'] else 'NAO'}")
    print("\nVEREDITO DA REPLICACAO (regra pre-registrada, por semente):")
    for k, v in out.items():
        bad = [str(r["seed"]) for r in v if not r["replica"]]
        print(f"  {k:15s} {'REPLICADO em todas as sementes' if not bad else 'NAO REPLICADO na(s) semente(s) ' + ', '.join(bad)}  ({len(v)} sementes)")
    save_result(out, experiment="replication_summary", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
