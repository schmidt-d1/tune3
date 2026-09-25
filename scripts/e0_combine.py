#!/usr/bin/env python
# scripts/e0_combine.py
"""
Combina varias rodadas do E0 (ex.: os 5 folds do DREBIN por cluster) numa estimativa unica.

Por que (25/09/2026). No DREBIN com divisao por cluster, cada fold retem um cluster diferente de
malware como "familia nova" -- cada um e' um cenario de deslocamento distinto. Um resultado que
aparece num fold so' pode ser particularidade daquele cluster; o que interessa e' (i) o efeito
medio e (ii) se os folds concordam. Regra fixada ANTES de rodar os folds:

  - por fold: a parcial rho(log Tr(H^2), CVaR_alvo | CVaR_val) ja' calculada pelo E0;
  - combinacao: media ponderada em z de Fisher, peso n - 4 (n configuracoes validas; 1 controle),
    IC95 da media; heterogeneidade por Q de Cochran e I^2;
  - o veredito combinado usa a MESMA regra do E0: H_G se o IC combinado ficar todo acima de zero;
  - I^2 alto (> 50 %) e' reportado: nesse caso o efeito medio resume mal os folds.

O mesmo e' feito para a parcial com os 5 hiperparametros (|ctrl,5hp) e para a checagem de
metades i.i.d. (que deve dar ~0 em todos os folds).

Uso:
    python scripts/e0_combine.py results/e0_generalization/*dionatan-drebin-cluster-f*.json
    python scripts/e0_combine.py --dataset drebin_cluster --target cvar_test_novel ARQ1.json ARQ2.json ...
"""
from __future__ import annotations

import argparse, glob, json
import numpy as np
from scipy.stats import norm


def fisher_combine(rs, ns, n_controls=1):
    rs = np.asarray(rs, float); ns = np.asarray(ns, float)
    ok = np.isfinite(rs) & (ns - 3 - n_controls > 0)
    rs, ns = rs[ok], ns[ok]
    if len(rs) == 0:
        return {"k": 0}
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999)); w = ns - 3 - n_controls
    zbar = float((w * z).sum() / w.sum()); se = float(1.0 / np.sqrt(w.sum()))
    Q = float((w * (z - zbar) ** 2).sum()); df = len(rs) - 1
    I2 = float(max(0.0, (Q - df) / Q)) if (df > 0 and Q > 0) else 0.0
    p = float(2 * (1 - norm.cdf(abs(zbar / se))))
    return {"k": int(len(rs)), "r": float(np.tanh(zbar)),
            "ci95": [float(np.tanh(zbar - 1.96 * se)), float(np.tanh(zbar + 1.96 * se))],
            "p": p, "Q": Q, "I2": I2}


def main():
    ap = argparse.ArgumentParser(description="Combina rodadas do E0 (Fisher z, heterogeneidade).")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dataset", default="drebin_cluster")
    ap.add_argument("--target", default="cvar_test_novel")
    args = ap.parse_args()
    files = sorted({f for pat in args.files for f in glob.glob(pat)})

    per = []
    for f in files:
        J = json.load(open(f)); meta = J.get("meta", {}); a = meta.get("args", {})
        for d in J["payload"]["datasets"]:
            if d["dataset"] != args.dataset or args.target not in d.get("results", {}):
                continue
            r = d["results"][args.target]; ac = d.get("artifact_checks", {}).get(args.target, {})
            per.append({"file": f, "fold": a.get("cluster_fold"), "n": d["n_ok"],
                        "partial": r["partial_given_val"], "ci": r["partial_ci95"],
                        "p5": r.get("partial_given_val_5hp", np.nan),
                        "halves": ac.get("partial_iid_halves", np.nan),
                        "dirty": meta.get("git", {}).get("dirty"), "commit": meta.get("git", {}).get("commit_short")})
    if not per:
        print("nenhuma rodada encontrada para", args.dataset, args.target); return
    per.sort(key=lambda x: (x["fold"] is None, x["fold"]))
    print(f"[E0 combinado] dataset={args.dataset} alvo={args.target}  ({len(per)} rodadas)\n")
    print(f"  {'fold':>4} {'n':>4} {'parcial|val':>12} {'IC95':>16} {'|val,5hp':>9} {'metades':>8}  commit")
    for x in per:
        print(f"  {str(x['fold']):>4} {x['n']:>4} {x['partial']:>+12.3f}  [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
              f" {x['p5']:>+9.3f} {x['halves']:>+8.3f}  {x['commit']}{' (SUJO)' if x['dirty'] else ''}")
    ns = [x["n"] for x in per]
    out = {"primary": fisher_combine([x["partial"] for x in per], ns, 1),
           "given_5hp": fisher_combine([x["p5"] for x in per], ns, 6),
           "iid_halves": fisher_combine([x["halves"] for x in per], ns, 1)}
    print()
    for key, lab in [("primary", "parcial | val (PRIMARIO)"), ("given_5hp", "parcial | val, 5hp"),
                     ("iid_halves", "metades i.i.d. (deve ser ~0)")]:
        c = out[key]
        if c.get("k", 0) == 0:
            print(f"  {lab:30s} sem dados"); continue
        print(f"  {lab:30s} r = {c['r']:+.3f}  IC95 [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  "
              f"p = {c['p']:.2g}  I2 = {c['I2']:.0%}  (k = {c['k']})")
    c = out["primary"]; lo, hi = c["ci95"]
    v = ("SUSTENTA H_G" if lo > 0 else "CONTRADIZ H_G" if hi < 0 else "NAO SUSTENTA H_G (IC contem zero)")
    het = " -- ATENCAO: folds heterogeneos (I2 > 50%), a media resume mal" if c["I2"] > 0.5 else ""
    print(f"\nVEREDITO COMBINADO: {v}{het}")
    if any(x["dirty"] for x in per):
        print("AVISO: ha' rodadas com arvore suja -- nao reportaveis como resultado oficial.")


if __name__ == "__main__":
    main()
