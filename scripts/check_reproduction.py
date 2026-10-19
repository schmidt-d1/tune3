#!/usr/bin/env python
# scripts/check_reproduction.py
"""
Conferencia de reproducao entre duas rodadas do E0 (C3-B, out/2026): a rodada com --measures deve
reproduzir BIT A BIT a rodada original da mesma semente (Tr(H^2) primaria, CVaRs, epoca de parada),
porque as medidas extras usam geradores proprios e restauram os parametros do modelo.

Para cada dataset presente nos dois JSONs, alinha as linhas por `cfg`, confere os hiperparametros
(iguais) e, para cada campo numerico comparado, conta linhas identicas (==) e a maior diferenca
absoluta e relativa. Codigo de saida 0 se TODOS os campos exigidos forem identicos dentro de --rtol
(padrao 0: bit a bit), 1 caso contrario (ou se faltar dataset/linha).

Uso:
    python scripts/check_reproduction.py results/e0_generalization/<original>.json results/e0_generalization/<com-measures>.json
    python scripts/check_reproduction.py "results/e0_generalization/*krono-s0.json" "results/e0_generalization/*krono-m0.json"
    python scripts/check_reproduction.py A.json B.json --rtol 1e-6      (tolera nao-determinismo de GPU; cada argumento pode ser um glob que case com UM arquivo)
"""
from __future__ import annotations

import argparse, glob, json, sys
import numpy as np

FIELDS_REQUIRED = ["curvature", "cvar_val", "cvar_test", "best_epoch", "epochs_run"]
FIELDS_OPTIONAL = ["cvar_test_novel", "cvar_test_known", "train_loss_final", "reached"]


def compare_dataset(da, db, rtol):
    """Devolve (resumo por campo, ok). Linhas alinhadas por cfg; hiperparametros devem ser iguais."""
    A = {r["cfg"]: r for r in da["rows"]}; B = {r["cfg"]: r for r in db["rows"]}
    common = sorted(set(A) & set(B)); summary = {"n_a": len(A), "n_b": len(B), "n_common": len(common), "fields": {}}
    ok = len(common) == len(A) == len(B) and len(common) > 0
    hp_diff = sum(A[c]["hparams"] != B[c]["hparams"] for c in common); summary["hparams_diff"] = int(hp_diff)
    ok = ok and hp_diff == 0
    for f in FIELDS_REQUIRED + FIELDS_OPTIONAL:
        in_a = all(f in A[c] for c in common); in_b = all(f in B[c] for c in common)
        if not (in_a and in_b):                      # ausente nos DOIS: nao conta; so' num deles: falha se exigido
            summary["fields"][f] = {"present": False, "in_a": in_a, "in_b": in_b}; ok = ok and (f not in FIELDS_REQUIRED or in_a == in_b); continue
        a = np.array([A[c][f] for c in common], float); b = np.array([B[c][f] for c in common], float)
        both_nan = np.isnan(a) & np.isnan(b); eq = (a == b) | both_nan
        d = np.abs(a - b); d[both_nan] = 0.0; rel = d / np.maximum(np.abs(a), 1e-300); rel[both_nan] = 0.0
        finite = np.isfinite(d)
        max_abs = float(np.nanmax(np.where(finite, d, np.nan))) if finite.any() else float("inf")
        max_rel = float(np.nanmax(np.where(finite, rel, np.nan))) if finite.any() else float("inf")
        within = bool(np.all(eq | (finite & (rel <= rtol))))
        summary["fields"][f] = {"present": True, "n_equal": int(eq.sum()), "n": int(len(common)), "max_abs_diff": max_abs,
                                "max_rel_diff": max_rel, "identical": bool(eq.all()), "within_rtol": within}
        if f in FIELDS_REQUIRED:
            ok = ok and within
    summary["ok"] = bool(ok)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Confere se duas rodadas do E0 reproduzem uma a outra (bit a bit por padrao).")
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()
    for k in ("a", "b"):                           # aceita um padrao glob, desde que case com UM arquivo
        hits = sorted(glob.glob(getattr(args, k)))
        if len(hits) != 1:
            print(f"[ERRO] '{getattr(args, k)}' casa com {len(hits)} arquivo(s): {hits}"); sys.exit(2)
        setattr(args, k, hits[0])
    JA = json.load(open(args.a)); JB = json.load(open(args.b))
    DA = {d["dataset"]: d for d in JA["payload"]["datasets"]}; DB = {d["dataset"]: d for d in JB["payload"]["datasets"]}
    sa, sb = JA["meta"].get("args", {}).get("seed"), JB["meta"].get("args", {}).get("seed")
    fa, fb = JA["meta"].get("args", {}).get("cluster_fold"), JB["meta"].get("args", {}).get("cluster_fold")
    print(f"A: {args.a}  (semente {sa}, fold {fa}, commit {JA['meta'].get('git', {}).get('commit_short')})")
    print(f"B: {args.b}  (semente {sb}, fold {fb}, commit {JB['meta'].get('git', {}).get('commit_short')})")
    all_ok = sa == sb and fa == fb and set(DA) == set(DB) and len(DA) > 0
    if not all_ok:
        print("  [ERRO] semente/fold/datasets diferentes entre A e B")
    for name in sorted(set(DA) & set(DB)):
        s = compare_dataset(DA[name], DB[name], args.rtol); all_ok = all_ok and s["ok"]
        print(f"\n== {name}: {s['n_a']} linhas em A, {s['n_b']} em B, {s['n_common']} em comum; hiperparametros diferentes: {s['hparams_diff']}")
        print(f"  {'campo':18s} {'iguais':>9} {'max |dif|':>10} {'max rel':>10}  leitura")
        for f, r in s["fields"].items():
            if not r["present"]:
                where = "nos dois" if not (r["in_a"] or r["in_b"]) else ("so' em A" if not r["in_a"] else "so' em B")
                print(f"  {f:18s} {'(ausente ' + where + ')':>22}"); continue
            lead = "IDENTICO" if r["identical"] else ("dentro da tolerancia" if r["within_rtol"] else "DIFERE")
            print(f"  {f:18s} {r['n_equal']:>4}/{r['n']:<4} {r['max_abs_diff']:>10.3g} {r['max_rel_diff']:>10.3g}  {lead}")
    print(f"\nREPRODUCAO: {'SIM -- bit a bit' if all_ok and args.rtol == 0 else ('SIM -- dentro de rtol=%g' % args.rtol if all_ok else 'NAO')}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
