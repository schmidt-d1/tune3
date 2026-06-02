#!/usr/bin/env python
# scripts/run_s2_zeroday.py
"""
Fase S2 (H3): zero-day por clusterizacao. Leave-one-cluster-out no DREBIN:
treina em K-1 clusters de malware, testa no cluster retido (variante nao vista).
Mede se a vantagem do Tune3 AMPLIFICA em malware fora da distribuicao de treino.

Uso:
    python scripts/run_s2_zeroday.py --csv data/drebin215.csv \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --n-clusters 5 --epochs 100 --device cuda --out s2_zeroday.json
"""
from __future__ import annotations
import argparse, json
import numpy as np

from tune3.data.drebin import DrebinLoader, DrebinConfig
from tune3.data.shift import cluster_holdout_split, n_malware_clusters
from tune3.experiments.security_protocol import run_security_protocol, SecurityProtocolConfig
from tune3.experiments.stats import compare_paired, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/drebin215.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-clusters", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--macro-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="s2_zeroday.json")
    args = ap.parse_args()

    # carregar dataset inteiro uma vez (para clusterizar)
    loader = DrebinLoader(DrebinConfig(csv_path=args.csv, random_state=0))
    Xtr, Xv, Xte, ytr, yv, yte = loader.load_splits()
    X_full = np.concatenate([Xtr, Xv, Xte]); y_full = np.concatenate([ytr, yv, yte])
    sizes = n_malware_clusters(X_full, y_full, args.n_clusters, seed=0)
    print(f"Tamanhos dos clusters de malware: {sizes}")

    results_by_fold = {}
    for fold in range(args.n_clusters):
        print(f"\n{'='*60}\n  S2 -- fold {fold} (cluster {fold} = teste zero-day)\n{'='*60}")

        def loader_fn(seed, _fold=fold):
            return cluster_holdout_split(X_full, y_full, args.n_clusters, _fold, seed=seed)

        cfg = SecurityProtocolConfig(seeds=args.seeds, epochs=args.epochs,
                                     n_init=args.n_init, n_iter=args.n_iter,
                                     device=args.device, macro_device=args.macro_device)
        res = run_security_protocol(loader_fn, cfg)
        results_by_fold[str(fold)] = res

        bm = res["by_metric"]
        print(f"\n  Resultado (fold {fold}):")
        for metric in ["cvar_test", "test_auc_pr", "test_tpr_at_1fpr"]:
            print(f"    {metric}:")
            for m in res["methods"]:
                s = summarize(bm[m][metric])
                print(f"      {m:16s} mediana={s['median']:.4f}")

    # agregacao sobre todos os folds (CVaR)
    print(f"\n{'='*60}\n  AGREGADO (todos os folds) -- CVaR\n{'='*60}")
    methods = list(results_by_fold["0"]["methods"])
    agg = {m: [] for m in methods}
    for fold_res in results_by_fold.values():
        for m in methods:
            agg[m].extend(fold_res["by_metric"][m]["cvar_test"])
    for m in methods:
        s = summarize(agg[m])
        print(f"  {m:16s}: mediana={s['median']:.4f} media={s['mean']:.4f}+-{s['std']:.4f}")
    t = agg["tune3_bestcvar"]; base = {k: agg[k] for k in methods if not k.startswith("tune3")}
    st = compare_paired(t, base, lower_is_better=True)
    print(f"\n  tune3_bestcvar vs baselines (agregado, {st['n_seeds']} pontos):")
    for name, c in st["comparisons"].items():
        sig = "SIGNIF" if c.get("significant") else "n.s."
        print(f"    vs {name:14s}: d_z={c['dz']:+.2f}, p_holm={c.get('p_holm',1):.4f} [{sig}]")

    with open(args.out, "w") as f:
        json.dump(results_by_fold, f, indent=2, default=float)
    print(f"\n[salvo] {args.out}")


if __name__ == "__main__":
    main()
