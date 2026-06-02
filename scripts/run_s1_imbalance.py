#!/usr/bin/env python
# scripts/run_s1_imbalance.py
"""
Fase S1 (H2): roda Tune3 + baselines no DREBIN com razoes de malware {5%,2%,1%}
no TREINO (val/teste mantem composicao natural). Mede se a vantagem do Tune3 em
CVaR e AUC-PR CRESCE conforme o malware fica mais raro.

Uso:
    python scripts/run_s1_imbalance.py --csv data/drebin215.csv \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
        --ratios 0.05 0.02 0.01 --epochs 100 --device cuda \
        --out s1_imbalance.json
"""
from __future__ import annotations
import argparse, json
import numpy as np

from tune3.data.drebin import DrebinLoader, DrebinConfig
from tune3.data.imbalance import make_imbalanced
from tune3.experiments.security_protocol import run_security_protocol, SecurityProtocolConfig
from tune3.experiments.stats import compare_paired, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/drebin215.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.02, 0.01])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--macro-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="s1_imbalance.json")
    args = ap.parse_args()

    results_by_ratio = {}
    for ratio in args.ratios:
        print(f"\n{'='*60}\n  S1 -- razao de malware no treino = {ratio:.0%}\n{'='*60}")

        def loader_fn(seed, _ratio=ratio):
            loader = DrebinLoader(DrebinConfig(csv_path=args.csv, random_state=seed))
            Xtr, Xv, Xte, ytr, yv, yte = loader.load_splits()
            Xtr_i, ytr_i = make_imbalanced(Xtr, ytr, _ratio, rng=np.random.default_rng(seed))
            return (Xtr_i, ytr_i, Xv, yv, Xte, yte)

        cfg = SecurityProtocolConfig(seeds=args.seeds, epochs=args.epochs,
                                     n_init=args.n_init, n_iter=args.n_iter,
                                     device=args.device, macro_device=args.macro_device)
        res = run_security_protocol(loader_fn, cfg)
        results_by_ratio[str(ratio)] = res

        # resumo da razao
        bm = res["by_metric"]
        print(f"\n  Resultado (razao {ratio:.0%}):")
        for metric in ["cvar_test", "test_auc_pr", "test_fpr_at_95tpr"]:
            print(f"    {metric}:")
            for m in res["methods"]:
                s = summarize(bm[m][metric])
                print(f"      {m:16s} mediana={s['median']:.4f}")
        # significancia tune3_bestcvar vs baselines no CVaR
        t = bm["tune3_bestcvar"]["cvar_test"]
        base = {k: bm[k]["cvar_test"] for k in res["methods"] if not k.startswith("tune3")}
        st = compare_paired(t, base, lower_is_better=True)
        print(f"    [CVaR] tune3_bestcvar vs baselines (p_holm):")
        for name, c in st["comparisons"].items():
            print(f"      vs {name:14s}: d_z={c['dz']:+.2f}, p_holm={c.get('p_holm',1):.4f}")

    with open(args.out, "w") as f:
        json.dump(results_by_ratio, f, indent=2, default=float)
    print(f"\n[salvo] {args.out}")


if __name__ == "__main__":
    main()
