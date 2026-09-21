#!/usr/bin/env python
# scripts/run_s1_imbalance.py
"""
Fase S1 (H2): roda Tune3 + baselines no DREBIN com razoes de malware {5%,2%,1%}
no TREINO (val/teste mantem composicao natural). Mede se a vantagem do Tune3 em
CVaR e AUC-PR CRESCE conforme o malware fica mais raro.

Uso:
    python scripts/run_s1_imbalance.py --csv data/drebin215.csv \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
        --ratios 0.05 0.02 0.01 --epochs 100 --device cuda --tag aluno-nome
"""
from __future__ import annotations
import argparse, time
import numpy as np

from tune3.data.drebin import DrebinLoader, DrebinConfig
from tune3.data.imbalance import make_imbalanced
from tune3.experiments.security_protocol import run_security_protocol, SecurityProtocolConfig
from tune3.experiments.protocol import ALL_METHODS
from tune3.experiments.stats import compare_paired, summarize
from tune3.experiments.results_io import save_result


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
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    ap.add_argument("--reference", default="tune3_bestcvar")
    ap.add_argument("--tag", default=None, help="identificador de quem roda (ex.: aluno-joao)")
    ap.add_argument("--out", default=None, help="(opcional) copia extra do JSON")
    args = ap.parse_args()
    t0 = time.time()

    results_by_ratio = {}; stats_by_ratio = {}
    for ratio in args.ratios:
        print(f"\n{'='*60}\n  S1 -- razao de malware no treino = {ratio:.0%}\n{'='*60}")

        def loader_fn(seed, _ratio=ratio):
            loader = DrebinLoader(DrebinConfig(csv_path=args.csv, random_state=seed))
            Xtr, Xv, Xte, ytr, yv, yte = loader.load_splits()
            Xtr_i, ytr_i = make_imbalanced(Xtr, ytr, _ratio, rng=np.random.default_rng(seed))
            return (Xtr_i, ytr_i, Xv, yv, Xte, yte)

        cfg = SecurityProtocolConfig(seeds=args.seeds, epochs=args.epochs,
                                     n_init=args.n_init, n_iter=args.n_iter,
                                     device=args.device, macro_device=args.macro_device,
                                     methods=list(args.methods))
        res = run_security_protocol(loader_fn, cfg)
        results_by_ratio[str(ratio)] = res

        bm = res["by_metric"]
        print(f"\n  Resultado (razao {ratio:.0%}):")
        for metric in ["cvar_test", "test_auc_pr", "test_fpr_at_95tpr"]:
            if metric not in res["metric_keys"]:
                continue
            print(f"    {metric}:")
            for m in res["methods"]:
                s = summarize(bm[m][metric])
                print(f"      {m:24s} mediana={s['median']:.4f}")
        ref = args.reference if args.reference in res["methods"] else res["methods"][0]
        t = bm[ref]["cvar_test"]
        base = {k: bm[k]["cvar_test"] for k in res["methods"] if k != ref}
        st = compare_paired(t, base, lower_is_better=True)
        stats_by_ratio[str(ratio)] = st
        print(f"    [CVaR] {ref} vs demais (p_holm; p-min atingivel={st['min_achievable_p']:.4f}):")
        for name, c in st["comparisons"].items():
            flag = "VITORIA PRE-REG" if c.get("prereg_win") else ("signif." if c.get("significant") else "n.s.")
            print(f"      vs {name:24s}: d_z={c['dz']:+.2f} IC95[{c['dz_ci95'][0]:+.2f},{c['dz_ci95'][1]:+.2f}] "
                  f"p_holm={c.get('p_holm', 1):.4f} [{flag}]")

    save_result({"results_by_ratio": results_by_ratio, "stats_by_ratio": stats_by_ratio},
                experiment="s1_imbalance", tag=args.tag, args=vars(args), started_at=t0,
                extra_path=args.out)


if __name__ == "__main__":
    main()
