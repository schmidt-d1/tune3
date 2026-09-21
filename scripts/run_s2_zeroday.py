#!/usr/bin/env python
# scripts/run_s2_zeroday.py
"""
Fase S2 (H3): zero-day por clusterizacao. Leave-one-cluster-out no DREBIN:
treina em K-1 clusters de malware, testa no cluster retido (variante nao vista).
Mede se a vantagem do Tune3 AMPLIFICA em malware fora da distribuicao de treino.

ANALISE PRE-REGISTRADA (set/2026): a unidade estatistica e' a SEED. Para cada
seed, o CVaR de teste e' a MEDIA sobre os K folds; o Wilcoxon pareado roda sobre
N pares (um por seed). A analise "pooled" (K*N pares) e' reportada apenas como
exploratoria -- ela trata como independentes observacoes que compartilham o
mesmo conjunto de teste (pseudo-replicacao).

Uso:
    python scripts/run_s2_zeroday.py --csv data/drebin215.csv \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --n-clusters 5 --epochs 100 --device cuda --tag aluno-nome
    # subconjunto de metodos (ex.: so' E1):
    python scripts/run_s2_zeroday.py ... --methods tune3_bestcvar tune3_placebo_bestcvar bo_mono_cvar
"""
from __future__ import annotations
import argparse, time
import numpy as np

from tune3.data.drebin import DrebinLoader, DrebinConfig
from tune3.data.shift import cluster_holdout_split, n_malware_clusters
from tune3.experiments.security_protocol import run_security_protocol, SecurityProtocolConfig
from tune3.experiments.protocol import ALL_METHODS
from tune3.experiments.stats import compare_paired, summarize, aggregate_per_seed
from tune3.experiments.results_io import save_result


def _report(title, t, base, n_label):
    st = compare_paired(t, base, lower_is_better=True)
    print(f"\n  {title} ({st['n_seeds']} {n_label}; p-min atingivel={st['min_achievable_p']:.4f})")
    for name, c in st["comparisons"].items():
        flag = "VITORIA PRE-REG" if c.get("prereg_win") else ("signif." if c.get("significant") else "n.s.")
        print(f"    vs {name:24s}: d_z={c['dz']:+.2f} IC95[{c['dz_ci95'][0]:+.2f},{c['dz_ci95'][1]:+.2f}] "
              f"p_holm={c.get('p_holm', 1):.4f} [{flag}]")
    return st


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
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    ap.add_argument("--reference", default="tune3_bestcvar", help="metodo de referencia nas comparacoes")
    ap.add_argument("--tag", default=None, help="identificador de quem roda (ex.: aluno-joao)")
    ap.add_argument("--out", default=None, help="(opcional) copia extra do JSON")
    args = ap.parse_args()
    t0 = time.time()

    loader = DrebinLoader(DrebinConfig(csv_path=args.csv, random_state=0))
    Xtr, Xv, Xte, ytr, yv, yte = loader.load_splits()
    X_full = np.concatenate([Xtr, Xv, Xte]); y_full = np.concatenate([ytr, yv, yte])
    sizes = n_malware_clusters(X_full, y_full, args.n_clusters, seed=0)
    print(f"Tamanhos dos clusters de malware (seed 0): {sizes}")

    results_by_fold = {}
    for fold in range(args.n_clusters):
        print(f"\n{'='*60}\n  S2 -- fold {fold} (cluster {fold} = teste zero-day)\n{'='*60}")

        def loader_fn(seed, _fold=fold):
            return cluster_holdout_split(X_full, y_full, args.n_clusters, _fold, seed=seed)

        cfg = SecurityProtocolConfig(seeds=args.seeds, epochs=args.epochs,
                                     n_init=args.n_init, n_iter=args.n_iter,
                                     device=args.device, macro_device=args.macro_device,
                                     methods=list(args.methods))
        res = run_security_protocol(loader_fn, cfg)
        results_by_fold[str(fold)] = res
        bm = res["by_metric"]
        print(f"\n  Resultado (fold {fold}):")
        for metric in ["cvar_test", "test_auc_pr", "test_tpr_at_1fpr"]:
            if metric not in res["metric_keys"]:
                continue
            print(f"    {metric}:")
            for m in res["methods"]:
                s = summarize(bm[m][metric])
                print(f"      {m:24s} mediana={s['median']:.4f}")

    methods = list(results_by_fold["0"]["methods"])
    ref = args.reference if args.reference in methods else methods[0]
    others = [m for m in methods if m != ref]

    # --- ANALISE PRIMARIA: uma observacao por seed (media sobre folds) ---
    print(f"\n{'='*60}\n  PRIMARIA -- CVaR por SEED (media sobre {args.n_clusters} folds)\n{'='*60}")
    per_seed = {m: aggregate_per_seed(results_by_fold, m, "cvar_test") for m in methods}
    for m in methods:
        s = summarize(per_seed[m])
        print(f"  {m:24s}: mediana={s['median']:.4f} media={s['mean']:.4f}+-{s['std']:.4f}")
    st_primary = _report(f"{ref} vs demais", per_seed[ref], {m: per_seed[m] for m in others}, "seeds")

    # --- EXPLORATORIA: pooled (pseudo-replicacao declarada) ---
    pooled = {m: [] for m in methods}
    for fr in results_by_fold.values():
        for m in methods:
            pooled[m].extend(fr["by_metric"][m]["cvar_test"])
    st_pooled = _report(f"[EXPLORATORIA, pooled folds x seeds] {ref} vs demais",
                        pooled[ref], {m: pooled[m] for m in others}, "pontos (nao independentes)")

    save_result({"results_by_fold": results_by_fold,
                 "primary_per_seed": {"scores": per_seed, "stats": st_primary},
                 "exploratory_pooled": {"scores": pooled, "stats": st_pooled},
                 "cluster_sizes_seed0": sizes},
                experiment="s2_zeroday", tag=args.tag, args=vars(args), started_at=t0,
                extra_path=args.out)


if __name__ == "__main__":
    main()
