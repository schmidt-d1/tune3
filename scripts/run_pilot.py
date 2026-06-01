#!/usr/bin/env python
# scripts/run_pilot.py
"""
Piloto estatistico do Tune3 no DREBIN-215: roda Tune3 + 3 baselines sobre N seeds
e aplica o protocolo estatistico (Wilcoxon + Holm + d_z + bootstrap).

Uso:
    python scripts/run_pilot.py --csv data/drebin215.csv --seeds 0 1 2 --epochs 20

Para escalar depois (maquina potente):
    python scripts/run_pilot.py --csv data/drebin215.csv \
        --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 50 --device cuda
"""
from __future__ import annotations

import argparse
import json

from tune3.data.drebin import DrebinLoader, DrebinConfig
from tune3.experiments.protocol import run_protocol, ProtocolConfig
from tune3.experiments.stats import compare_paired, summarize


def main():
    ap = argparse.ArgumentParser(description="Piloto estatistico do Tune3.")
    ap.add_argument("--csv", default="data/drebin215.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="pilot_resultado.json")
    args = ap.parse_args()

    # loader dependente da seed (split estratificado reproduzivel por seed)
    def loader_fn(seed: int):
        loader = DrebinLoader(DrebinConfig(csv_path=args.csv, random_state=seed))
        Xtr, Xv, Xte, ytr, yv, yte = loader.load_splits()
        return (Xtr, ytr, Xv, yv, Xte, yte)

    cfg = ProtocolConfig(seeds=args.seeds, epochs=args.epochs,
                         n_init=args.n_init, n_iter=args.n_iter, device=args.device)

    print(f"[piloto] {len(args.seeds)} seeds, {args.epochs} epocas, device={args.device}")
    print(f"[piloto] metodos: Tune3, RandomSearch, ASHA, SAM\n")

    results = run_protocol(loader_fn, cfg)
    scores = results["cvar_test_by_method"]

    # analise estatistica: Tune3 vs cada baseline
    baselines = {k: v for k, v in scores.items() if k != "tune3"}
    stats = compare_paired(scores["tune3"], baselines, lower_is_better=True)

    # relatorio
    print("\n===== RESULTADO DO PILOTO (CVaR no TESTE, menor=melhor) =====")
    for m in results["methods"]:
        s = summarize(scores[m])
        print(f"  {m:14s}: mediana={s['median']:.4f}  IQR={s['iqr']:.4f}  "
              f"media={s['mean']:.4f}+-{s['std']:.4f}")

    print(f"\n  Seeds: {stats['n_seeds']} | p-minimo atingivel (Wilcoxon): {stats['min_achievable_p']:.4f}")
    if stats["min_achievable_p"] >= 0.05:
        print("  AVISO: com este nº de seeds e' IMPOSSIVEL atingir p<0.05.")
        print("         Este piloto valida o PIPELINE; escale para >=6 (ideal 20) seeds.")

    print("\n  Tune3 vs baselines (pareado):")
    for name, r in stats["comparisons"].items():
        vit = "Tune3 melhor" if r["win"] else "baseline melhor"
        sig = "SIGNIF" if r.get("significant") else "n.s."
        print(f"    vs {name:14s}: diff={r['diff_mean']:+.4f} ({vit}), "
              f"d_z={r['dz']:+.2f}, p_holm={r.get('p_holm', float('nan')):.4f} [{sig}]")

    with open(args.out, "w") as f:
        json.dump({"scores": scores, "stats": stats,
                   "per_seed": results["per_seed"]}, f, indent=2)
    print(f"\n[salvo] {args.out}")


if __name__ == "__main__":
    main()
