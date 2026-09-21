#!/usr/bin/env python
# scripts/run_pilot.py
"""
Piloto estatistico do Tune3 no DREBIN-215 (H1, in-distribution): roda Tune3 +
baselines sobre N seeds e aplica o protocolo estatistico pre-registrado
(Wilcoxon + Holm + d_z + IC BCa + regra |d_z|>=0.30).

Uso (sanidade do pipeline, ~minutos):
    python scripts/run_pilot.py --csv data/drebin215.csv --seeds 0 1 2 --epochs 20 --tag aluno-nome

Para escalar (maquina potente):
    python scripts/run_pilot.py --csv data/drebin215.csv \
        --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 100 --device cuda --tag aluno-nome
"""
from __future__ import annotations

import argparse, time

from tune3.data.drebin import DrebinLoader, DrebinConfig
from tune3.experiments.protocol import run_protocol, ProtocolConfig, ALL_METHODS
from tune3.experiments.stats import compare_paired, summarize
from tune3.experiments.results_io import save_result


def main():
    ap = argparse.ArgumentParser(description="Piloto estatistico do Tune3.")
    ap.add_argument("--csv", default="data/drebin215.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--curvature-every", type=int, default=5,
                    help="cadencia (em epocas) da re-estimativa de Tr(H^2) dentro do trial; "
                         "5 = padrao, 1 = a cada epoca (mais caro). Use para medir o custo dos dois.")
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    ap.add_argument("--reference", default="tune3_bestcvar")
    ap.add_argument("--tag", default=None, help="identificador de quem roda (ex.: aluno-joao)")
    ap.add_argument("--out", default=None, help="(opcional) copia extra do JSON")
    args = ap.parse_args()
    t0 = time.time()

    def loader_fn(seed: int):
        loader = DrebinLoader(DrebinConfig(csv_path=args.csv, random_state=seed))
        Xtr, Xv, Xte, ytr, yv, yte = loader.load_splits()
        return (Xtr, ytr, Xv, yv, Xte, yte)

    cfg = ProtocolConfig(seeds=args.seeds, epochs=args.epochs, n_init=args.n_init,
                         n_iter=args.n_iter, device=args.device, methods=list(args.methods),
                         curvature_every=args.curvature_every)

    print(f"[piloto] {len(args.seeds)} seeds, {args.epochs} epocas, device={args.device}, "
          f"curvature_every={args.curvature_every}")
    print(f"[piloto] metodos: {', '.join(cfg.methods)}\n")

    results = run_protocol(loader_fn, cfg)
    scores = results["cvar_test_by_method"]
    ref = args.reference if args.reference in scores else results["methods"][0]
    baselines = {k: v for k, v in scores.items() if k != ref}
    stats = compare_paired(scores[ref], baselines, lower_is_better=True)

    print("\n===== RESULTADO DO PILOTO (CVaR no TESTE, menor=melhor) =====")
    for m in results["methods"]:
        s = summarize(scores[m])
        print(f"  {m:24s}: mediana={s['median']:.4f}  IQR={s['iqr']:.4f}  "
              f"media={s['mean']:.4f}+-{s['std']:.4f}")

    print(f"\n  Seeds: {stats['n_seeds']} | p-minimo atingivel (Wilcoxon): {stats['min_achievable_p']:.4f}")
    if stats["underpowered"]:
        print("  AVISO: com este nº de seeds e' IMPOSSIVEL atingir p<0.05.")
        print("         Este piloto valida o PIPELINE; escale para >=6 (protocolo: 20) seeds.")

    print(f"\n  {ref} vs demais (pareado; regra pre-registrada = p_holm<0.05 E |d_z|>=0.30):")
    for name, r in stats["comparisons"].items():
        vit = f"{ref} melhor" if r["win"] else "baseline melhor"
        flag = "VITORIA PRE-REG" if r.get("prereg_win") else ("signif." if r.get("significant") else "n.s.")
        print(f"    vs {name:24s}: diff={r['diff_mean']:+.4f} ({vit}), d_z={r['dz']:+.2f} "
              f"IC95[{r['dz_ci95'][0]:+.2f},{r['dz_ci95'][1]:+.2f}], p_holm={r.get('p_holm', float('nan')):.4f} [{flag}]")

    save_result({"scores": scores, "stats": stats, "per_seed": results["per_seed"],
                 "methods": results["methods"], "reference": ref},
                experiment="pilot_h1", tag=args.tag, args=vars(args), started_at=t0,
                extra_path=args.out)


if __name__ == "__main__":
    main()
