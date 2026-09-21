#!/usr/bin/env python
# scripts/run_pilot_parallel.py
"""
Versao PARALELA do piloto estatistico (H1) -- extrai mais de uma GPU grande.

Como funciona:
  - Divide as seeds em N_WORKERS grupos
  - Lanca N_WORKERS processos Python em PARALELO, cada um rodando run_pilot.py
    com seu grupo de seeds (compartilham a GPU via contextos CUDA separados)
  - Espera todos terminarem, JUNTA os JSONs e recalcula a estatistica
    pre-registrada (compare_paired) sobre os resultados unidos.

Speedup esperado: 3-4x para o DREBIN-MLP (limitado pelo BoTorch em CPU).

Uso:
    python scripts/run_pilot_parallel.py --csv data/drebin215.csv \
        --seeds 0 1 2 ... 19 --epochs 100 --n-init 8 --n-iter 20 \
        --device cuda --workers 4 --tag aluno-nome
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

from tune3.experiments.protocol import ALL_METHODS
from tune3.experiments.stats import compare_paired, summarize
from tune3.experiments.results_io import save_result


def merge_jsons(files):
    merged_scores, merged_per_seed = {}, []
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"[AVISO] arquivo nao encontrado: {f}"); continue
        with open(p, encoding="utf-8") as fp:
            doc = json.load(fp)
        r = doc.get("payload", doc)             # formato novo {"meta","payload"} ou antigo
        for m, vals in r["scores"].items():
            merged_scores.setdefault(m, []).extend(vals)
        merged_per_seed.extend(r.get("per_seed", []))
    merged_per_seed.sort(key=lambda x: x["seed"])
    return {"scores": merged_scores, "per_seed": merged_per_seed}


def run_block(seeds, args, out_file, worker_id):
    cmd = [sys.executable, "scripts/run_pilot.py", "--csv", args.csv,
           "--seeds", *map(str, seeds), "--epochs", str(args.epochs),
           "--n-init", str(args.n_init), "--n-iter", str(args.n_iter),
           "--device", args.device, "--methods", *args.methods, "--out", out_file]
    if args.tag:
        cmd += ["--tag", f"{args.tag}-w{worker_id}"]
    print(f"[worker {worker_id}] seeds={seeds[0]}-{seeds[-1]} -> {out_file}")
    rc = subprocess.run(cmd).returncode
    print(f"[worker {worker_id}] CONCLUIDO (exit={rc})")
    return rc


def split_seeds(seeds, n_workers):
    size = max(1, -(-len(seeds) // n_workers))     # teto
    return [seeds[i:i + size] for i in range(0, len(seeds), size)]


def main():
    ap = argparse.ArgumentParser(description="Piloto paralelo do Tune3.")
    ap.add_argument("--csv", default="data/drebin215.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    ap.add_argument("--reference", default="tune3_bestcvar")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None, help="(opcional) copia extra do JSON final")
    args = ap.parse_args()
    t0 = time.time()

    groups = split_seeds(args.seeds, args.workers)
    tmp_files = [f"_tmp_worker_{i}.json" for i in range(len(groups))]
    print(f"\n{'='*60}\n  TUNE3 PILOTO PARALELO -- {len(args.seeds)} seeds / {len(groups)} workers\n{'='*60}")
    for i, g in enumerate(groups):
        print(f"  Worker {i}: seeds {g[0]}-{g[-1]} ({len(g)} seeds)")

    threads = [threading.Thread(target=run_block, args=(g, args, f, i))
               for i, (g, f) in enumerate(zip(groups, tmp_files))]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"\n[paralelo] todos workers concluidos em {(time.time()-t0)/60:.1f} minutos")

    merged = merge_jsons(tmp_files)
    scores = merged["scores"]
    ref = args.reference if args.reference in scores else list(scores)[0]
    stats = compare_paired(scores[ref], {k: v for k, v in scores.items() if k != ref},
                           lower_is_better=True)

    print(f"\n{'='*60}\n  RESULTADO FINAL ({stats['n_seeds']} seeds) -- referencia: {ref}\n{'='*60}")
    for m in scores:
        s = summarize(scores[m])
        print(f"  {m:24s}: mediana={s['median']:.4f}  IQR={s['iqr']:.4f}  media={s['mean']:.4f}+-{s['std']:.4f}")
    print(f"\n  p-minimo atingivel: {stats['min_achievable_p']:.6f}")
    for name, r in stats["comparisons"].items():
        flag = "VITORIA PRE-REG" if r.get("prereg_win") else ("signif." if r.get("significant") else "n.s.")
        print(f"    vs {name:24s}: diff={r['diff_mean']:+.4f}, d_z={r['dz']:+.2f} "
              f"IC95[{r['dz_ci95'][0]:+.2f},{r['dz_ci95'][1]:+.2f}], p_holm={r.get('p_holm',1):.5f} [{flag}]")

    save_result({"scores": scores, "stats": stats, "per_seed": merged["per_seed"],
                 "reference": ref, "workers": len(groups)},
                experiment="pilot_h1", tag=args.tag, args=vars(args), started_at=t0,
                extra_path=args.out)
    for f in tmp_files:
        if Path(f).exists():
            os.remove(f)


if __name__ == "__main__":
    main()
