#!/usr/bin/env python
# scripts/run_pilot_parallel.py
"""
Versao PARALELA do piloto estatistico -- extrai muito mais da A100.

Como funciona:
  - Divide as seeds em N_WORKERS grupos
  - Lanca N_WORKERS processos Python em PARALELO, cada um rodando run_pilot.py
    com seu grupo de seeds (compartilham a GPU via contextos CUDA separados)
  - Espera todos terminarem e JUNTA os JSONs automaticamente
  - Calcula a estatistica final sobre os resultados unidos

Speedup esperado: 3-4x numa A100 para o DREBIN-MLP (limitado pelo BoTorch CPU).
Para modelos maiores (CNN/CIFAR): speedup menor pois a GPU fica saturada.

Uso:
    python scripts/run_pilot_parallel.py \
        --csv data/drebin215.csv \
        --seeds 0 1 2 ... 19 \
        --epochs 100 --n-init 8 --n-iter 20 \
        --device cuda --workers 4 \
        --out resultado_paralelo.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ #
#  MERGE de resultados                                                 #
# ------------------------------------------------------------------ #
def merge_jsons(files: list[str]) -> dict:
    """Une varios JSONs de run_pilot.py num unico resultado."""
    merged_scores = {}
    merged_per_seed = []
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"[AVISO] arquivo nao encontrado: {f}")
            continue
        with open(p) as fp:
            r = json.load(fp)
        for m, vals in r["scores"].items():
            merged_scores.setdefault(m, []).extend(vals)
        merged_per_seed.extend(r.get("per_seed", []))
    # ordenar per_seed por seed
    merged_per_seed.sort(key=lambda x: x["seed"])
    return {"scores": merged_scores, "per_seed": merged_per_seed}


def compute_stats(scores: dict) -> dict:
    """Recalcula estatistica Wilcoxon+Holm sobre os scores unidos."""
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    from tune3.experiments.stats import cohens_dz, bootstrap_ci, min_achievable_p, summarize

    methods = list(scores.keys())
    tune3_key = "tune3_bestcvar" if "tune3_bestcvar" in methods else "tune3"
    baselines = {k: v for k, v in scores.items() if not k.startswith("tune3")}
    t = np.array(scores[tune3_key])
    n = len(t)

    raw_p, names, rows = [], list(baselines.keys()), {}
    for name in names:
        b = np.array(baselines[name])
        diff = b - t
        try: _, p = wilcoxon(t, b)
        except: p = 1.0
        raw_p.append(p)
        rows[name] = {"diff_mean": float(diff.mean()), "dz": cohens_dz(diff),
                      "ci95": bootstrap_ci(diff), "p_raw": float(p), "win": bool(diff.mean() > 0)}
    if raw_p:
        rej, pc, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
        for i, name in enumerate(names):
            rows[name]["p_holm"] = float(pc[i]); rows[name]["significant"] = bool(rej[i])
    return {"n_seeds": n, "min_achievable_p": min_achievable_p(n),
            "primary": tune3_key, "comparisons": rows}


# ------------------------------------------------------------------ #
#  RUNNER PARALELO                                                     #
# ------------------------------------------------------------------ #
def run_block(seeds: list[int], args, out_file: str, worker_id: int) -> int:
    """Lanca run_pilot.py para um bloco de seeds como subprocesso."""
    seed_str = " ".join(str(s) for s in seeds)
    cmd = (
        f"python scripts/run_pilot.py "
        f"--csv {args.csv} "
        f"--seeds {seed_str} "
        f"--epochs {args.epochs} "
        f"--n-init {args.n_init} "
        f"--n-iter {args.n_iter} "
        f"--device {args.device} "
        f"--out {out_file}"
    )
    print(f"[worker {worker_id}] seeds={seeds[0]}-{seeds[-1]} -> {out_file}")
    result = subprocess.run(cmd, shell=True)
    print(f"[worker {worker_id}] CONCLUIDO (exit={result.returncode})")
    return result.returncode


def split_seeds(seeds: list[int], n_workers: int) -> list[list[int]]:
    """Divide seeds em N grupos balanceados."""
    size = max(1, len(seeds) // n_workers)
    groups = []
    for i in range(0, len(seeds), size):
        g = seeds[i:i + size]
        if g:
            groups.append(g)
    return groups[:n_workers]


def main():
    ap = argparse.ArgumentParser(description="Piloto paralelo do Tune3.")
    ap.add_argument("--csv", default="data/drebin215.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=4,
                    help="Numero de workers paralelos (4 e' bom para A100+DREBIN)")
    ap.add_argument("--out", default="resultado_paralelo.json")
    args = ap.parse_args()

    groups = split_seeds(args.seeds, args.workers)
    tmp_files = [f"_tmp_worker_{i}.json" for i in range(len(groups))]

    print(f"\n{'='*60}")
    print(f"  TUNE3 PILOTO PARALELO -- {len(args.seeds)} seeds / {len(groups)} workers")
    print(f"  GPU: {args.device} | epochs: {args.epochs} | n-iter: {args.n_iter}")
    print(f"{'='*60}")
    for i, g in enumerate(groups):
        print(f"  Worker {i}: seeds {g[0]}-{g[-1]} ({len(g)} seeds)")
    print()

    # Lanca todos os workers em threads (cada um como subprocesso separado)
    t0 = time.time()
    threads = []
    for i, (group, out_file) in enumerate(zip(groups, tmp_files)):
        t = threading.Thread(target=run_block, args=(group, args, out_file, i))
        t.start(); threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.time() - t0
    print(f"\n[paralelo] todos workers concluidos em {elapsed/60:.1f} minutos")

    # Junta e calcula estatistica
    merged = merge_jsons(tmp_files)
    stats = compute_stats(merged["scores"])
    merged["stats"] = stats

    # Relatorio
    print(f"\n{'='*60}")
    print(f"  RESULTADO FINAL ({stats['n_seeds']} seeds)")
    print(f"  metrica primaria: {stats['primary']}")
    print(f"{'='*60}")

    from tune3.experiments.stats import summarize
    for m in merged["scores"]:
        s = summarize(merged["scores"][m])
        print(f"  {m:16s}: mediana={s['median']:.4f}  IQR={s['iqr']:.4f}  "
              f"media={s['mean']:.4f}+-{s['std']:.4f}")

    print(f"\n  p-minimo atingivel: {stats['min_achievable_p']:.6f}")
    print("\n  Comparacoes:")
    for name, row in stats["comparisons"].items():
        sig = "SIGNIF ✓" if row.get("significant") else "n.s."
        vit = "✅ melhor" if row["win"] else "❌ pior"
        print(f"    vs {name:14s}: diff={row['diff_mean']:+.4f}, "
              f"d_z={row['dz']:+.2f}, p_holm={row.get('p_holm',1):.5f} [{sig}] {vit}")

    with open(args.out, "w") as f:
        json.dump({"scores": merged["scores"], "stats": stats,
                   "per_seed": merged["per_seed"]}, f, indent=2)
    print(f"\n[salvo] {args.out}")

    # Limpa temporarios
    for f in tmp_files:
        if Path(f).exists():
            os.remove(f)


if __name__ == "__main__":
    main()
