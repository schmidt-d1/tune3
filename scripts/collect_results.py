#!/usr/bin/env python
# scripts/collect_results.py
"""
Consolida todos os JSONs de results/ numa tabela para conferencia do coordenador.

    python scripts/collect_results.py                 # tabela no terminal + results/SUMMARY.md
    python scripts/collect_results.py --experiment s2_zeroday
    python scripts/collect_results.py --csv results/summary.csv

Para cada arquivo mostra: experimento, quem rodou (tag), data, commit (e se a
arvore estava suja), GPU, seeds, metodos, e as comparacoes pre-registradas
(d_z, p_holm, vitoria pre-registrada) do metodo de referencia contra cada baseline.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

from tune3.experiments.results_io import load_results


def _stats_block(payload: dict) -> dict:
    """Localiza o bloco de estatistica primaria em qualquer dos formatos de script."""
    if "stats" in payload:                                  # pilot
        return payload["stats"]
    if "primary_per_seed" in payload:                       # s2
        return payload["primary_per_seed"]["stats"]
    if "stats_by_ratio" in payload:                         # s1: pega a razao mais severa
        k = sorted(payload["stats_by_ratio"], key=float)[0]
        return payload["stats_by_ratio"][k]
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--dir", default="results")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default="results/SUMMARY.md")
    args = ap.parse_args()

    rows = []
    for path, doc in load_results(args.experiment, args.dir):
        meta = doc.get("meta", {}); payload = doc.get("payload", doc)
        st = _stats_block(payload)
        base = {"file": path, "experiment": meta.get("experiment", Path(path).parent.name),
                "tag": meta.get("tag"), "timestamp": meta.get("timestamp"),
                "commit": (meta.get("git") or {}).get("commit_short"),
                "dirty": (meta.get("git") or {}).get("dirty"),
                "gpu": (meta.get("env") or {}).get("gpu"),
                "torch": (meta.get("env") or {}).get("torch"),
                "n_seeds": st.get("n_seeds"), "duration_min": round((meta.get("duration_s") or 0) / 60, 1)}
        comps = st.get("comparisons", {})
        if not comps:
            rows.append({**base, "baseline": None})
        for name, c in comps.items():
            rows.append({**base, "baseline": name, "dz": round(c.get("dz", float("nan")), 3),
                         "p_holm": round(c.get("p_holm", float("nan")), 4),
                         "prereg_win": c.get("prereg_win"), "significant": c.get("significant")})

    if not rows:
        print("nenhum resultado em", args.dir); return
    cols = ["experiment", "tag", "timestamp", "commit", "dirty", "gpu", "n_seeds",
            "duration_min", "baseline", "dz", "p_holm", "prereg_win", "file"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    print(line); print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write("# Resumo dos resultados (gerado por scripts/collect_results.py)\n\n")
        f.write("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    print(f"\n[salvo] {args.md}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        print(f"[salvo] {args.csv}")


if __name__ == "__main__":
    main()
