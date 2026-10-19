#!/usr/bin/env python
# scripts/measures_replication.py
"""
C3-B2 (out/2026): replicacao POR MEDIDA entre sementes, a partir dos JSONs do e0_measures.py.

Le os JSONs do e0_measures (um por chamada; cada um com uma ou mais rodadas do E0), descobre a semente e o
fold de cada rodada no JSON de origem do E0 (results/e0_generalization/<src>.json) e agrupa por CENARIO
(nslkdd; kronodroid; drebin_cluster fold k). Para cada cenario x medida:
  - por semente: parcial | val [IC95 em dois niveis, ou por configuracoes se nao houver perdas gravadas],
    parcial | val, 5 hp [IC95 por configuracoes] e o veredito do e0_measures (S = sustenta, P = proxy de
    hiperparametros, C = contradiz, 0 = nao sustenta);
  - combinacao das sementes em z de Fisher (efeito fixo; peso n - 3 - controles) das duas parciais, com IC95
    e I^2 -- resumo descritivo, nao substitui a regra por semente;
  - REGRA (fixada em 14/10/2026, antes das sementes 1 e 2 com medidas): um veredito so' conta como
    replicado se for o MESMO em todas as sementes (3/3); qualquer mistura e' "NAO replicado". Tambem
    contados: sementes com | val, 5hp acima de zero (padrao de supressao quando a parcial | val nao e' > 0).
Para o DREBIN por cluster, imprime ainda o tau de Kendall entre sementes da ORDEM dos folds (parcial | val),
por medida -- a heterogeneidade entre familias e' reproduzivel?

Uso:
    python scripts/measures_replication.py "results/e0_measures/*krono-m*.json" "results/e0_measures/*drebin-m*.json" \\
        "results/e0_measures/*nsl-m*.json" --tag dionatan
"""
from __future__ import annotations

import argparse, glob, json, os, re, time
from itertools import combinations
import numpy as np
from scipy.stats import kendalltau

from tune3.experiments.results_io import save_result
from tune3.experiments.stats import fisher_combine

MEASURES = ["curvature", "sharp_0.1", "sharp_0.5", "lmax", "curv2", "curvb"]
CONTROLS_5HP = 5


def code(verdict: str) -> str:
    v = (verdict or "").lower()
    if v.startswith("sustenta"):
        return "S"
    if v.startswith("proxy"):
        return "P"
    if v.startswith("contradiz"):
        return "C"
    return "0"


def resolve_seed_fold(run, e0_dir):
    """Semente e fold da rodada: do JSON de origem do E0; senao, do nome/tag."""
    src = run.get("src", ""); name = run.get("name", "")
    p = os.path.join(e0_dir, src + ".json")
    if os.path.exists(p):
        a = json.load(open(p))["meta"].get("args", {})
        seed = a.get("seed"); fold = a.get("cluster_fold") if name.startswith("drebin_cluster") else None
        if seed is not None:
            return int(seed), (int(fold) if fold is not None else None)
    m = re.search(r"-[ms](\d+)(?:-f\d+)?$", src); seed = int(m.group(1)) if m else None
    if seed is None:
        m = re.search(r"semente (\d+)", name); seed = int(m.group(1)) if m else None
    m = re.search(r"fold (\d+)", name); fold = int(m.group(1)) if m else None
    return seed, fold


def load_runs(files, e0_dir):
    runs = []
    for f in sorted(files):
        J = json.load(open(f))
        for r in J["payload"]["runs"]:
            seed, fold = resolve_seed_fold(r, e0_dir)
            ds = r["name"].split(" ")[0]
            scen = f"{ds} fold {fold}" if ds == "drebin_cluster" else ds
            runs.append({"scenario": scen, "dataset": ds, "fold": fold, "seed": seed, "src": r.get("src"), "file": os.path.basename(f),
                         "target": r.get("target"), "measures": r["measures"]})
    return runs


def ci_of(m):
    ci2 = m.get("ci95_two_level", [float("nan")] * 2)
    return ci2 if (ci2 and np.isfinite(ci2[0])) else m["ci95_configs"]


def rule(codes):
    k = len(codes)
    if k == 0:
        return "sem rodadas"
    if k == 1:
        return f"{codes[0]} (1 semente: nao replicado por desenho)"
    if len(set(codes)) == 1:
        return f"{codes[0]} em {k}/{k} (replicado)"
    return f"NAO replicado ({'/'.join(codes)})"


def analyze(runs):
    scen_names = []
    for r in runs:
        if r["scenario"] not in scen_names:
            scen_names.append(r["scenario"])
    out = []
    for scen in scen_names:
        rs = sorted([r for r in runs if r["scenario"] == scen], key=lambda r: (r["seed"] if r["seed"] is not None else -1))
        seeds = [r["seed"] for r in rs]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"[erro] sementes repetidas no cenario {scen}: {seeds} (arquivos: {[r['file'] for r in rs]})")
        entry = {"scenario": scen, "seeds": seeds, "files": [r["file"] for r in rs], "measures": {}}
        for key in MEASURES:
            per = []
            for r in rs:
                m = r["measures"].get(key)
                if m is None:
                    continue
                per.append({"seed": r["seed"], "n_ok": m["n_ok"], "partial_val": m["partial_val"], "ci_val": ci_of(m),
                            "partial_5hp": m["partial_val_5hp"], "ci_5hp": m["ci95_5hp"], "verdict": m["verdict"], "code": code(m["verdict"])})
            if not per:
                continue
            ns = [p["n_ok"] for p in per]
            fe = fisher_combine([p["partial_val"] for p in per], ns, 1); f5 = fisher_combine([p["partial_5hp"] for p in per], ns, 1 + CONTROLS_5HP)
            codes = [p["code"] for p in per]
            entry["measures"][key] = {"per_seed": per, "codes": codes, "rule": rule(codes),
                                      "n_val_above0": int(sum(p["ci_val"][0] > 0 for p in per)), "n_val_below0": int(sum(p["ci_val"][1] < 0 for p in per)),
                                      "n_5hp_above0": int(sum(p["ci_5hp"][0] > 0 for p in per)), "k": len(per),
                                      "fe_val": fe, "fe_5hp": f5}
        out.append(entry)
    return out


def fold_ordering(runs):
    """DREBIN por cluster: tau de Kendall entre pares de sementes da ordem dos folds (parcial | val), por medida."""
    dre = [r for r in runs if r["dataset"] == "drebin_cluster" and r["seed"] is not None and r["fold"] is not None]
    seeds = sorted({r["seed"] for r in dre}); res = {}
    for key in MEASURES:
        by_seed = {}
        for s in seeds:
            v = {r["fold"]: r["measures"][key]["partial_val"] for r in dre if r["seed"] == s and key in r["measures"]}
            if len(v) >= 3:
                by_seed[s] = v
        taus = {}
        for a, b in combinations(sorted(by_seed), 2):
            folds = sorted(set(by_seed[a]) & set(by_seed[b]))
            if len(folds) >= 3:
                taus[f"{a}-{b}"] = float(kendalltau([by_seed[a][f] for f in folds], [by_seed[b][f] for f in folds])[0])
        if taus:
            res[key] = {"tau": taus, "order_by_seed": {str(s): sorted(by_seed[s], key=lambda f: -by_seed[s][f]) for s in by_seed}}
    return res


def main():
    ap = argparse.ArgumentParser(description="C3-B2: replicacao por medida entre sementes (JSONs do e0_measures).")
    ap.add_argument("files", nargs="+"); ap.add_argument("--e0-dir", default="results/e0_generalization"); ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time()
    files = sorted({f for pat in args.files for f in glob.glob(pat)})
    runs = load_runs(files, args.e0_dir)
    print(f"[measures_replication] {len(files)} arquivos do e0_measures, {len(runs)} rodadas do E0")
    scen = analyze(runs)
    for e in scen:
        print(f"\n== {e['scenario']} -- sementes {e['seeds']} ==")
        print(f"  {'medida':10s} {'sem.':>4} {'n':>4} {'parcial|val':>11} {'IC95':>16} {'|val,5hp':>9} {'IC95':>16}  cod")
        for key, m in e["measures"].items():
            for p in m["per_seed"]:
                print(f"  {key:10s} {str(p['seed']):>4} {p['n_ok']:>4} {p['partial_val']:>+11.3f} [{p['ci_val'][0]:+.2f}, {p['ci_val'][1]:+.2f}] "
                      f"{p['partial_5hp']:>+9.3f} [{p['ci_5hp'][0]:+.2f}, {p['ci_5hp'][1]:+.2f}]  {p['code']}")
            fe, f5 = m["fe_val"], m["fe_5hp"]
            print(f"  {'':10s} {'FE':>4} {'':>4} {fe['r']:>+11.3f} [{fe['ci95'][0]:+.2f}, {fe['ci95'][1]:+.2f}] "
                  f"{f5['r']:>+9.3f} [{f5['ci95'][0]:+.2f}, {f5['ci95'][1]:+.2f}]  I2 {fe['I2']:.0%}/{f5['I2']:.0%}"
                  f"  -> {m['rule']}; |val>0 em {m['n_val_above0']}/{m['k']}, |val<0 em {m['n_val_below0']}/{m['k']}, 5hp>0 em {m['n_5hp_above0']}/{m['k']}")
    tau = fold_ordering(runs)
    if tau:
        print("\n== DREBIN por cluster: ordem dos folds (parcial | val) entre sementes, tau de Kendall ==")
        for key, t in tau.items():
            print(f"  {key:10s} " + "  ".join(f"s{k}: {v:+.2f}" for k, v in t["tau"].items()) + "   ordem: " +
                  "; ".join(f"s{s}: " + ">".join(f"f{f}" for f in o) for s, o in t["order_by_seed"].items()))
    print("\n== resumo (cenario x medida; S sustenta, P proxy, C contradiz, 0 nao sustenta; por semente -> regra) ==")
    for e in scen:
        print(f"  {e['scenario']:24s} " + "  ".join(f"{key}={''.join(m['codes'])}" for key, m in e["measures"].items()))
    save_result({"scenarios": scen, "fold_ordering_tau": tau, "runs": [{k: v for k, v in r.items() if k != 'measures'} for r in runs]},
                experiment="measures_replication", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
