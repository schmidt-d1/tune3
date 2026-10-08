#!/usr/bin/env python
# scripts/f1_val_size.py
"""
F1 -- escassez de validacao (pre-registrado em claude/TUNE3_PREREGISTRO_FASE_DIRECAO.md, "Fase F").
Nao treina nada: le resultados existentes e as perdas por amostra em results/raw/.

Fontes aceitas (podem ser combinadas numa chamada):
  --d2 JSON --d2-tag TAG          grade da D2 (CIFAR); alvos: CVaR no CIFAR-10.1 (deslocado) e CVaR no
                                  teste CIFAR-10 (i.i.d., referencia); medidas: log Tr(H^2) (primaria),
                                  log sharpness adaptativa 0,5 e 0,1
  --e0 JSON [JSON ...]            rodadas do E0 com --save-losses (ex.: os 5 folds do DREBIN por cluster;
                                  NSL-KDD apos a F2); alvo: CVaR nos ataques/malware NOVOS; medida: log Tr(H^2).
                                  Folds do mesmo dataset sao combinados por efeitos aleatorios (Fisher z).

Uso:
    python scripts/f1_val_size.py --d2 results/d2_cifar/20261007-211209_995071b_dionatan-grid.json --d2-tag dionatan-grid \\
        --e0 "results/e0_generalization/*dionatan-drebin-cluster-f*.json" --tag dionatan
"""
from __future__ import annotations

import argparse, glob, json, os, time
import numpy as np

from tune3.experiments.results_io import save_result
from tune3.experiments.stats import fisher_combine
from tune3.experiments.val_scarcity import scan, format_scan, scan_verdict, SIZES_DEFAULT

HP = ["learning_rate", "weight_decay", "dropout", "hidden_dim", "n_layers"]


def hp_arrays(rows):
    h = lambda k: np.array([r["hparams"][k] for r in rows], float)
    d = {"log_lr": np.log10(h("learning_rate")), "log_wd": np.log10(h("weight_decay") + 1e-6),
         "dropout": h("dropout"), "width": h("hidden_dim"), "depth": h("n_layers")}
    return d, [d["depth"], d["width"]]


def run_d2(path, tag, args):
    J = json.load(open(path)); rows = [r for r in J["payload"]["rows"] if r["reached"]]
    raw = os.path.join("results", "raw", f"d2_{tag}_grid")
    Z = [np.load(os.path.join(raw, f"cfg_{r['cfg']:03d}.npz")) for r in rows]
    V = np.stack([z["val"].astype(np.float64) for z in Z]); T = np.stack([z["test"].astype(np.float64) for z in Z])
    n10 = T.shape[1] - 2000
    from tune3.experiments.val_scarcity import cvar_rows
    targets = {"cvar_c101 (deslocado)": cvar_rows(T[:, n10:]), "cvar_c10 (i.i.d.)": cvar_rows(T[:, :n10])}
    measures = {"log Tr(H2)": np.log10([r["curvature"] for r in rows]),
                "log sharp 0.5": np.log10([r["sharp_0.5"] for r in rows]),
                "log sharp 0.1": np.log10([r["sharp_0.1"] for r in rows])}
    hp, arch = hp_arrays(rows)
    out = {"source": os.path.basename(path), "n": len(rows), "results": {}}
    print(f"\n== D2 / CIFAR: {len(rows)} modelos ({os.path.basename(path)}) ==")
    for stat in args.stats:
        for tname, y in targets.items():
            for mname, x in measures.items():
                res = scan(V, y, x, arch=arch, hparams=hp, sizes=args.sizes, stat=stat, R=args.R, B=args.B, seed=args.seed)
                key = f"{stat} :: {tname} :: {mname}"; out["results"][key] = res
                print(format_scan(res, f"CIFAR | alvo {tname} | medida {mname}"))
    return out


def run_e0(paths, args):
    from tune3.experiments.val_scarcity import cvar_rows
    by_ds = {}
    for p in paths:
        J = json.load(open(p)); a = J["meta"].get("args", {})
        for d in J["payload"]["datasets"]:
            stem = os.path.splitext(os.path.basename(p))[0]
            npz = os.path.join("results", "raw", f"{stem}_{d['dataset']}.npz")
            if not os.path.exists(npz):
                print(f"[aviso] {stem}/{d['dataset']}: sem perdas por amostra ({npz}); pulando"); continue
            rows = [r for r in d["rows"] if r.get("reached", True) and np.isfinite(r["curvature"]) and 0 < r["curvature"] != 1e3]
            Z = np.load(npz); idx = np.array([r["cfg"] for r in rows])
            V = Z["val"][idx].astype(np.float64); T = Z["test"][idx].astype(np.float64); nov = Z["test_novel_mask"]
            y = cvar_rows(T[:, nov]) if nov.any() else cvar_rows(T)
            hp, arch = hp_arrays(rows)
            by_ds.setdefault(d["dataset"], []).append({"fold": a.get("cluster_fold"), "src": stem, "n": len(rows), "V": V, "y": y,
                                                       "x": np.log10([r["curvature"] for r in rows]), "hp": hp, "arch": arch})
    out = {}
    for ds, runs in by_ds.items():
        print(f"\n== E0 / {ds}: {len(runs)} rodada(s), alvo = CVaR nos {'ataques/malware NOVOS'} ==")
        out[ds] = {"runs": [], "combined": {}}
        for stat in args.stats:
            per = []
            for R_ in runs:
                res = scan(R_["V"], R_["y"], R_["x"], arch=R_["arch"], hparams=R_["hp"], sizes=args.sizes, stat=stat,
                           R=args.R, B=args.B, seed=args.seed)
                per.append(res); out[ds]["runs"].append({"fold": R_["fold"], "src": R_["src"], "stat": stat, "scan": res})
                print(format_scan(res, f"{ds} fold {R_['fold']} | {stat} | log Tr(H2)"))
            if len(per) > 1:                      # combinacao por tamanho (efeitos aleatorios), como na D1
                comb = {"stat": stat, "n_models": sum(R_["n"] for R_ in runs), "n_val": per[0]["n_val"], "R": args.R, "B": args.B,
                        "sizes": per[0]["sizes"], "per_size": []}
                for k, nv in enumerate(per[0]["sizes"]):
                    rs = [p["per_size"][k]["P"] for p in per]; ns = [R_["n"] for R_ in runs]
                    fc = fisher_combine(rs, ns, 1)
                    row = {"n_val": nv, "P": fc["r_re"], "P_ci95": fc["ci95_re"], "I2": fc["I2"],
                           "rho_val_target": float(np.mean([p["per_size"][k]["rho_val_target"] for p in per])),
                           "hparam_partial_given_val": {h: float(np.mean([p["per_size"][k]["hparam_partial_given_val"][h] for p in per]))
                                                        for h in per[0]["per_size"][k]["hparam_partial_given_val"]}}
                    if "P_arch" in per[0]["per_size"][k]:
                        fa = fisher_combine([p["per_size"][k]["P_arch"] for p in per], ns, 3)
                        row["P_arch"] = fa["r_re"]; row["P_arch_ci95"] = fa["ci95_re"]
                    comb["per_size"].append(row)
                comb["verdict"] = scan_verdict(comb); out[ds]["combined"][stat] = comb
                print(format_scan(comb, f"{ds} COMBINADO (efeitos aleatorios, {len(per)} folds) | {stat} | log Tr(H2)"))
                print("     I2 por tamanho: " + "  ".join(f"{r['n_val']}:{r['I2']:.0%}" for r in comb["per_size"]))
    return out


def main():
    ap = argparse.ArgumentParser(description="F1: a curvatura acrescenta informacao quando a validacao e' escassa?")
    ap.add_argument("--d2", default=None); ap.add_argument("--d2-tag", default="dionatan-grid")
    ap.add_argument("--e0", nargs="*", default=[])
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES_DEFAULT))
    ap.add_argument("--stats", nargs="+", default=["cvar", "mean"], choices=["cvar", "mean"])
    ap.add_argument("--R", type=int, default=200); ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time()
    out = {"sizes": args.sizes, "stats": args.stats}
    if args.d2:
        out["d2"] = run_d2(args.d2, args.d2_tag, args)
    files = sorted({f for pat in args.e0 for f in glob.glob(pat)})
    if files:
        out["e0"] = run_e0(files, args)
    print("\nREGRA PRE-REGISTRADA (estatistica CVaR, alvo deslocado, medida log Tr(H2)):")
    if "d2" in out:
        k = "cvar :: cvar_c101 (deslocado) :: log Tr(H2)"
        print("  CIFAR-10.1:", out["d2"]["results"][k]["verdict"])
    for ds, v in out.get("e0", {}).items():
        c = v["combined"].get("cvar") or (v["runs"][0]["scan"] if v["runs"] else None)
        if c:
            print(f"  {ds}:", c["verdict"])
    save_result(out, experiment="f1_val_size", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
