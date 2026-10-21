#!/usr/bin/env python
# scripts/e1_offline.py
"""
E1-offline (out/2026): o CRITERIO DE ESCOLHA do afinador, com e sem geometria, simulado sobre os pools do E0.

Nao treina nada. Para cada rodada do E0 (pool de configuracoes que atingiram T, com CVaR de validacao, CVaR no alvo
deslocado e medidas de geometria), simula um afinador com orcamento b: sorteia b configuracoes do pool (busca aleatoria;
o MESMO subconjunto para todos os bracos, pareado) e escolhe uma por criterio:
  V      argmin CVaR_val
  V+Tr   argmin [posto(CVaR_val) + w * posto(log Tr(H^2))]       (escalarizacao em postos; menor curvatura = "melhor")
  V+S    idem com log sharpness adaptativa 0,1
  V+lmax idem com log lambda_max
  V+P    idem com um objetivo ALEATORIO (placebo: responde H_M -- "qualquer segundo objetivo ajuda?")
  aleatorio / oraculo: referencias (uma configuracao ao acaso; argmin do alvo no subconjunto)
Metrica: CVaR deslocado da configuracao escolhida, media sobre R subconjuntos; Delta = media(braco) - media(V), pareada.
Incerteza: bootstrap do POOL (reamostra as configuracoes com reposicao, B vezes, R_b subconjuntos cada) -> IC95 de Delta.
Tambem: arrependimento (escolhido - oraculo) e fracao de subconjuntos em que o braco escolhe pior que V.

Regra (pre-registrada em 21/10/2026, "E1-offline" no pre-registro): por cenario, um braco AJUDA se Delta < 0 com IC95
inteiramente abaixo de zero nas 3 sementes (b = 30, w = 1); ATRAPALHA se inteiramente acima nas 3; senao nulo. Se o placebo
tambem "ajudar" no mesmo cenario, o efeito e' de busca (H_M), nao de geometria.

Uso:
    python scripts/e1_offline.py "results/e0_generalization/*-m[0-2]*.json" "results/e0_generalization/*k600-stop-s[12].json" --tag dionatan
"""
from __future__ import annotations

import argparse, glob, json, os, re, time
import numpy as np
from scipy.stats import rankdata, spearmanr

from tune3.experiments.results_io import save_result

MEASURES = [("curvature", "V+Tr"), ("sharp_0.1", "V+S"), ("lmax", "V+lmax")]
ARMS = ["V", "V+Tr", "V+S", "V+lmax", "V+P", "aleatorio", "oraculo"]


def target_of(dataset: str) -> str:
    return "cvar_test" if dataset == "kronodroid" else "cvar_test_novel"


def load_pool(d, args_meta):
    """Pool de uma rodada do E0: configuracoes validas, com val, alvo e as medidas presentes (em log10)."""
    tgt = target_of(d["dataset"])
    if tgt not in d["rows"][0]:
        tgt = "cvar_test"
    keys = [k for k, _ in MEASURES if k in d["rows"][0]]
    rows = []
    for r in d["rows"]:
        ok = r.get("reached", r.get("best_epoch", 2) >= 2) and np.isfinite(r["cvar_val"]) and np.isfinite(r[tgt]) and r[tgt] != 1e3
        ok = ok and all(np.isfinite(r.get(k, np.nan)) and 0 < r.get(k, 0) != 1e3 for k in keys)
        if ok:
            rows.append(r)
    name = (f"{d['dataset']} fold {args_meta.get('cluster_fold')}" if d["dataset"] == "drebin_cluster"
            else f"{d['dataset']} semente {args_meta.get('seed')}")
    return {"name": name, "dataset": d["dataset"], "seed": args_meta.get("seed"),
            "fold": args_meta.get("cluster_fold") if d["dataset"] == "drebin_cluster" else None,
            "target": tgt, "n": len(rows), "val": np.array([r["cvar_val"] for r in rows]), "y": np.array([r[tgt] for r in rows]),
            "measures": {k: np.log10([r[k] for r in rows]) for k in keys}}


def _subsets(n, b, R, rng):
    """R subconjuntos de tamanho b (sem reposicao dentro do subconjunto), como matriz de indices (R, b)."""
    return np.argsort(rng.random((R, n)), axis=1)[:, :b]


def _pick(score):
    return np.argmin(score, axis=1)


def simulate(pool, b, w, R, rng):
    """Alvo da configuracao escolhida por cada braco, em R subconjuntos pareados. Devolve {braco: array (R,)}."""
    n = len(pool["y"]); b = min(b, n)
    idx = _subsets(n, b, R, rng); V = pool["val"][idx]; Y = pool["y"][idx]; rows = np.arange(R)
    rV = rankdata(V, axis=1)
    out = {"V": Y[rows, _pick(rV)], "oraculo": Y.min(axis=1), "aleatorio": Y[rows, rng.integers(0, b, R)]}
    for key, arm in MEASURES:
        if key in pool["measures"]:
            rM = rankdata(pool["measures"][key][idx], axis=1)
            out[arm] = Y[rows, _pick(rV + w * rM + 1e-9 * rV)]          # empate: decide a validacao
    rP = rankdata(rng.random((R, b)), axis=1)
    out["V+P"] = Y[rows, _pick(rV + w * rP + 1e-9 * rV)]
    return out


def summarize(sim):
    base = sim["V"]; res = {}
    for arm, y in sim.items():
        res[arm] = {"mean_target": float(y.mean()), "delta_vs_V": float((y - base).mean()),
                    "regret": float((y - sim["oraculo"]).mean()), "frac_worse_than_V": float((y > base).mean())}
    return res


def bootstrap_deltas(pool, b, w, B, Rb, rng):
    """IC95 de Delta por braco, reamostrando o POOL com reposicao (B vezes), Rb subconjuntos em cada reamostra."""
    n = len(pool["y"]); acc = {arm: [] for arm in ARMS}
    for _ in range(B):
        i = rng.integers(0, n, n)
        p2 = {"val": pool["val"][i], "y": pool["y"][i], "measures": {k: v[i] for k, v in pool["measures"].items()}}
        sim = simulate(p2, b, w, Rb, rng); base = sim["V"]
        for arm, y in sim.items():
            acc[arm].append(float((y - base).mean()))
    return {arm: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for arm, v in acc.items() if v}


def verdict(delta, ci):
    if ci[1] < 0:
        return "ajuda"
    if ci[0] > 0:
        return "atrapalha"
    return "nulo"


def analyze_pool(pool, budgets, weights, R, B, Rb, seed):
    rng = np.random.default_rng(seed)
    out = {"name": pool["name"], "dataset": pool["dataset"], "seed": pool["seed"], "fold": pool["fold"], "target": pool["target"],
           "n_pool": pool["n"], "rho_val_target": float(spearmanr(pool["val"], pool["y"])[0]),
           "rho_measure_target": {k: float(spearmanr(v, pool["y"])[0]) for k, v in pool["measures"].items()},
           "rho_measure_val": {k: float(spearmanr(v, pool["val"])[0]) for k, v in pool["measures"].items()}, "runs": []}
    for b in budgets:
        for w in weights:
            sim = simulate(pool, b, w, R, rng); s = summarize(sim); ci = bootstrap_deltas(pool, b, w, B, Rb, rng)
            for arm in s:
                s[arm]["ci95_delta"] = ci.get(arm, [float("nan")] * 2)
                s[arm]["verdict"] = verdict(s[arm]["delta_vs_V"], s[arm]["ci95_delta"]) if arm not in ("V", "oraculo", "aleatorio") else ""
            out["runs"].append({"budget": int(min(b, pool["n"])), "w": float(w), "arms": s})
    return out


def rule_across_seeds(results, budget, w):
    """Regra 3/3 por cenario (dataset + fold): mesmo veredito em todas as sementes."""
    groups = {}
    for res in results:
        key = f"{res['dataset']} fold {res['fold']}" if res["dataset"] == "drebin_cluster" else res["dataset"]
        groups.setdefault(key, []).append(res)
    table = {}
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: (r["seed"] if r["seed"] is not None else -1))
        table[key] = {"seeds": [r["seed"] for r in rs], "arms": {}}
        for arm in ("V+Tr", "V+S", "V+lmax", "V+P"):
            vs = []
            for r in rs:
                run = [x for x in r["runs"] if x["budget"] == min(budget, r["n_pool"]) and x["w"] == w]
                if run and arm in run[0]["arms"]:
                    vs.append(run[0]["arms"][arm]["verdict"])
            if not vs:
                continue
            k = len(vs)
            lead = ("sem rodadas" if k == 0 else f"{vs[0]} ({k} semente: nao replicado por desenho)" if k == 1
                    else (f"{vs[0]} em {k}/{k}" if len(set(vs)) == 1 else f"misto ({'/'.join(vs)})"))
            table[key]["arms"][arm] = {"verdicts": vs, "rule": lead}
    return table


def main():
    ap = argparse.ArgumentParser(description="E1-offline: criterio de escolha do afinador com e sem geometria, sobre os pools do E0.")
    ap.add_argument("files", nargs="+"); ap.add_argument("--budgets", type=int, nargs="+", default=[10, 30, 100])
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0, 0.5]); ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--B", type=int, default=200); ap.add_argument("--Rb", type=int, default=300)
    ap.add_argument("--primary-budget", type=int, default=30); ap.add_argument("--primary-w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time()
    files = sorted({f for pat in args.files for f in glob.glob(pat)}); results = []
    for f in files:
        J = json.load(open(f)); a_ = J["meta"].get("args", {})
        for d in J["payload"]["datasets"]:
            pool = load_pool(d, a_)
            if pool["n"] < 20:
                print(f"[aviso] {pool['name']}: pool com {pool['n']} configuracoes; pulando"); continue
            res = analyze_pool(pool, args.budgets, args.weights, args.R, args.B, args.Rb, args.seed); res["src"] = os.path.basename(f)
            results.append(res)
            pb = [x for x in res["runs"] if x["budget"] == min(args.primary_budget, pool["n"]) and x["w"] == args.primary_w][0]
            print(f"\n== {pool['name']} ({os.path.splitext(os.path.basename(f))[0]}) ==  alvo {pool['target']}; pool n = {pool['n']}; "
                  f"rho(val, alvo) = {res['rho_val_target']:+.2f}; rho(medida, alvo): " + ", ".join(f"{k} {v:+.2f}" for k, v in res["rho_measure_target"].items()))
            print(f"  b = {pb['budget']}, w = {pb['w']:g}   {'braco':10s} {'alvo medio':>10} {'Delta vs V':>10} {'IC95 (pool)':>16} {'arrepend.':>9} {'pior que V':>10}  leitura")
            for arm in ARMS:
                if arm not in pb["arms"]:
                    continue
                s = pb["arms"][arm]; ci = s["ci95_delta"]
                cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if np.isfinite(ci[0]) else ""
                print(f"  {'':18s} {arm:10s} {s['mean_target']:>10.3f} {s['delta_vs_V']:>+10.3f} {cis:>16} {s['regret']:>9.3f} {s['frac_worse_than_V']:>10.2f}  {s['verdict']}")
            others = [x for x in res["runs"] if not (x["budget"] == pb["budget"] and x["w"] == pb["w"])]
            if others:
                print("  outros (Delta vs V [IC]): " + "; ".join(
                    f"b={x['budget']},w={x['w']:g}: " + ", ".join(f"{arm} {x['arms'][arm]['delta_vs_V']:+.3f} [{x['arms'][arm]['ci95_delta'][0]:+.2f},{x['arms'][arm]['ci95_delta'][1]:+.2f}]"
                                                              for arm in ("V+Tr", "V+S", "V+lmax", "V+P") if arm in x["arms"]) for x in others))
    rule = rule_across_seeds(results, args.primary_budget, args.primary_w)
    print(f"\n== regra por cenario (b = {args.primary_budget}, w = {args.primary_w:g}; mesmo veredito nas 3 sementes) ==")
    for key, t in rule.items():
        print(f"  {key:24s} sementes {t['seeds']}: " + "; ".join(f"{arm}: {v['rule']}" for arm, v in t["arms"].items()))
    save_result({"results": results, "rule": rule, "primary": {"budget": args.primary_budget, "w": args.primary_w}},
                experiment="e1_offline", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
