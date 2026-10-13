#!/usr/bin/env python
# scripts/e0_measures.py
"""
Cenario 2 (out/2026): a MESMA analise do E0 para cada medida geometrica gravada nas linhas de uma rodada
com --measures (Tr(H^2) primaria, sharpness adaptativa rho=0,1 e 0,5, lambda_max, 2a estimativa de Tr(H^2)),
mais a CONFIABILIDADE do estimador de Tr(H^2).

Para cada medida m (em log10; linhas com m <= 0 ou nao finito sao excluidas e contadas):
  rho bruta com o alvo; parcial | CVaR_val [IC95 bootstrap sobre configuracoes]; | val, lr, wd;
  | val, 5 hiperparametros [IC95]; e, se as perdas por amostra existirem (results/raw/<stem>_<dataset>.npz),
  o IC em DOIS NIVEIS (instancias + configuracoes) da parcial | val.
Confiabilidade: correlacao (Spearman e Pearson em log) entre a Tr(H^2) primaria e a 2a estimativa no mesmo
lote (ruido de sondas) e no 2o lote (ruido de lote); atenuacao maxima esperada da parcial = sqrt(r).

Regra (fixada antes de rodar, "Cenario 3" no pre-registro): le-se cada medida pela regra do E0 -- IC95 da
parcial | val inteiramente > 0 e | val, 5 hp tambem > 0 -> sustenta; so' a primeira -> proxy de
hiperparametros; inteiramente < 0 -> contradiz; senao -> nao sustenta.

Uso:
    python scripts/e0_measures.py "results/e0_generalization/*krono-m0.json" --target cvar_test
    python scripts/e0_measures.py "results/e0_generalization/*drebin-cluster-m0-f*.json"   (alvo padrao: cvar_test_novel)
"""
from __future__ import annotations

import argparse, glob, json, os, time
import numpy as np
from scipy.stats import spearmanr, pearsonr

from tune3.experiments.results_io import save_result
from tune3.experiments.stats import partial_spearman

MEASURES = [("curvature", "log Tr(H2)"), ("sharp_0.1", "log sharp 0.1"), ("sharp_0.5", "log sharp 0.5"),
            ("lmax", "log lambda_max"), ("curv2", "log Tr(H2) 2a est."), ("curvb", "log Tr(H2) 2o lote")]


def cvar_rows(L, gamma=0.95):
    n = L.shape[1]; pos = gamma * (n - 1); lo = int(np.floor(pos)); hi = min(lo + 1, n - 1); w = pos - lo
    P = np.partition(L, [lo, hi], axis=1); q = P[:, lo] * (1.0 - w) + P[:, hi] * w; m = L >= q[:, None]
    return (L * m).sum(1) / m.sum(1)


def valid(r):
    ok = np.isfinite(r["curvature"]) and r["curvature"] > 0 and r["curvature"] != 1e3 and np.isfinite(r["cvar_test"]) and r["cvar_test"] != 1e3
    return ok and (r["reached"] if "reached" in r else r.get("best_epoch", 2) >= 2)


def boot(fn, n, B, rng):
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n); x = fn(i)
        if np.isfinite(x):
            v.append(x)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= B // 2 else [float("nan")] * 2


def verdict(p, ci, ci5):
    if ci[0] > 0 and ci5[0] > 0:
        return "SUSTENTA (alem dos hiperparametros)"
    if ci[0] > 0:
        return "proxy de hiperparametros (some dados os 5 hp)"
    if ci[1] < 0:
        return "CONTRADIZ"
    return "nao sustenta"


def analyze(d, stem, target, B, B2, rng):
    rows = [r for r in d["rows"] if valid(r)]
    if target not in rows[0]:
        target = "cvar_test"
    y = np.array([r[target] for r in rows]); v = np.array([r["cvar_val"] for r in rows]); n = len(rows)
    h = lambda k: np.array([r["hparams"][k] for r in rows], float)
    H = [np.log10(h("learning_rate")), np.log10(h("weight_decay") + 1e-6), h("dropout"), h("hidden_dim"), h("n_layers")]
    npz = f"results/raw/{stem}_{d['dataset']}.npz"; Z = np.load(npz) if os.path.exists(npz) else None
    out = {"target": target, "n": n, "measures": {}, "reliability": {}}
    print(f"\n  alvo = {target}; n = {n} configuracoes validas; perdas por amostra: {'sim' if Z is not None else 'nao'}")
    print(f"  {'medida':20s} {'n_ok':>4} {'rho bruta':>9} {'parcial|val':>11} {'IC95 configs':>16} {'IC95 2 niveis':>16} {'|val,lr,wd':>10} {'|val,5hp':>8} {'IC95':>16}  leitura")
    for key, label in MEASURES:
        if key not in rows[0]:
            continue
        m = np.array([r.get(key, np.nan) for r in rows], float); okm = np.isfinite(m) & (m > 0)
        if okm.sum() < 10:
            print(f"  {label:20s} {int(okm.sum()):>4}  (poucas linhas validas)"); continue
        x = np.log10(m[okm]); yy, vv = y[okm], v[okm]; HH = [c[okm] for c in H]; nn = int(okm.sum())
        raw = float(spearmanr(x, yy)[0]); p1 = partial_spearman(yy, x, [vv]); p2 = partial_spearman(yy, x, [vv, HH[0], HH[1]]); p5 = partial_spearman(yy, x, [vv] + HH)
        ci = boot(lambda i: partial_spearman(yy[i], x[i], [vv[i]]), nn, B, rng)
        ci5 = boot(lambda i: partial_spearman(yy[i], x[i], [vv[i]] + [c[i] for c in HH]), nn, B, rng)
        ci2 = [float("nan")] * 2
        if Z is not None:
            idx = np.array([r["cfg"] for r in rows])[okm]; V = Z["val"][idx].astype(np.float64); T = Z["test"][idx].astype(np.float64)
            mask = {"cvar_test": np.ones(T.shape[1], bool), "cvar_test_novel": Z["test_novel_mask"], "cvar_test_known": ~Z["test_novel_mask"]}[target]
            TT = T[:, mask]; vals = []
            for _ in range(B2):
                vb = cvar_rows(V[:, rng.integers(0, V.shape[1], V.shape[1])]); yb = cvar_rows(TT[:, rng.integers(0, TT.shape[1], TT.shape[1])])
                i = rng.integers(0, nn, nn); vals.append(partial_spearman(yb[i], x[i], [vb[i]]))
            vals = np.array(vals); vals = vals[np.isfinite(vals)]; ci2 = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        lead = verdict(p1, ci2 if np.isfinite(ci2[0]) else ci, ci5)
        out["measures"][key] = {"label": label, "n_ok": nn, "rho_raw": raw, "partial_val": p1, "ci95_configs": ci, "ci95_two_level": ci2,
                                "partial_val_lr_wd": p2, "partial_val_5hp": p5, "ci95_5hp": ci5, "verdict": lead}
        print(f"  {label:20s} {nn:>4} {raw:>+9.3f} {p1:>+11.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}] [{ci2[0]:+.2f}, {ci2[1]:+.2f}] {p2:>+10.3f} {p5:>+8.3f} [{ci5[0]:+.2f}, {ci5[1]:+.2f}]  {lead}")
    # confiabilidade do estimador
    c = np.array([r["curvature"] for r in rows], float)
    for key, label in (("curv2", "mesmo lote (sondas)"), ("curvb", "2o lote (lote + sondas)")):
        if key in rows[0]:
            m = np.array([r.get(key, np.nan) for r in rows], float); ok2 = np.isfinite(m) & (m > 0) & (c > 0)
            rs = float(spearmanr(np.log10(c[ok2]), np.log10(m[ok2]))[0]); rp = float(pearsonr(np.log10(c[ok2]), np.log10(m[ok2]))[0])
            out["reliability"][key] = {"spearman": rs, "pearson_log": rp, "attenuation_sqrt_r": float(np.sqrt(max(rs, 0.0)))}
            print(f"  confiabilidade Tr(H2) {label:24s}: Spearman {rs:+.3f}, Pearson(log) {rp:+.3f} -> atenuacao maxima da parcial x{np.sqrt(max(rs, 0.0)):.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Analise do E0 por medida geometrica (rodadas com --measures).")
    ap.add_argument("files", nargs="+"); ap.add_argument("--target", default="cvar_test_novel", choices=["cvar_test", "cvar_test_novel", "cvar_test_known"])
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--B2", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(args.seed)
    files = sorted({f for pat in args.files for f in glob.glob(pat)}); out = {"runs": []}
    for f in files:
        J = json.load(open(f)); stem = os.path.splitext(os.path.basename(f))[0]
        for d in J["payload"]["datasets"]:
            a_ = J["meta"].get("args", {})
            name = f"{d['dataset']} fold {a_.get('cluster_fold')}" if d["dataset"] == "drebin_cluster" else f"{d['dataset']} semente {a_.get('seed')}"
            print(f"\n== {name} ({stem}) ==")
            target = "cvar_test" if d["dataset"] == "kronodroid" else args.target
            res = analyze(d, stem, target, args.B, args.B2, rng); res["name"] = name; res["src"] = stem; out["runs"].append(res)
    save_result(out, experiment="e0_measures", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
