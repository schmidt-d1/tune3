#!/usr/bin/env python
# scripts/d2_followup.py
"""
Analises complementares da D2 (CIFAR-10), feitas DEPOIS de ver o resultado da grade -- exploratorias,
e rotuladas assim. Leem o JSON da grade e as perdas por amostra em results/raw/d2_<tag>_grid/ (nao
treinam nada). Existem para que os numeros citados na interpretacao da D2 fiquem rastreaveis.

  (1) Referencias para a analise (A): tau de Kendall com o gap de medidas triviais -- o proprio erro de
      validacao e hiperparametros isolados -- ao lado de Tr(H^2) e da sharpness adaptativa.
  (2) "Folga" da analise (B): quanto do CVaR no alvo NAO e' explicado pela validacao (e arquitetura) e
      e' REPRODUTIVEL. Duas metades independentes (validacao V1/V2 e alvo T1/T2); residuo de T1 dado V1
      contra residuo de T2 dado V2. Correlacao ~0 => o que sobra alem da validacao e' ruido, e nenhuma
      medida poderia ter parcial alta ali (o teste (B) nao tinha como discriminar).
  (3) O que carrega informacao alem da validacao no alvo deslocado: cada hiperparametro e cada medida,
      no CVaR (controle: CVaR val + arquitetura) e no erro (controle: erro val + arquitetura).
  (4) A curvatura e' so' um indicador de outro fator? Parciais da curvatura controlando tambem o lr e a
      profundidade -- o padrao visto no NSL-KDD e no DREBIN por cluster (D1).

Uso:
    python scripts/d2_followup.py results/d2_cifar/<arquivo-da-grade>.json --tag dionatan-grid
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
from scipy.stats import rankdata, spearmanr

from tune3.experiments.generalization_measures import kendall_tau
from tune3.experiments.results_io import save_result
from tune3.experiments.stats import partial_spearman


def cvar_rows(L, g=0.95):
    q = np.quantile(L, g, axis=1, keepdims=True); m = L >= q
    return (L * m).sum(1) / m.sum(1)


def resid(y, controls):
    X = np.column_stack([np.ones(len(y))] + [rankdata(c) for c in controls])
    ry = rankdata(y); b, *_ = np.linalg.lstsq(X, ry, rcond=None)
    return ry - X @ b


def boot(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    return [float(x) for x in np.percentile([fn(rng.integers(0, n, n)) for _ in range(B)], [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser(description="D2: analises complementares (exploratorias).")
    ap.add_argument("grid_json")
    ap.add_argument("--tag", default="dionatan-grid", help="tag da rodada da grade (localiza results/raw/d2_<tag>_grid)")
    ap.add_argument("--splits", type=int, default=60)
    args = ap.parse_args(); t0 = time.time()

    J = json.load(open(args.grid_json))
    rows = [r for r in J["payload"]["rows"] if r["reached"]]
    raw = os.path.join("results", "raw", f"d2_{args.tag}_grid")
    Z = [np.load(os.path.join(raw, f"cfg_{r['cfg']:03d}.npz")) for r in rows]
    V = np.stack([z["val"].astype(np.float64) for z in Z]); T = np.stack([z["test"].astype(np.float64) for z in Z])
    n_c10 = T.shape[1] - 2000
    T101, T10 = T[:, n_c10:], T[:, :n_c10]
    n = len(rows)
    A = lambda k: np.array([r[k] for r in rows], float)
    H = lambda k: np.array([r["hparams"][k] for r in rows], float)
    dep, wid, lr, wd, do = H("n_layers"), H("hidden_dim"), np.log10(H("learning_rate")), H("weight_decay"), H("dropout")
    lc, s1, s5 = np.log10(A("curvature")), np.log10(A("sharp_0.1")), np.log10(A("sharp_0.5"))
    gap, v, y101, c10 = A("gap_error"), A("cvar_val"), A("cvar_c101"), A("cvar_c10")
    ev, e101, e10 = A("val_error"), A("test_error_c101"), A("test_error_c10")
    out = {"n": n, "source": os.path.basename(args.grid_json)}
    print(f"[D2 complementar -- EXPLORATORIO] {n} modelos da grade ({os.path.basename(args.grid_json)})")

    print("\n(1) Referencias para (A): tau de Kendall com o gap (erro teste CIFAR-10 - erro treino)")
    ref = {}
    for name, x in [("erro de validacao", ev), ("CVaR de validacao", v), ("-log lr (so' o lr)", -lr),
                    ("-dropout (so' o dropout)", -do), ("profundidade", dep),
                    ("log Tr(H2)", lc), ("log sharp 0.1", s1), ("log sharp 0.5", s5)]:
        t = kendall_tau(x, gap); ref[name] = t
        print(f"  {name:28s} tau = {t:+.3f}")
    out["reference_tau"] = ref
    print(f"  rho(log Tr(H2), x): prof {spearmanr(lc, dep)[0]:+.2f}  larg {spearmanr(lc, wid)[0]:+.2f}  "
          f"log lr {spearmanr(lc, lr)[0]:+.2f}  wd {spearmanr(lc, wd)[0]:+.2f}  dropout {spearmanr(lc, do)[0]:+.2f}")
    print(f"  rho(CVaR val, CVaR 10.1) = {spearmanr(v, y101)[0]:.3f}   rho(erro val, erro 10.1) = {spearmanr(ev, e101)[0]:.3f}")

    print("\n(2) Folga de (B): confiabilidade do CVaR 'alem da validacao' (metades independentes)")
    rng = np.random.default_rng(0); fol = {}
    for key, TT, lab in [("cvar_c101", T101, "CVaR CIFAR-10.1 (alvo de B)"), ("cvar_c10", T10, "CVaR CIFAR-10 (i.i.d., referencia)")]:
        for arch in (True, False):
            rs = []
            for _ in range(args.splits):
                pv = rng.permutation(V.shape[1]); hv = V.shape[1] // 2
                pt = rng.permutation(TT.shape[1])[:2000]; ht = 1000          # mesmo tamanho nos dois alvos
                v1, v2 = cvar_rows(V[:, pv[:hv]]), cvar_rows(V[:, pv[hv:]])
                t1, t2 = cvar_rows(TT[:, pt[:ht]]), cvar_rows(TT[:, pt[ht:]])
                c1 = [v1, dep, wid] if arch else [v1]; c2 = [v2, dep, wid] if arch else [v2]
                rs.append(np.corrcoef(resid(t1, c1), resid(t2, c2))[0, 1])
            r = float(np.mean(rs)); full = 2 * r / (1 + r)
            k = f"{key}|{'val+arq' if arch else 'val'}"
            fol[k] = {"r_halves": r, "spread95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
                      "reliability_full": full}
            print(f"  {lab:38s} controle {'val + arq' if arch else 'so val  '}: r(metades) {r:+.3f} "
                  f"[{fol[k]['spread95'][0]:+.2f}, {fol[k]['spread95'][1]:+.2f}] -> confiabilidade ~ {full:+.2f}")
    out["headroom"] = fol

    print("\n(3) O que preve o CIFAR-10.1 alem da validacao")
    beyond = {}
    for tgt, ctrl_v, lab in [(y101, v, "CVaR 10.1 | CVaR val"), (e101, ev, "erro 10.1 | erro val")]:
        print(f"  [{lab}]")
        for name, x, extra in [("log lr", lr, [dep, wid]), ("weight decay", wd, [dep, wid]), ("dropout", do, [dep, wid]),
                               ("profundidade", dep, [wid]), ("largura", wid, [dep]),
                               ("log Tr(H2)", lc, [dep, wid]), ("log sharp 0.1", s1, [dep, wid]), ("log sharp 0.5", s5, [dep, wid])]:
            C = [ctrl_v] + extra
            p = partial_spearman(tgt, x, C); ci = boot(lambda i: partial_spearman(tgt[i], x[i], [c[i] for c in C]), n)
            beyond[f"{lab} :: {name}"] = {"partial": p, "ci95": ci}
            print(f"    {name:16s} (+ arq) {p:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    out["beyond_validation"] = beyond

    print("\n(4) A curvatura e' indicador de outro fator?")
    proxy = {}
    for lab, tgt, C in [("CVaR 10.1 ~ Tr(H2) | val", y101, [v]), ("CVaR 10.1 ~ Tr(H2) | val, prof", y101, [v, dep]),
                        ("erro 10.1 ~ Tr(H2) | erro val, arq", e101, [ev, dep, wid]),
                        ("erro 10.1 ~ Tr(H2) | erro val, arq, log lr", e101, [ev, dep, wid, lr]),
                        ("erro 10.1 ~ Tr(H2) | erro val, 5 hiperparametros", e101, [ev, dep, wid, lr, wd, do]),
                        ("erro c10 (i.i.d.) ~ Tr(H2) | erro val, arq", e10, [ev, dep, wid])]:
        p = partial_spearman(tgt, lc, C); ci = boot(lambda i: partial_spearman(tgt[i], lc[i], [c[i] for c in C]), n)
        proxy[lab] = {"partial": p, "ci95": ci}
        print(f"  {lab:48s} {p:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    out["proxy"] = proxy
    save_result(out, experiment="d2_followup", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
