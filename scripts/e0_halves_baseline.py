#!/usr/bin/env python
# scripts/e0_halves_baseline.py
"""
Linha de base das "metades i.i.d." como MEDIA SOBRE DIVISOES (regra 4 da F3, pre-registrada em 11/10/2026).

Por que. O E0 imprime parcial(CVaR_H2, log Tr(H^2) | CVaR_H1) para UMA divisao aleatoria do alvo em duas
metades, com IC so' sobre configuracoes. Em 09-10/10/2026 (Bloco 2 da F2) ficou medido que (i) esse valor
oscila entre divisoes com desvio 0,07-0,35, logo uma divisao nao serve de checagem; e (ii) o valor esperado
NAO e' zero: e' o vies de erro de medida do controle ruidoso, com o sinal da correlacao bruta entre a
medida e o alvo. A leitura correta compara a parcial | val do E0 com esta linha de base, nao com zero.

Nao treina nada: le o JSON do E0 e as perdas por amostra em results/raw/<stem>_<dataset>.npz
(--save-losses). Para cada alvo (teste inteiro, 'novel', 'known'): R divisoes aleatorias em metades de
mesmo tamanho; parcial(H2, curv | H1) em cada uma; media, desvio e quantis 2,5/97,5 %; ao lado, o valor
da divisao unica do E0 e a parcial | val, para leitura.

Uso:
    python scripts/e0_halves_baseline.py "results/e0_generalization/*krono-s0.json" --R 300
    python scripts/e0_halves_baseline.py "results/e0_generalization/*drebin-cluster-s1-f*.json"
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np

from tune3.experiments.stats import partial_spearman


def cvar_rows(L, gamma=0.95):
    """CVaR empirico linha a linha, IDENTICO a tune3.core.objectives.cvar (quantil com interpolacao
    linear + media da cauda >= quantil), mas com np.partition (O(n) por linha) em vez de ordenar:
    R divisoes x alvos x metades sobre 300 modelos ficam em segundos."""
    n = L.shape[1]; pos = gamma * (n - 1); lo = int(np.floor(pos)); hi = min(lo + 1, n - 1); w = pos - lo
    P = np.partition(L, [lo, hi], axis=1)
    q = P[:, lo] * (1.0 - w) + P[:, hi] * w
    m = L >= q[:, None]
    return (L * m).sum(1) / m.sum(1)


def valid(r):
    ok = np.isfinite(r["curvature"]) and r["curvature"] > 0 and r["curvature"] != 1e3 \
        and np.isfinite(r["cvar_test"]) and r["cvar_test"] != 1e3
    return ok and (r["reached"] if "reached" in r else r.get("best_epoch", 2) >= 2)


def main():
    ap = argparse.ArgumentParser(description="Linha de base das metades i.i.d. como media sobre R divisoes.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = sorted({f for pat in args.files for f in glob.glob(pat)})
    rng = np.random.default_rng(args.seed)
    print(f"[metades i.i.d.: media sobre {args.R} divisoes]  (ler a parcial|val contra a MEDIA, nao contra zero)\n")
    print(f"  {'rodada':30s} {'alvo':16s} {'n':>4} {'rho_bruta':>9} {'parcial|val':>11} {'E0 (1 div.)':>11} {'media':>7} {'desvio':>7} {'q2.5':>6} {'q97.5':>6}")
    for f in files:
        J = json.load(open(f)); stem = os.path.splitext(os.path.basename(f))[0]; a_ = J["meta"].get("args", {})
        for d in J["payload"]["datasets"]:
            npz = f"results/raw/{stem}_{d['dataset']}.npz"
            if not os.path.exists(npz):
                print(f"  [aviso] sem perdas gravadas para {stem} / {d['dataset']} ({npz}); pulando"); continue
            Z = np.load(npz); ok = [r for r in d["rows"] if valid(r)]
            idx = np.array([r["cfg"] for r in ok]); T = Z["test"][idx].astype(np.float64); V = Z["val"][idx].astype(np.float64)
            lc = np.log10([r["curvature"] for r in ok]); v = cvar_rows(V); nov = Z["test_novel_mask"]
            name = (f"{d['dataset']} fold {a_.get('cluster_fold')}" if d["dataset"] == "drebin_cluster"
                    else f"{d['dataset']} semente {a_.get('seed')}")
            masks = {"cvar_test": np.ones(T.shape[1], bool)}
            if nov.any() and not nov.all():
                masks["cvar_test_novel"] = nov; masks["cvar_test_known"] = ~nov
            elif nov.any():
                masks["cvar_test_novel"] = nov
            for key, m in masks.items():
                TT = T[:, m]; y = cvar_rows(TT); nt = TT.shape[1]; h = nt // 2; vals = []
                for _ in range(args.R):
                    perm = rng.permutation(nt)
                    vals.append(partial_spearman(cvar_rows(TT[:, perm[h:2 * h]]), lc, [cvar_rows(TT[:, perm[:h]])]))
                vals = np.array(vals); vals = vals[np.isfinite(vals)]
                raw = partial_spearman(y, lc, [])
                pv = d["results"].get(key, {}).get("partial_given_val", float("nan"))
                one = d.get("artifact_checks", {}).get(key, {}).get("partial_iid_halves", float("nan"))
                q = np.percentile(vals, [2.5, 97.5])
                print(f"  {name:30s} {key:16s} {len(ok):>4} {raw:>+9.3f} {pv:>+11.3f} {one:>+11.3f} {vals.mean():>+7.3f} {vals.std():>7.3f} {q[0]:>+6.2f} {q[1]:>+6.2f}")
    print("\n  leitura: 'E0 (1 div.)' e' o valor de uma divisao so' (nao e' checagem); a media e' o vies esperado do controle"
          " ruidoso; a parcial|val do E0 so' indica algo alem da validacao se ficar claramente fora do intervalo [q2.5, q97.5].")


if __name__ == "__main__":
    main()
