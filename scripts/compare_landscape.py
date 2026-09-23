#!/usr/bin/env python
# scripts/compare_landscape.py
"""
A PREMISSA DA TRILHA NSL-KDD, TESTADA: a curvatura varia mais no NSL-KDD que no DREBIN?

Motivacao (23/09/2026). A trilha NSL-KDD foi proposta porque o DREBIN-215 tem 215 atributos
binarios -- superficie de perda bem condicionada -- e a tese do Tune3 e' sobre curvatura.
Se no DREBIN a curvatura quase nao varia entre configuracoes, o 2o objetivo do MACRO
(Tr(H^2)) nao tem o que discriminar, e isso explicaria um E1 sem efeito. Antes de gastar
GPU numa campanha inteira, medimos se a premissa e' verdadeira.

O que mede, para as MESMAS K configuracoes de hiperparametros (sorteadas do espaco de busca
do protocolo) em cada dataset:
  - Tr(H^2) ao fim de um treino curto (Hutchinson, M=20, em model.eval());
  - dispersao de log10 Tr(H^2) entre configuracoes (desvio e amplitude);
  - correlacao de Spearman entre Tr(H^2) e CVaR de validacao (o 2o objetivo carrega
    informacao sobre o 1o?);
  - trials abortados pelo Cantelli (treinabilidade).

Uso:
    python scripts/compare_landscape.py --device cuda --tag dionatan
    python scripts/compare_landscape.py --k 12 --epochs 15 --nsl-scaling standard log_standard
"""
from __future__ import annotations

import argparse, time
import numpy as np
from scipy.stats import spearmanr

from tune3.data.registry import load_dataset
from tune3.integration.trial import run_trial, TrialConfig
from tune3.experiments.results_io import save_result

SPACE = {"log_lr": (-5.0, -2.0), "log_wd": (-6.0, -2.0), "dropout": (0.0, 0.5),
         "hidden_dim": (32.0, 256.0), "n_layers": (1.0, 4.0)}


def sample_configs(k, seed):
    rng = np.random.default_rng(seed); out = []
    for _ in range(k):
        r = {n: rng.uniform(lo, hi) for n, (lo, hi) in SPACE.items()}
        out.append({"learning_rate": 10 ** r["log_lr"], "weight_decay": 10 ** r["log_wd"],
                    "dropout": r["dropout"], "hidden_dim": int(round(r["hidden_dim"])),
                    "n_layers": int(round(r["n_layers"]))})
    return out


def run_arm(name, data, configs, args):
    rows = []
    for i, hp in enumerate(configs):
        cfg = TrialConfig(max_epochs=args.epochs, patience=args.epochs, device=args.device,
                          seed=args.seed, curvature_every=args.epochs)   # curvatura so' no fim
        cfg.hutchinson.num_probes = 20
        r = run_trial(hp, data, cfg)
        rows.append({"cfg": i, "hparams": hp, "curvature": float(r["curvature"]),
                     "cvar": float(r["cvar"]), "aborted": bool(r["aborted"])})
    ok = [r for r in rows if not r["aborted"] and np.isfinite(r["curvature"]) and r["curvature"] > 0]
    lc = np.log10([r["curvature"] for r in ok]) if ok else np.array([])
    rho = (float(spearmanr([r["curvature"] for r in ok], [r["cvar"] for r in ok])[0])
           if len(ok) >= 4 else float("nan"))
    s = {"arm": name, "n_ok": len(ok), "n_aborted": len(rows) - len(ok),
         "log10_curv_mean": float(lc.mean()) if lc.size else float("nan"),
         "log10_curv_sd": float(lc.std(ddof=1)) if lc.size > 1 else float("nan"),
         "log10_curv_range": float(lc.max() - lc.min()) if lc.size else float("nan"),
         "spearman_curv_cvar": rho,
         "cvar_mean": float(np.mean([r["cvar"] for r in ok])) if ok else float("nan"),
         "rows": rows}
    print(f"  {name:24s} ok={s['n_ok']:>2}/{len(rows)}  log10 Tr(H^2): media={s['log10_curv_mean']:+.2f} "
          f"dp={s['log10_curv_sd']:.2f} amplitude={s['log10_curv_range']:.2f}  "
          f"Spearman(curv,CVaR)={rho:+.2f}  CVaR medio={s['cvar_mean']:.4f}")
    return s


def main():
    ap = argparse.ArgumentParser(description="Compara a paisagem de curvatura DREBIN x NSL-KDD.")
    ap.add_argument("--drebin", default="data/drebin215.csv")
    ap.add_argument("--nslkdd", default="data/nsl_kdd")
    ap.add_argument("--nsl-scaling", nargs="+", default=["standard", "log_standard"])
    ap.add_argument("--nsl-max-train", type=int, default=25000,
                    help="subamostra estratificada do treino NSL-KDD (tamanho do 20%% oficial)")
    ap.add_argument("--k", type=int, default=10, help="configuracoes sorteadas por dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time()

    configs = sample_configs(args.k, args.seed)          # MESMAS configuracoes nos dois
    arms = []
    print(f"[paisagem] {args.k} configuracoes, {args.epochs} epocas, device={args.device}\n")
    d = load_dataset("drebin", args.drebin, args.seed)
    arms.append(run_arm("drebin", (d[0], d[3], d[1], d[4]), configs, args))
    for sc in args.nsl_scaling:
        d = load_dataset("nslkdd", args.nslkdd, args.seed, scaling=sc, max_train=args.nsl_max_train)
        arms.append(run_arm(f"nslkdd/{sc}", (d[0], d[3], d[1], d[4]), configs, args))

    base = arms[0]
    print("\nVEREDITO (premissa: 'a curvatura varia mais no NSL-KDD'):")
    for a in arms[1:]:
        r = a["log10_curv_sd"] / base["log10_curv_sd"] if base["log10_curv_sd"] > 0 else float("nan")
        if min(a["n_ok"], base["n_ok"]) < 8:      # desvio-padrao de poucos pontos nao decide nada
            v = f"EM ABERTO (so' {min(a['n_ok'], base['n_ok'])} configuracoes validas; use --k >= 10)"
        else:
            v = ("CONFIRMADA" if r >= 1.5 else "NAO confirmada" if r <= 1.0 else "fraca")
        print(f"  {a['arm']:24s} dispersao de log10 Tr(H^2) = {r:.2f}x a do DREBIN -> {v}")
    save_result({"arms": arms, "configs": configs}, experiment="landscape",
                tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
