#!/usr/bin/env python
# scripts/diag_ddkf.py
"""
DIAGNOSTICO: sob que configuracao o DDKF (nivel MICRO) chega a ATUAR?

Motivacao (21/09/2026). Nos pilotos H1 com 20 epocas, n_ddkf_active_epochs = 0 em
TODOS os 39 trials x 3 seeds x 2 cadencias. Isso NAO e' evidencia de que o
controlador nao ajuda: e' consequencia aritmetica do desenho. O DDKF so' produz
correcao depois de acumular `DDKFConfig.min_samples` observacoes (padrao 30, uma
por epoca) e, mesmo depois disso, a correcao e' zerada pelo RegimeGate enquanto
R^2 < r2_low (padrao 0.10). Com trials de 11 a 20 epocas, a primeira condicao
sozinha ja' garante zero.

Este script separa as DUAS causas possiveis:
  (C1) buffer curto demais  -> n_ddkf_observations < min_samples
  (C2) gate fechado         -> buffer suficiente, mas R^2 nunca passa de r2_low

E compara os DOIS VETORES DE OBSERVACAO (--obs-vector, braco D1):
  "loss"      z = (train_loss, val_loss, log GSNR)        -- o que o codigo sempre fez
  "curvature" z = (train_loss, log(1+Tr(H^2)), log GSNR)  -- o que o manuscrito descrevia
A pergunta e' se a curvatura torna o sistema identificavel onde a perda nao torna.
Com "curvature" o script forca curvature_every=1: com cadencia maior a observacao
repete entre reestimativas e a covariancia zera por construcao.

Sem essa separacao, congelar o protocolo com E2 ("o DDKF engaja?") seria
pre-registrar uma tautologia: a resposta ja' esta determinada pela configuracao.

Uso:
    python scripts/diag_ddkf.py --csv data/drebin215.csv --device cuda --tag dionatan
    python scripts/diag_ddkf.py                      # dados sinteticos, CPU

Saida: uma linha por celula da grade (min_samples x epocas x excitacao), com
n_ddkf_observations, R^2 maximo/medio, fracoes de regime, epocas ativas e se o lr
de fato mudou. Grava results/diag_ddkf/<ts>_<commit>_<tag>.json.
"""
from __future__ import annotations

import argparse, time, itertools
import numpy as np

from tune3.integration.trial import run_trial, TrialConfig
from tune3.micro import DDKFConfig
from tune3.experiments.results_io import save_result


def _synthetic(n, seed=0, d=215):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, d)) < 0.3).astype(np.float32)
    w = rng.normal(size=d)
    y = (X @ w + rng.normal(scale=0.5, size=n) > np.median(X @ w)).astype(np.int64)
    return X, y


def load(args):
    if args.dataset == "nslkdd":
        from tune3.data.registry import load_dataset
        Xtr, Xv, Xte, ytr, yv, yte = load_dataset("nslkdd", args.data_path, 0,
                                                  max_train=args.nsl_max_train)
        return Xtr, ytr, Xv, yv
    if args.csv:
        from tune3.data.drebin import DrebinLoader, DrebinConfig
        Xtr, Xv, Xte, ytr, yv, yte = DrebinLoader(
            DrebinConfig(csv_path=args.csv, random_state=0)).load_splits()
        return Xtr, ytr, Xv, yv
    Xtr, ytr = _synthetic(6000, 0)
    Xv, yv = _synthetic(2000, 1)
    return Xtr, ytr, Xv, yv


def main():
    ap = argparse.ArgumentParser(description="Diagnostico de engajamento do DDKF.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dataset", default=None, choices=["drebin", "nslkdd"])
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--nsl-max-train", type=int, default=25000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--min-samples", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--epochs", type=int, nargs="+", default=[30, 60])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exploration-std", type=float, nargs="+", default=[0.03, 0.30],
                    help="amplitude da excitacao do log-lr na identificacao (padrao do projeto: "
                         "0.03, ~3%% em log-lr). Excitacao maior torna Cov(x,z) mensuravel, ao "
                         "preco de perturbar o proprio treino -- e' esse o trade-off a medir.")
    ap.add_argument("--obs-vector", nargs="+", default=["loss", "curvature"],
                    choices=["loss", "curvature"],
                    help="vetores de observacao a comparar (padrao: os dois)")
    args = ap.parse_args(); t0 = time.time()

    data = load(args)
    # hiperparametros fixos e plausiveis: o alvo aqui e' o CONTROLADOR, nao a busca
    hp = {"learning_rate": 1e-3, "weight_decay": 1e-4,
          "dropout": 0.1, "hidden_dim": 128, "n_layers": 2}

    rows = []
    print(f"{'z':>10} {'min_s':>6} {'epocas':>7} {'std':>5} {'decai':>6} | {'obs':>4} {'R2max':>7} {'R2med':>7} "
          f"{'A':>5} {'B':>5} {'C':>5} | {'ativas':>6} {'lr mudou':>9} {'epocas':>7}")
    print("-" * 107)
    for obs, min_s, ep, std, decay in itertools.product(args.obs_vector, args.min_samples,
                                                        args.epochs, args.exploration_std,
                                                        [0.0, 0.02]):
        # com observacao de curvatura, cadencia 1 e' obrigatoria (ver docstring)
        cev = 1 if obs == "curvature" else 5
        cfg = TrialConfig(
            max_epochs=ep, patience=ep,          # patience = ep: NAO deixa o early stopping
            device=args.device, seed=args.seed,  # encurtar o trial e mascarar a causa
            obs_vector=obs, curvature_every=cev,
            ddkf=DDKFConfig(rho=0.95, sigma_eta=0.05,
                            min_samples=min_s, exploration_std=std, exploration_decay=decay),
        )
        r = run_trial(hp, data, cfg)
        t = r["_telemetry"] if "_telemetry" in r else r
        rf = t["regime_fractions"]
        row = {"obs_vector": obs, "curvature_every": cev,
               "min_samples": min_s, "epochs": ep, "exploration_std": std,
               "exploration_decay": decay,
               "n_ddkf_observations": t["n_ddkf_observations"], "r2_max": t["r2_max"],
               "r2_mean": t["r2_mean"], "regime_fractions": rf,
               "n_ddkf_active_epochs": t["n_ddkf_active_epochs"],
               "lr_change_fraction": t["lr_change_fraction"],
               "n_epochs_run": t["n_epochs_run"], "cvar": r.get("cvar")}
        rows.append(row)
        print(f"{obs:>10} {min_s:>6} {ep:>7} {std:>5.2f} {decay:>6.2f} | {row['n_ddkf_observations']:>4} "
              f"{row['r2_max']:>7.3f} {row['r2_mean']:>7.3f} "
              f"{rf['A']:>5.2f} {rf['B']:>5.2f} {rf['C']:>5.2f} | "
              f"{row['n_ddkf_active_epochs']:>6} {row['lr_change_fraction']:>9.2f} "
              f"{row['n_epochs_run']:>7}")

    # ---- veredito automatico, na taxonomia do projeto, POR VETOR ----
    print()
    for obs in args.obs_vector:
        sub = [r for r in rows if r["obs_vector"] == obs]
        r2m = max(r["r2_max"] for r in sub); ativas = sum(r["n_ddkf_active_epochs"] for r in sub)
        estado = "IDENTIFICAVEL" if r2m >= 0.10 else "NAO identificavel"
        melhor_std = max(sub, key=lambda r: r["r2_max"])["exploration_std"]
        print(f"  z='{obs}': R2max={r2m:.4f} (limiar 0.10, melhor com exploration_std="
              f"{melhor_std:.2f}) -> {estado}; epocas com DDKF ativo (soma) = {ativas}")

    ativou = [r for r in rows if r["n_ddkf_active_epochs"] > 0]
    com_buffer = [r for r in rows if r["n_ddkf_observations"] >= r["min_samples"]]
    print()
    if not com_buffer:
        v = ("EM ABERTO: nenhuma celula acumulou min_samples observacoes -- "
             "aumente --epochs para separar as causas")
    elif ativou:
        melhor = max(ativou, key=lambda r: r["n_ddkf_active_epochs"])
        v = (f"O DDKF ATUA com z='{melhor['obs_vector']}', min_samples={melhor['min_samples']} e "
             f"{melhor['epochs']} epocas: {melhor['n_ddkf_active_epochs']} epocas ativas, "
             f"R2max={melhor['r2_max']:.3f}. Este e' o braco a pre-registrar no E2.")
    else:
        r2m = max(r["r2_max"] for r in com_buffer)
        v = (f"CAUSA (C2) gate fechado em TODOS os vetores testados: buffer suficiente, mas R2 "
             f"nunca passou de {r2m:.3f} (limiar r2_low=0.10). O DDKF e' inerte por falta de "
             f"informacao, nao por falta de amostras -- nem com a curvatura como observacao.")
    print("VEREDITO:", v)
    save_result({"rows": rows, "verdict": v, "hparams": hp},
                experiment="diag_ddkf", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
