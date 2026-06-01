#!/usr/bin/env python
# scripts/run_tune3_drebin.py
"""
Ponto de entrada: roda o Tune3 completo no DREBIN-215.

Uso (do diretorio raiz do projeto):
    python scripts/run_tune3_drebin.py --csv data/drebin215.csv --epochs 50 \
        --n-init 8 --n-iter 20 --device cpu --seed 0

Para o PILOTO rapido (validar o pipeline):
    python scripts/run_tune3_drebin.py --csv data/drebin215.csv --epochs 20 \
        --n-init 5 --n-iter 8 --seed 0

Use --device cuda na maquina com GPU.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tune3.data.drebin import DrebinLoader, DrebinConfig
from tune3.integration.runner import run_tune3, Tune3RunConfig
from tune3.integration.trial import TrialConfig
from tune3.macro.bo_loop import MacroConfig


def main():
    ap = argparse.ArgumentParser(description="Roda o Tune3 no DREBIN-215.")
    ap.add_argument("--csv", default="data/drebin215.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--wandb", action="store_true", help="logar no Weights & Biases")
    ap.add_argument("--out", default="tune3_resultado.json")
    args = ap.parse_args()

    # 1) dados
    loader = DrebinLoader(DrebinConfig(csv_path=args.csv))
    X_tr, X_val, X_te, y_tr, y_val, y_te = loader.load_splits()
    print(f"[dados] treino={X_tr.shape} val={X_val.shape} teste={X_te.shape} "
          f"frac_malware={y_tr.mean():.3f}")

    # 2) (opcional) W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="tune3", config=vars(args),
                               name=f"tune3-drebin-seed{args.seed}")

    # 3) configura e roda o Tune3
    run_cfg = Tune3RunConfig(
        macro=MacroConfig(n_init=args.n_init, n_iter=args.n_iter,
                          device="cpu", seed=args.seed),
        trial=TrialConfig(max_epochs=args.epochs, batch_size=args.batch_size,
                          optimizer=args.optimizer, device=args.device,
                          gamma=args.gamma, seed=args.seed),
    )
    out = run_tune3((X_tr, y_tr, X_val, y_val), run_cfg, wandb_run=wandb_run)

    # 4) resultados
    print("\n===== RESULTADO TUNE3 =====")
    print(f"avaliacoes: {out['pareto']['n_evaluations']}")
    print(f"pontos na frente de Pareto: {len(out['pareto']['X'])}")
    print("hiperparametros recomendados (knee point):")
    print(json.dumps(out["knee_hparams"], indent=2))

    with open(args.out, "w") as f:
        json.dump({"pareto": out["pareto"], "knee_hparams": out["knee_hparams"]}, f, indent=2)
    print(f"\n[salvo] {args.out}")

    if wandb_run is not None:
        wandb_run.summary["pareto_size"] = len(out["pareto"]["X"])
        wandb_run.summary["knee_hparams"] = out["knee_hparams"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
