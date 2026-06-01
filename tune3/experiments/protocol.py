# tune3/experiments/protocol.py
"""
Orquestracao experimental do Tune3 (Fase 6).

Protocolo cientificamente correto:
  1. HPO (Tune3 ou baseline) seleciona hiperparametros usando TREINO/VALIDACAO;
  2. os hiperparametros escolhidos sao re-treinados e avaliados no TESTE (holdout);
  3. a metrica de TESTE (CVaR) e' comparada entre metodos, PAREADA por seed.

Isso evita o vies de selecionar e avaliar no mesmo conjunto. Cada seed re-faz o
split (estratificado) e re-roda todos os metodos, garantindo pareamento.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import structlog

from tune3.integration.runner import run_tune3, Tune3RunConfig
from tune3.integration.trial import TrialConfig
from tune3.macro.bo_loop import MacroConfig
from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
from tune3.baselines.hpo import RandomSearchHPO, ASHA, _decode
from tune3.core.objectives import cvar

logger = structlog.get_logger()


@dataclass
class ProtocolConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    gamma: float = 0.95
    device: str = "cpu"
    epochs: int = 20
    # orcamentos (iguais entre metodos p/ justica)
    n_init: int = 5
    n_iter: int = 8         # Tune3 e Random gastam n_init+n_iter trials
    asha_configs: int = 16
    search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "log_lr": (-5.0, -2.0), "log_wd": (-6.0, -2.0),
        "dropout": (0.0, 0.5), "hidden_dim": (32.0, 256.0), "n_layers": (1.0, 4.0),
    })


def _eval_on_test(hparams: Dict[str, float], data_full, seed: int,
                  cfg: ProtocolConfig, optimizer: str = "sgd") -> Dict:
    """Re-treina com hparams (train) e avalia CVaR/curvatura no TESTE."""
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    # treina em train, avalia em teste (passando teste como 'val' do plain_trial)
    tc = PlainTrialConfig(max_epochs=cfg.epochs, optimizer=optimizer,
                          device=cfg.device, gamma=cfg.gamma, seed=seed)
    r = plain_trial(hparams, (X_tr, y_tr, X_te, y_te), tc)
    return {"cvar_test": r["cvar"], "curvature_test": r["curvature"],
            "test_loss": r["final_val_loss"]}


def run_seed(seed: int, loader_fn: Callable[[int], Tuple], cfg: ProtocolConfig) -> Dict:
    """Roda todos os metodos numa seed e devolve as metricas de TESTE."""
    data_full = loader_fn(seed)               # (Xtr,ytr,Xval,yval,Xte,yte)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    train_val = (X_tr, y_tr, X_val, y_val)
    out = {}

    # --- Tune3 ---
    run_cfg = Tune3RunConfig(
        macro=MacroConfig(n_init=cfg.n_init, n_iter=cfg.n_iter, device="cpu", seed=seed),
        trial=TrialConfig(max_epochs=cfg.epochs, optimizer="sgd",
                          device=cfg.device, gamma=cfg.gamma, seed=seed),
        search_space=dict(cfg.search_space),
    )
    t3 = run_tune3(train_val, run_cfg)
    out["tune3"] = _eval_on_test(t3["knee_hparams"], data_full, seed, cfg, "sgd")

    # --- B1 Random Search ---
    rs = RandomSearchHPO(cfg.search_space, train_val,
                         n_trials=cfg.n_init + cfg.n_iter,
                         trial_cfg=PlainTrialConfig(max_epochs=cfg.epochs, optimizer="sgd",
                                                    device=cfg.device, gamma=cfg.gamma, seed=seed),
                         seed=seed)
    out["random_search"] = _eval_on_test(rs.run()["best"]["hparams"], data_full, seed, cfg, "sgd")

    # --- B2 ASHA ---
    asha = ASHA(cfg.search_space, train_val, n_configs=cfg.asha_configs, r0=4, eta=2,
                trial_cfg=PlainTrialConfig(max_epochs=cfg.epochs, optimizer="sgd",
                                           device=cfg.device, gamma=cfg.gamma, seed=seed),
                seed=seed)
    out["asha"] = _eval_on_test(asha.run()["best"]["hparams"], data_full, seed, cfg, "sgd")

    # --- B3 SAM (random search HPO + otimizador SAM) ---
    rs_sam = RandomSearchHPO(cfg.search_space, train_val,
                             n_trials=cfg.n_init + cfg.n_iter,
                             trial_cfg=PlainTrialConfig(max_epochs=cfg.epochs, optimizer="sam",
                                                        device=cfg.device, gamma=cfg.gamma, seed=seed),
                             seed=seed)
    out["sam"] = _eval_on_test(rs_sam.run()["best"]["hparams"], data_full, seed, cfg, "sam")

    logger.info("seed concluida", seed=seed,
                cvar={k: round(v["cvar_test"], 4) for k, v in out.items()})
    return out


def run_protocol(loader_fn: Callable[[int], Tuple], cfg: ProtocolConfig) -> Dict:
    """
    Roda o protocolo completo sobre cfg.seeds e devolve as metricas de teste
    por metodo/seed, prontas para a analise estatistica (compare_paired).
    loader_fn(seed) -> (Xtr,ytr,Xval,yval,Xte,yte) com split dependente da seed.
    """
    methods = ["tune3", "random_search", "asha", "sam"]
    scores = {m: [] for m in methods}
    per_seed = []
    for seed in cfg.seeds:
        res = run_seed(seed, loader_fn, cfg)
        per_seed.append({"seed": seed, **res})
        for m in methods:
            scores[m].append(res[m]["cvar_test"])
    return {"cvar_test_by_method": scores, "per_seed": per_seed,
            "seeds": list(cfg.seeds), "methods": methods}
