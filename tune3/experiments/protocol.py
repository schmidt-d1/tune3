# tune3/experiments/protocol.py  (ATUALIZADO Fase 8)
"""Orquestracao experimental do Tune3. Agora avalia DUAS selecoes do Tune3:
- tune3          : knee point (equilibrio CVaR/curvatura)
- tune3_bestcvar : ponto de Pareto de melhor CVaR (alinhado a metrica de deploy)
Isso isola se o resultado depende da SELECAO ou da BUSCA."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import structlog

from tune3.integration.runner import run_tune3, Tune3RunConfig
from tune3.integration.trial import TrialConfig
from tune3.macro.bo_loop import MacroConfig
from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
from tune3.baselines.hpo import RandomSearchHPO, ASHA

logger = structlog.get_logger()


@dataclass
class ProtocolConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    gamma: float = 0.95
    device: str = "cpu"
    epochs: int = 100
    patience: int = 10
    n_init: int = 8
    n_iter: int = 20
    asha_configs: int = 32
    search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "log_lr": (-5.0, -2.0), "log_wd": (-6.0, -2.0),
        "dropout": (0.0, 0.5), "hidden_dim": (32.0, 256.0), "n_layers": (1.0, 4.0)})


def _eval_on_test(hparams, data_full, seed, cfg, optimizer="sgd"):
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    tc = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer=optimizer,
                          device=cfg.device, gamma=cfg.gamma, seed=seed)
    r = plain_trial(hparams, (X_tr, y_tr, X_te, y_te), tc)
    return {"cvar_test": r["cvar"], "curvature_test": r["curvature"], "test_loss": r["final_val_loss"]}


def run_seed(seed, loader_fn, cfg):
    data_full = loader_fn(seed)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    train_val = (X_tr, y_tr, X_val, y_val)
    out = {}

    run_cfg = Tune3RunConfig(
        macro=MacroConfig(n_init=cfg.n_init, n_iter=cfg.n_iter, device=cfg.device, seed=seed),
        trial=TrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sgd",
                          device=cfg.device, gamma=cfg.gamma, seed=seed),
        search_space=dict(cfg.search_space))
    t3 = run_tune3(train_val, run_cfg)
    out["tune3"] = _eval_on_test(t3["knee_hparams"], data_full, seed, cfg, "sgd")
    out["tune3_bestcvar"] = _eval_on_test(t3["best_cvar_hparams"], data_full, seed, cfg, "sgd")

    tc = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, device=cfg.device,
                          gamma=cfg.gamma, seed=seed)
    rs = RandomSearchHPO(cfg.search_space, train_val, n_trials=cfg.n_init + cfg.n_iter,
                         trial_cfg=tc, seed=seed)
    out["random_search"] = _eval_on_test(rs.run()["best"]["hparams"], data_full, seed, cfg, "sgd")

    asha = ASHA(cfg.search_space, train_val, n_configs=cfg.asha_configs, r0=4, eta=2,
                trial_cfg=tc, seed=seed)
    out["asha"] = _eval_on_test(asha.run()["best"]["hparams"], data_full, seed, cfg, "sgd")

    tc_sam = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sam",
                              device=cfg.device, gamma=cfg.gamma, seed=seed)
    rs_sam = RandomSearchHPO(cfg.search_space, train_val, n_trials=cfg.n_init + cfg.n_iter,
                             trial_cfg=tc_sam, seed=seed)
    out["sam"] = _eval_on_test(rs_sam.run()["best"]["hparams"], data_full, seed, cfg, "sam")

    logger.info("seed concluida", seed=seed,
                cvar={k: round(v["cvar_test"], 4) for k, v in out.items()})
    return out


def run_protocol(loader_fn, cfg):
    methods = ["tune3", "tune3_bestcvar", "random_search", "asha", "sam"]
    scores = {m: [] for m in methods}; per_seed = []
    for seed in cfg.seeds:
        res = run_seed(seed, loader_fn, cfg); per_seed.append({"seed": seed, **res})
        for m in methods: scores[m].append(res[m]["cvar_test"])
    return {"cvar_test_by_method": scores, "per_seed": per_seed,
            "seeds": list(cfg.seeds), "methods": methods}
