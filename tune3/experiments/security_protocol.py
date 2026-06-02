# tune3/experiments/security_protocol.py
"""
Protocolo de seguranca (Fases S1/S2): roda Tune3 + baselines e avalia no teste
com CVaR + curvatura + metricas operacionais (FPR@TPR, AUC-ROC, AUC-PR, F1, MCC).

Generaliza o protocol.py para um loader_fn(seed) arbitrario, permitindo plugar
o loader desbalanceado (S1) ou de deslocamento por cluster (S2).
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
from tune3.baselines.hpo import RandomSearchHPO, ASHA
from tune3.experiments.metrics import all_metrics

logger = structlog.get_logger()


@dataclass
class SecurityProtocolConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    gamma: float = 0.95
    device: str = "cpu"
    epochs: int = 100
    patience: int = 10
    n_init: int = 8
    n_iter: int = 20
    asha_configs: int = 32
    macro_device: str = "cpu"
    search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "log_lr": (-5.0, -2.0), "log_wd": (-6.0, -2.0),
        "dropout": (0.0, 0.5), "hidden_dim": (32.0, 256.0), "n_layers": (1.0, 4.0)})


def _eval_full(hparams, X_tr, y_tr, X_te, y_te, cfg, optimizer="sgd"):
    """Treina com hparams e avalia CVaR + curvatura + metricas de seguranca no teste."""
    tc = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer=optimizer,
                          device=cfg.device, gamma=cfg.gamma, seed=cfg.seeds[0] if cfg.seeds else 0)
    r = plain_trial(hparams, (X_tr, y_tr, X_te, y_te), tc, return_scores=True)
    out = {"cvar_test": r["cvar"], "curvature_test": r["curvature"]}
    if r.get("scores") is not None:
        m = all_metrics(y_te, r["scores"], tpr_target=0.95, fpr_target=0.01)
        out.update({f"test_{k}": v for k, v in m.items()})
    return out


def run_security_seed(seed, loader_fn, cfg):
    data_full = loader_fn(seed)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    train_val = (X_tr, y_tr, X_val, y_val)
    out = {}
    # seed do trial de avaliacao
    cfg_seed = SecurityProtocolConfig(**{**cfg.__dict__, "seeds": [seed]})

    run_cfg = Tune3RunConfig(
        macro=MacroConfig(n_init=cfg.n_init, n_iter=cfg.n_iter, device=cfg.macro_device, seed=seed),
        trial=TrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sgd",
                          device=cfg.device, gamma=cfg.gamma, seed=seed),
        search_space=dict(cfg.search_space))
    t3 = run_tune3(train_val, run_cfg)
    out["tune3"] = _eval_full(t3["knee_hparams"], X_tr, y_tr, X_te, y_te, cfg_seed, "sgd")
    out["tune3_bestcvar"] = _eval_full(t3["best_cvar_hparams"], X_tr, y_tr, X_te, y_te, cfg_seed, "sgd")

    tc = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, device=cfg.device,
                          gamma=cfg.gamma, seed=seed)
    rs = RandomSearchHPO(cfg.search_space, train_val, n_trials=cfg.n_init + cfg.n_iter, trial_cfg=tc, seed=seed)
    out["random_search"] = _eval_full(rs.run()["best"]["hparams"], X_tr, y_tr, X_te, y_te, cfg_seed, "sgd")

    asha = ASHA(cfg.search_space, train_val, n_configs=cfg.asha_configs, r0=4, eta=2, trial_cfg=tc, seed=seed)
    out["asha"] = _eval_full(asha.run()["best"]["hparams"], X_tr, y_tr, X_te, y_te, cfg_seed, "sgd")

    tc_sam = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sam",
                              device=cfg.device, gamma=cfg.gamma, seed=seed)
    rs_sam = RandomSearchHPO(cfg.search_space, train_val, n_trials=cfg.n_init + cfg.n_iter, trial_cfg=tc_sam, seed=seed)
    out["sam"] = _eval_full(rs_sam.run()["best"]["hparams"], X_tr, y_tr, X_te, y_te, cfg_seed, "sam")

    logger.info("security seed concluida", seed=seed,
                cvar={k: round(v["cvar_test"], 4) for k, v in out.items()})
    return out


def run_security_protocol(loader_fn, cfg):
    methods = ["tune3", "tune3_bestcvar", "random_search", "asha", "sam"]
    per_seed = []
    for seed in cfg.seeds:
        res = run_security_seed(seed, loader_fn, cfg)
        per_seed.append({"seed": seed, **res})
    # organizar por metrica
    metric_keys = [k for k in per_seed[0]["tune3"].keys()]
    by_metric = {m: {mk: [] for mk in metric_keys} for m in methods}
    for entry in per_seed:
        for m in methods:
            for mk in metric_keys:
                by_metric[m][mk].append(entry[m][mk])
    return {"by_metric": by_metric, "per_seed": per_seed,
            "methods": methods, "seeds": list(cfg.seeds), "metric_keys": metric_keys}
