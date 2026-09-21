# tune3/experiments/security_protocol.py
"""
Protocolo de seguranca (Fases S1/S2): roda Tune3 + baselines e avalia no teste
com CVaR + curvatura + metricas operacionais (FPR@TPR, AUC-ROC, AUC-PR, F1, MCC).

Generaliza o protocol.py para um loader_fn(seed) arbitrario, permitindo plugar
o loader desbalanceado (S1) ou de deslocamento por cluster (S2).

CORRECAO (set/2026): a avaliacao final treina com (treino, VALIDACAO), escolhe a
epoca na validacao, e avalia o modelo escolhido no TESTE uma unica vez
(plain_trial(..., test_data=...)). Antes, o teste era usado como validacao
(early stopping e melhor epoca escolhidos no teste): vazamento.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import structlog

from tune3.integration.runner import run_tune3, Tune3RunConfig
from tune3.integration.trial import TrialConfig, run_trial
from tune3.macro.bo_loop import MacroConfig
from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
from tune3.baselines.hpo import RandomSearchHPO, ASHA
from tune3.baselines.bo_mono import MonoObjectiveBO
from tune3.experiments.metrics import all_metrics
from tune3.experiments.protocol import ALL_METHODS, CORE_METHODS

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
    methods: List[str] = field(default_factory=lambda: list(ALL_METHODS))
    search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "log_lr": (-5.0, -2.0), "log_wd": (-6.0, -2.0),
        "dropout": (0.0, 0.5), "hidden_dim": (32.0, 256.0), "n_layers": (1.0, 4.0)})

    def __post_init__(self):
        bad = [m for m in self.methods if m not in ALL_METHODS]
        if bad:
            raise ValueError(f"metodos desconhecidos: {bad}; validos: {ALL_METHODS}")


def _eval_full(hparams, data_full, seed, cfg, optimizer="sgd"):
    """Treina em (treino, val) com early stopping na VAL; avalia o modelo escolhido no TESTE."""
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    tc = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer=optimizer,
                          device=cfg.device, gamma=cfg.gamma, seed=seed)
    r = plain_trial(hparams, (X_tr, y_tr, X_val, y_val), tc,
                    return_scores=True, test_data=(X_te, y_te))
    out = {"cvar_test": r["cvar_test"], "cvar_val": r["cvar"], "curvature_test": r["curvature"],
           "test_loss": r["test_loss"], "best_epoch": r["best_epoch"], "hparams": hparams}
    if r.get("scores_test") is not None:
        m = all_metrics(y_te, r["scores_test"], tpr_target=0.95, fpr_target=0.01)
        out.update({f"test_{k}": v for k, v in m.items()})
    return out


def _tune3_variant(train_val, cfg, seed, objective2="curvature", ddkf_enabled=True):
    run_cfg = Tune3RunConfig(
        macro=MacroConfig(n_init=cfg.n_init, n_iter=cfg.n_iter, device=cfg.macro_device, seed=seed),
        trial=TrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sgd",
                          device=cfg.device, gamma=cfg.gamma, seed=seed, ddkf_enabled=ddkf_enabled),
        search_space=dict(cfg.search_space), objective2=objective2)
    return run_tune3(train_val, run_cfg)


def run_security_seed(seed, loader_fn, cfg):
    data_full = loader_fn(seed)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    train_val = (X_tr, y_tr, X_val, y_val)
    out = {}; telemetry = {}
    M = set(cfg.methods)

    if M & {"tune3", "tune3_bestcvar"}:
        t3 = _tune3_variant(train_val, cfg, seed)
        telemetry["tune3"] = t3["trials"]
        if "tune3" in M:
            out["tune3"] = _eval_full(t3["knee_hparams"], data_full, seed, cfg, "sgd")
        if "tune3_bestcvar" in M:
            out["tune3_bestcvar"] = _eval_full(t3["best_cvar_hparams"], data_full, seed, cfg, "sgd")

    if "tune3_noddkf_bestcvar" in M:
        t3n = _tune3_variant(train_val, cfg, seed, ddkf_enabled=False)
        telemetry["tune3_noddkf"] = t3n["trials"]
        out["tune3_noddkf_bestcvar"] = _eval_full(t3n["best_cvar_hparams"], data_full, seed, cfg, "sgd")

    if "tune3_placebo_bestcvar" in M:
        t3p = _tune3_variant(train_val, cfg, seed, objective2="random")
        telemetry["tune3_placebo"] = t3p["trials"]
        out["tune3_placebo_bestcvar"] = _eval_full(t3p["best_cvar_hparams"], data_full, seed, cfg, "sgd")

    if "bo_mono_cvar" in M:
        trial_cfg = TrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sgd",
                                device=cfg.device, gamma=cfg.gamma, seed=seed)
        b4 = MonoObjectiveBO(cfg.search_space,
                             evaluate_fn=lambda hp: run_trial(hp, train_val, trial_cfg),
                             n_init=cfg.n_init, n_iter=cfg.n_iter, seed=seed).run()
        out["bo_mono_cvar"] = _eval_full(b4["best"]["hparams"], data_full, seed, cfg, "sgd")

    tc = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, device=cfg.device,
                          gamma=cfg.gamma, seed=seed)
    if "random_search" in M:
        rs = RandomSearchHPO(cfg.search_space, train_val, n_trials=cfg.n_init + cfg.n_iter,
                             trial_cfg=tc, seed=seed)
        out["random_search"] = _eval_full(rs.run()["best"]["hparams"], data_full, seed, cfg, "sgd")

    if "asha" in M:
        asha = ASHA(cfg.search_space, train_val, n_configs=cfg.asha_configs, r0=4, eta=2,
                    trial_cfg=tc, seed=seed)
        out["asha"] = _eval_full(asha.run()["best"]["hparams"], data_full, seed, cfg, "sgd")

    if "sam" in M:
        tc_sam = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sam",
                                  device=cfg.device, gamma=cfg.gamma, seed=seed)
        rs_sam = RandomSearchHPO(cfg.search_space, train_val, n_trials=cfg.n_init + cfg.n_iter,
                                 trial_cfg=tc_sam, seed=seed)
        out["sam"] = _eval_full(rs_sam.run()["best"]["hparams"], data_full, seed, cfg, "sam")

    logger.info("security seed concluida", seed=seed,
                cvar={k: round(v["cvar_test"], 4) for k, v in out.items()})
    out["_telemetry"] = telemetry
    return out


def run_security_protocol(loader_fn, cfg):
    methods = list(cfg.methods)
    per_seed = []
    for seed in cfg.seeds:
        res = run_security_seed(seed, loader_fn, cfg)
        per_seed.append({"seed": seed, **res})
    # organizar por metrica (so' chaves numericas escalares)
    first = per_seed[0][methods[0]]
    metric_keys = [k for k, v in first.items() if isinstance(v, (int, float))]
    by_metric = {m: {mk: [] for mk in metric_keys} for m in methods}
    for entry in per_seed:
        for m in methods:
            for mk in metric_keys:
                by_metric[m][mk].append(entry[m][mk])
    return {"by_metric": by_metric, "per_seed": per_seed,
            "methods": methods, "seeds": list(cfg.seeds), "metric_keys": metric_keys}
