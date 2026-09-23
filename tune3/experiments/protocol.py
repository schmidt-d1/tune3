# tune3/experiments/protocol.py  (ATUALIZADO Fase 8 + auditoria set/2026)
"""Orquestracao experimental do Tune3 (H1, in-distribution).

METODOS (todos com o MESMO orcamento de n_init + n_iter avaliacoes, exceto ASHA,
que conta orcamento em epocas):
  tune3                   : knee point da frente de Pareto (CVaR, Tr(H^2))
  tune3_bestcvar          : ponto de Pareto de melhor CVaR (metrica de deploy)
  tune3_noddkf_bestcvar   : E2 -- Tune3 com DDKF DESLIGADO (lr fixo); isola o meta-scheduler
  tune3_placebo_bestcvar  : E1 -- 2o objetivo = ruido N(0,1) no lugar da curvatura
  bo_mono_cvar            : E1 -- B4: BO mono-objetivo (GP + logEI) so' em CVaR
  random_search, asha, sam: baselines classicos

CORRECAO (set/2026): a avaliacao final treina com (treino, VALIDACAO) -- early
stopping e escolha de epoca na validacao -- e so' entao avalia o modelo escolhido
no TESTE, uma unica vez. Antes, o teste era passado como "validacao" e a melhor
epoca era escolhida olhando o teste (vazamento).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import structlog

from tune3.integration.runner import run_tune3, Tune3RunConfig
from tune3.integration.trial import TrialConfig, run_trial
from tune3.micro import DDKFConfig
from tune3.macro.bo_loop import MacroConfig
from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
from tune3.baselines.hpo import RandomSearchHPO, ASHA
from tune3.baselines.bo_mono import MonoObjectiveBO

logger = structlog.get_logger()

ALL_METHODS = ["tune3", "tune3_bestcvar", "tune3_noddkf_bestcvar", "tune3_placebo_bestcvar",
               "bo_mono_cvar", "random_search", "asha", "sam"]
CORE_METHODS = ["tune3", "tune3_bestcvar", "random_search", "asha", "sam"]


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
    # Cadencia de re-estimativa da curvatura dentro do trial (5 = padrao; 1 = a cada
    # epoca, mais caro). Exposto para medir o custo dos dois regimes (decisao D-C).
    curvature_every: int = 5
    # Braco de ablacao do vetor de observacao do DDKF: "loss" | "curvature" (D1).
    obs_vector: str = "loss"
    # Amplitude da excitacao do log-lr na identificacao. O padrao do projeto (0.03) deixa
    # o sistema NAO identificavel (R^2 ~ 0.005 << r2_low=0.10) e o DDKF nunca atua; com
    # 0.30 o gate abre. E' o parametro que torna o E2 uma pergunta e nao uma tautologia.
    ddkf_exploration_std: float = 0.03
    methods: List[str] = field(default_factory=lambda: list(ALL_METHODS))
    search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "log_lr": (-5.0, -2.0), "log_wd": (-6.0, -2.0),
        "dropout": (0.0, 0.5), "hidden_dim": (32.0, 256.0), "n_layers": (1.0, 4.0)})

    def __post_init__(self):
        bad = [m for m in self.methods if m not in ALL_METHODS]
        if bad:
            raise ValueError(f"metodos desconhecidos: {bad}; validos: {ALL_METHODS}")


def _eval_on_test(hparams, data_full, seed, cfg, optimizer="sgd"):
    """Treina em (treino, val) com early stopping na VAL e avalia uma vez no TESTE."""
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    tc = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer=optimizer,
                          device=cfg.device, gamma=cfg.gamma, seed=seed)
    r = plain_trial(hparams, (X_tr, y_tr, X_val, y_val), tc, test_data=(X_te, y_te))
    return {"cvar_test": r["cvar_test"], "cvar_val": r["cvar"],
            "curvature_test": r["curvature"],      # Tr(H^2) do modelo escolhido (batch de treino)
            "test_loss": r["test_loss"], "best_epoch": r["best_epoch"], "hparams": hparams}


def _ddkf_cfg(cfg):
    return DDKFConfig(rho=0.95, sigma_eta=0.05, exploration_std=cfg.ddkf_exploration_std)


def _tune3_variant(train_val, cfg, seed, objective2="curvature", ddkf_enabled=True):
    run_cfg = Tune3RunConfig(
        macro=MacroConfig(n_init=cfg.n_init, n_iter=cfg.n_iter, device="cpu", seed=seed),
        trial=TrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sgd",
                          device=cfg.device, gamma=cfg.gamma, seed=seed, ddkf_enabled=ddkf_enabled,
                          curvature_every=cfg.curvature_every, obs_vector=cfg.obs_vector,
                          ddkf=_ddkf_cfg(cfg)),
        search_space=dict(cfg.search_space), objective2=objective2)
    return run_tune3(train_val, run_cfg)


def run_seed(seed, loader_fn, cfg):
    data_full = loader_fn(seed)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_full
    train_val = (X_tr, y_tr, X_val, y_val)
    out = {}; telemetry = {}
    M = set(cfg.methods)

    if M & {"tune3", "tune3_bestcvar"}:
        t3 = _tune3_variant(train_val, cfg, seed)
        telemetry["tune3"] = t3["trials"]
        if "tune3" in M:
            out["tune3"] = _eval_on_test(t3["knee_hparams"], data_full, seed, cfg, "sgd")
        if "tune3_bestcvar" in M:
            out["tune3_bestcvar"] = _eval_on_test(t3["best_cvar_hparams"], data_full, seed, cfg, "sgd")

    if "tune3_noddkf_bestcvar" in M:                       # E2
        t3n = _tune3_variant(train_val, cfg, seed, ddkf_enabled=False)
        telemetry["tune3_noddkf"] = t3n["trials"]
        out["tune3_noddkf_bestcvar"] = _eval_on_test(t3n["best_cvar_hparams"], data_full, seed, cfg, "sgd")

    if "tune3_placebo_bestcvar" in M:                      # E1 (placebo)
        t3p = _tune3_variant(train_val, cfg, seed, objective2="random")
        telemetry["tune3_placebo"] = t3p["trials"]
        out["tune3_placebo_bestcvar"] = _eval_on_test(t3p["best_cvar_hparams"], data_full, seed, cfg, "sgd")

    if "bo_mono_cvar" in M:                                # E1 (B4)
        trial_cfg = TrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sgd",
                                device=cfg.device, gamma=cfg.gamma, seed=seed,
                                curvature_every=cfg.curvature_every, obs_vector=cfg.obs_vector,
                                ddkf=_ddkf_cfg(cfg))
        b4 = MonoObjectiveBO(cfg.search_space,
                             evaluate_fn=lambda hp: run_trial(hp, train_val, trial_cfg),
                             n_init=cfg.n_init, n_iter=cfg.n_iter, seed=seed).run()
        out["bo_mono_cvar"] = _eval_on_test(b4["best"]["hparams"], data_full, seed, cfg, "sgd")

    tc = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, device=cfg.device,
                          gamma=cfg.gamma, seed=seed)
    if "random_search" in M:
        rs = RandomSearchHPO(cfg.search_space, train_val, n_trials=cfg.n_init + cfg.n_iter,
                             trial_cfg=tc, seed=seed)
        out["random_search"] = _eval_on_test(rs.run()["best"]["hparams"], data_full, seed, cfg, "sgd")

    if "asha" in M:
        asha = ASHA(cfg.search_space, train_val, n_configs=cfg.asha_configs, r0=4, eta=2,
                    trial_cfg=tc, seed=seed)
        out["asha"] = _eval_on_test(asha.run()["best"]["hparams"], data_full, seed, cfg, "sgd")

    if "sam" in M:
        tc_sam = PlainTrialConfig(max_epochs=cfg.epochs, patience=cfg.patience, optimizer="sam",
                                  device=cfg.device, gamma=cfg.gamma, seed=seed)
        rs_sam = RandomSearchHPO(cfg.search_space, train_val, n_trials=cfg.n_init + cfg.n_iter,
                                 trial_cfg=tc_sam, seed=seed)
        out["sam"] = _eval_on_test(rs_sam.run()["best"]["hparams"], data_full, seed, cfg, "sam")

    logger.info("seed concluida", seed=seed,
                cvar={k: round(v["cvar_test"], 4) for k, v in out.items()})
    out["_telemetry"] = telemetry
    return out


def run_protocol(loader_fn, cfg):
    methods = list(cfg.methods)
    scores = {m: [] for m in methods}; per_seed = []
    for seed in cfg.seeds:
        res = run_seed(seed, loader_fn, cfg); per_seed.append({"seed": seed, **res})
        for m in methods: scores[m].append(res[m]["cvar_test"])
    return {"cvar_test_by_method": scores, "per_seed": per_seed,
            "seeds": list(cfg.seeds), "methods": methods}
