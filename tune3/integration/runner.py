# tune3/integration/runner.py  (ATUALIZADO Fase 8 + auditoria set/2026)
"""Runner do Tune3 -- conecta MACRO ao trial. Reporta DUAS selecoes:
knee point (equilibrio CVaR/curvatura) e best_cvar (Pareto de melhor CVaR).

NOVO (E1 -- controle placebo): `Tune3RunConfig.objective2` escolhe o 2o objetivo
do nivel MACRO:
  - "curvature" (padrao): Tr(H^2) medido no trial -- o Tune3 real.
  - "random":  Z ~ N(0,1) i.i.d. por trial, no lugar da curvatura. Se o placebo
    empata com o Tune3 real em CVaR de teste, o ganho vem da EXPLORACAO
    bi-objetivo, nao da geometria (cenario C2 da arvore de decisao).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import structlog

from tune3.macro.bo_loop import Tune3MacroLoop, MacroConfig
from tune3.integration.trial import run_trial, TrialConfig

logger = structlog.get_logger()

DEFAULT_SEARCH_SPACE = {
    "log_lr": (-5.0, -2.0), "log_wd": (-6.0, -2.0),
    "dropout": (0.0, 0.5), "hidden_dim": (32.0, 256.0), "n_layers": (1.0, 4.0),
}


@dataclass
class Tune3RunConfig:
    macro: MacroConfig = field(default_factory=lambda: MacroConfig(n_init=8, n_iter=20))
    trial: TrialConfig = field(default_factory=TrialConfig)
    search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_SEARCH_SPACE))
    objective2: str = "curvature"   # "curvature" | "random" (placebo, E1)

    def __post_init__(self):
        if self.objective2 not in ("curvature", "random"):
            raise ValueError("objective2 deve ser 'curvature' ou 'random'")


def _decode(raw):
    return {"learning_rate": float(10.0 ** raw["log_lr"]),
            "weight_decay": float(10.0 ** raw["log_wd"]),
            "dropout": float(raw["dropout"]),
            "hidden_dim": int(round(raw["hidden_dim"])),
            "n_layers": int(round(raw["n_layers"]))}


def run_tune3(data, config=None, wandb_run=None):
    cfg = config or Tune3RunConfig()
    placebo_rng = np.random.default_rng(10_000 + cfg.macro.seed)
    trial_log = []   # telemetria por trial (E2: engajamento do DDKF)

    def evaluate(raw):
        hp = _decode(raw); res = run_trial(hp, data, cfg.trial, wandb_run=wandb_run)
        trial_log.append({"hparams": hp, "cvar": res["cvar"], "curvature": res["curvature"],
                          "aborted": res["aborted"],
                          "regime_fractions": res.get("regime_fractions"),
                          "n_eos_triggers": res.get("n_eos_triggers"),
                          "n_ddkf_active_epochs": res.get("n_ddkf_active_epochs"),
                          "n_epochs_run": res.get("n_epochs_run"),
                          "lr_change_fraction": res.get("lr_change_fraction"),
                          "lr0": res.get("lr0"), "lr_final": res.get("lr_final"),
                          # E2: por que o DDKF (nao) atua. n_ddkf_active_epochs sozinho nao
                          # distingue "buffer curto" de "gate fechado"; r2_max e
                          # n_ddkf_observations distinguem.
                          "r2_max": res.get("r2_max"), "r2_mean": res.get("r2_mean"),
                          "n_ddkf_observations": res.get("n_ddkf_observations"),
                          "n_curvature_estimates": res.get("n_curvature_estimates"),
                          "obs_vector": res.get("obs_vector")})
        if cfg.objective2 == "random":
            second = float(placebo_rng.normal())      # placebo: ruido no lugar da curvatura
        else:
            second = res["curvature"]
        return res["cvar"], second

    loop = Tune3MacroLoop(cfg.search_space, evaluate, cfg.macro)
    pareto = loop.run()
    knee = loop.knee_point()
    best_cvar = loop.best_objective_point(obj_index=0)   # melhor CVaR (metrica de deploy)
    return {"pareto": pareto,
            "objective2": cfg.objective2,
            "knee_raw": knee, "knee_hparams": _decode(knee["x"]) if knee else {},
            "best_cvar_raw": best_cvar,
            "best_cvar_hparams": _decode(best_cvar["x"]) if best_cvar else {},
            "trials": trial_log}
