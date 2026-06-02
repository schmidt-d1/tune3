# tune3/integration/runner.py  (ATUALIZADO Fase 8)
"""Runner do Tune3 -- conecta MACRO ao trial. Agora reporta DUAS selecoes:
knee point (equilibrio CVaR/curvatura) e best_cvar (Pareto de melhor CVaR)."""
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


def _decode(raw):
    return {"learning_rate": float(10.0 ** raw["log_lr"]),
            "weight_decay": float(10.0 ** raw["log_wd"]),
            "dropout": float(raw["dropout"]),
            "hidden_dim": int(round(raw["hidden_dim"])),
            "n_layers": int(round(raw["n_layers"]))}


def run_tune3(data, config=None, wandb_run=None):
    cfg = config or Tune3RunConfig()

    def evaluate(raw):
        hp = _decode(raw); res = run_trial(hp, data, cfg.trial, wandb_run=wandb_run)
        return res["cvar"], res["curvature"]

    loop = Tune3MacroLoop(cfg.search_space, evaluate, cfg.macro)
    pareto = loop.run()
    knee = loop.knee_point()
    best_cvar = loop.best_objective_point(obj_index=0)   # NOVO: melhor CVaR
    return {"pareto": pareto,
            "knee_raw": knee, "knee_hparams": _decode(knee["x"]) if knee else {},
            "best_cvar_raw": best_cvar,
            "best_cvar_hparams": _decode(best_cvar["x"]) if best_cvar else {}}
