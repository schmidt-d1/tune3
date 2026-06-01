# tune3/integration/runner.py
"""
Runner do Tune3 -- conecta o nivel MACRO (BO) ao trial de treino (MICRO).

Define o espaco de busca, cria a funcao evaluate(hparams)->(cvar, curvatura)
que o Tune3MacroLoop chama, e executa o loop completo. Este e' o ponto de
entrada que roda o Tune3 de ponta a ponta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import structlog

from tune3.macro.bo_loop import Tune3MacroLoop, MacroConfig
from tune3.integration.trial import run_trial, TrialConfig

logger = structlog.get_logger()

# Espaco de busca PADRAO (escala natural). lr e wd serao tratados em log pelo usuario
# do dicionario; aqui definimos limites diretos e deixamos o BO amostrar linearmente
# nesses limites -- para lr/wd convem amostrar em log, entao definimos os limites
# ja como expoentes e exponenciamos no wrapper (ver evaluate).
DEFAULT_SEARCH_SPACE = {
    "log_lr":      (-5.0, -2.0),   # lr = 10^x  => [1e-5, 1e-2]
    "log_wd":      (-6.0, -2.0),   # wd = 10^x  => [1e-6, 1e-2]
    "dropout":     (0.0, 0.5),
    "hidden_dim":  (32.0, 256.0),
    "n_layers":    (1.0, 4.0),
}


@dataclass
class Tune3RunConfig:
    macro: MacroConfig = field(default_factory=lambda: MacroConfig(n_init=8, n_iter=20))
    trial: TrialConfig = field(default_factory=TrialConfig)
    search_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_SEARCH_SPACE))


def _decode(raw: Dict[str, float]) -> Dict[str, float]:
    """Converte os parametros de busca (com log) nos hiperparametros reais."""
    return {
        "learning_rate": float(10.0 ** raw["log_lr"]),
        "weight_decay": float(10.0 ** raw["log_wd"]),
        "dropout": float(raw["dropout"]),
        "hidden_dim": int(round(raw["hidden_dim"])),
        "n_layers": int(round(raw["n_layers"])),
    }


def run_tune3(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: Optional[Tune3RunConfig] = None,
    wandb_run=None,
) -> Dict:
    """
    Executa o Tune3 completo sobre (X_train, y_train, X_val, y_val).
    Retorna a frente de Pareto + knee point.
    """
    cfg = config or Tune3RunConfig()

    def evaluate(raw_hparams: Dict[str, float]) -> Tuple[float, float]:
        hp = _decode(raw_hparams)
        res = run_trial(hp, data, cfg.trial, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.log({"trial_cvar": res["cvar"], "trial_curvature": res["curvature"],
                           **{f"hp_{k}": v for k, v in hp.items()}})
        # MACRO minimiza ambos
        return res["cvar"], res["curvature"]

    loop = Tune3MacroLoop(cfg.search_space, evaluate, cfg.macro)
    pareto = loop.run()
    knee = loop.knee_point()
    # decodifica os hiperparametros do knee e da frente para escala real
    knee_real = _decode(knee["x"]) if knee else {}
    logger.info("Tune3 concluido", n_eval=pareto["n_evaluations"],
                pareto_size=len(pareto["X"]), knee=knee_real)
    return {"pareto": pareto, "knee_raw": knee, "knee_hparams": knee_real}
