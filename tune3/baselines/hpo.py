# tune3/baselines/hpo.py
"""
Estrategias de HPO baseline (Fase 5):
  RandomSearchHPO -- amostra configs aleatorias (controle ingenuo)      [RS]
  ASHA            -- successive halving (early-stopping de configs ruins) [ASHA]
  SAM             -- nao e' HPO; e' RandomSearch + otimizador SAM no plain_trial.
  B4 (BO mono-objetivo em CVaR) esta em tune3/baselines/bo_mono.py.

Todos compartilham o MESMO plain_trial, dados e objetivo (CVaR de VALIDACAO)
que o Tune3, garantindo comparacao justa. O espaco de busca e' o mesmo do
runner do Tune3. A avaliacao final no TESTE e' feita pelo protocolo, uma unica
vez, com o modelo da melhor epoca de validacao (ver plain_trial.test_data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig

SearchSpace = Dict[str, Tuple[float, float]]


def _decode(raw: Dict[str, float]) -> Dict[str, float]:
    """Mesma decodificacao do runner do Tune3 (log_lr/log_wd -> lr/wd)."""
    return {
        "learning_rate": float(10.0 ** raw["log_lr"]),
        "weight_decay": float(10.0 ** raw["log_wd"]),
        "dropout": float(raw["dropout"]),
        "hidden_dim": int(round(raw["hidden_dim"])),
        "n_layers": int(round(raw["n_layers"])),
    }


def _sample(space: SearchSpace, rng) -> Dict[str, float]:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in space.items()}


@dataclass
class RandomSearchHPO:
    """B1: amostra n_trials configs; retorna a melhor por CVaR + frente de Pareto."""
    space: SearchSpace
    data: Tuple
    n_trials: int = 13
    trial_cfg: PlainTrialConfig = field(default_factory=PlainTrialConfig)
    seed: int = 0

    def run(self) -> Dict:
        rng = np.random.default_rng(self.seed)
        results = []
        for i in range(self.n_trials):
            raw = _sample(self.space, rng)
            hp = _decode(raw)
            r = plain_trial(hp, self.data, self.trial_cfg)
            results.append({"hparams": hp, "cvar": r["cvar"], "curvature": r["curvature"]})
        best = min(results, key=lambda d: d["cvar"])
        return {"method": "random_search", "n_evaluations": self.n_trials,
                "results": results, "best": best}


@dataclass
class ASHA:
    """
    B2: Successive Halving sincrono. Comeca com n_configs por r0 epocas; mantem
    top 1/eta; multiplica orcamento por eta; repete ate sobrar 1 config.
    Conta o orcamento total gasto (em epocas) para comparacao de custo justa.
    """
    space: SearchSpace
    data: Tuple
    n_configs: int = 16
    r0: int = 4              # orcamento inicial (epocas)
    eta: int = 2            # fator de reducao
    trial_cfg: PlainTrialConfig = field(default_factory=PlainTrialConfig)
    seed: int = 0

    def run(self) -> Dict:
        rng = np.random.default_rng(self.seed)
        configs = [_decode(_sample(self.space, rng)) for _ in range(self.n_configs)]
        budget = self.r0
        total_epochs = 0
        survivors = configs
        while len(survivors) > 1:
            scored = []
            for hp in survivors:
                r = plain_trial(hp, self.data, self.trial_cfg, max_epochs_override=budget)
                total_epochs += budget
                scored.append((r["cvar"], hp, r))
            scored.sort(key=lambda t: t[0])
            keep = max(1, len(survivors) // self.eta)
            survivors = [hp for _, hp, _ in scored[:keep]]
            budget *= self.eta
        # avaliacao final do sobrevivente com orcamento cheio
        final_hp = survivors[0]
        r = plain_trial(final_hp, self.data, self.trial_cfg)
        total_epochs += self.trial_cfg.max_epochs
        return {"method": "asha", "n_configs": self.n_configs,
                "total_epochs": total_epochs,
                "best": {"hparams": final_hp, "cvar": r["cvar"], "curvature": r["curvature"]}}
