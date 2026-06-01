# tune3/macro/bo_loop.py
"""
Nivel MACRO do Tune3 -- otimizacao Bayesiana MULTI-OBJETIVO via BoTorch.

Objetivos (ambos MINIMIZADOS):
    f1 = CVaR_gamma(perda)     -- risco de cauda (SecOps)
    f2 = Tr(H^2)               -- curvatura (planura/robustez)

Surrogate: um GP por objetivo (ModelListGP de SingleTaskGP).
Aquisicao:  qLogNoisyExpectedHypervolumeImprovement (cEHVI / qLogNEHVI),
            que e' exatamente a aquisicao de Daulton et al. 2020/2021 citada
            no manual. Suporta ruido de observacao e restricoes de saida.

CONVENCAO: BoTorch MAXIMIZA. Como queremos MINIMIZAR (CVaR, Tr(H^2)), passamos
Y_max = -Y (negacao). O ref_point e' definido no espaco negado, abaixo (pior que)
todos os pontos observados.

Uso (a funcao de avaliacao sera ligada ao treino real na Fase 4):
    def evaluate(x: dict) -> tuple[float, float]:
        # roda um trial de treino com os hiperparametros x
        # retorna (cvar, curvatura)
        ...
    loop = Tune3MacroLoop(search_space, evaluate)
    pareto = loop.run(n_init=8, n_iter=20)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.sampling.normal import SobolQMCNormalSampler

import structlog

logger = structlog.get_logger()

# tipo: espaco de busca = {nome: (low, high)} em escala NATURAL do hiperparametro
SearchSpace = Dict[str, Tuple[float, float]]
EvalFn = Callable[[Dict[str, float]], Tuple[float, float]]


@dataclass
class MacroConfig:
    n_init: int = 8           # pontos iniciais (Sobol)
    n_iter: int = 20          # iteracoes de BO
    batch_q: int = 1          # candidatos por iteracao (q)
    mc_samples: int = 128     # amostras MC do sampler
    num_restarts: int = 10    # reinicios da otimizacao da aquisicao
    raw_samples: int = 256    # amostras brutas p/ inicializacao
    ref_point_eps: float = 0.1  # folga relativa do ref_point abaixo do pior ponto
    seed: int = 42
    device: str = "cpu"       # MACRO roda em CPU; o TREINO (Fase 4) usa a GPU


class Tune3MacroLoop:
    def __init__(self, search_space: SearchSpace, evaluate: EvalFn,
                 config: MacroConfig | None = None):
        self.space = search_space
        self.evaluate = evaluate
        self.cfg = config or MacroConfig()
        self.names = list(search_space.keys())
        self.dim = len(self.names)
        self._bounds = torch.tensor(
            [[search_space[n][0] for n in self.names],
             [search_space[n][1] for n in self.names]],
            dtype=torch.double, device=self.cfg.device,
        )  # (2, d)
        self.train_X = torch.empty((0, self.dim), dtype=torch.double, device=self.cfg.device)
        self.train_Y = torch.empty((0, 2), dtype=torch.double, device=self.cfg.device)  # (cvar, curv)
        torch.manual_seed(self.cfg.seed)

    # ------------------------------------------------------------------ #
    def _x_to_dict(self, x_row: torch.Tensor) -> Dict[str, float]:
        return {n: float(x_row[i]) for i, n in enumerate(self.names)}

    def _sobol_init(self) -> torch.Tensor:
        from torch.quasirandom import SobolEngine
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.cfg.seed)
        unit = sobol.draw(self.cfg.n_init).to(dtype=torch.double, device=self.cfg.device)  # (n,d) em [0,1]
        lo, hi = self._bounds[0], self._bounds[1]
        return lo + (hi - lo) * unit

    def _observe(self, X: torch.Tensor) -> None:
        """Avalia cada linha de X e acumula em train_X/train_Y."""
        ys = []
        for i in range(X.shape[0]):
            xd = self._x_to_dict(X[i])
            cvar_val, curv_val = self.evaluate(xd)
            ys.append([cvar_val, curv_val])
            logger.info("MACRO eval", x=xd, cvar=round(cvar_val, 5), curv=round(curv_val, 5))
        Y = torch.tensor(ys, dtype=torch.double, device=self.cfg.device)
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_Y = torch.cat([self.train_Y, Y], dim=0)

    def _fit_model(self) -> ModelListGP:
        # BoTorch MAXIMIZA: negamos Y (queremos minimizar ambos os objetivos)
        Y_max = -self.train_Y
        models = []
        for j in range(2):
            yj = Y_max[:, j: j + 1]
            models.append(
                SingleTaskGP(
                    self.train_X, yj,
                    input_transform=Normalize(d=self.dim, bounds=self._bounds),
                    outcome_transform=Standardize(m=1),
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _ref_point(self) -> torch.Tensor:
        """Ref point no espaco MAXIMIZADO (negado), abaixo do pior ponto observado."""
        Y_max = -self.train_Y
        worst = Y_max.min(dim=0).values            # pior (menor) por objetivo
        span = Y_max.max(dim=0).values - worst
        span = torch.clamp(span, min=1e-6)
        return worst - self.cfg.ref_point_eps * span

    def _suggest(self, model: ModelListGP) -> torch.Tensor:
        sampler = SobolQMCNormalSampler(torch.Size([self.cfg.mc_samples]))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self._ref_point(),
            X_baseline=self.train_X,
            sampler=sampler,
            prune_baseline=True,
        )
        cand, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self._bounds,
            q=self.cfg.batch_q,
            num_restarts=self.cfg.num_restarts,
            raw_samples=self.cfg.raw_samples,
        )
        return cand.detach()

    # ------------------------------------------------------------------ #
    def run(self) -> Dict:
        """Executa o loop completo e retorna a frente de Pareto."""
        self._observe(self._sobol_init())
        for it in range(self.cfg.n_iter):
            model = self._fit_model()
            cand = self._suggest(model)
            self._observe(cand)
            logger.info("MACRO iter", it=it + 1, n_obs=self.train_X.shape[0])
        return self.pareto_front()

    def pareto_front(self) -> Dict:
        """Pontos nao-dominados (no espaco de MINIMIZACAO original)."""
        Y_max = -self.train_Y
        mask = is_non_dominated(Y_max)
        idx = torch.where(mask)[0]
        pf_X = self.train_X[idx]
        pf_Y = self.train_Y[idx]   # de volta ao espaco original (cvar, curv)
        return {
            "X": [self._x_to_dict(pf_X[i]) for i in range(pf_X.shape[0])],
            "objectives": pf_Y.cpu().numpy().tolist(),  # [[cvar, curv], ...]
            "names": ["cvar", "curvature"],
            "n_evaluations": int(self.train_X.shape[0]),
        }

    def knee_point(self) -> Dict:
        """Seleciona o ponto 'joelho' por norma L1 normalizada (compromisso)."""
        pf = self.pareto_front()
        Y = np.asarray(pf["objectives"])
        if len(Y) == 0:
            return {}
        lo = Y.min(axis=0); hi = Y.max(axis=0)
        span = np.clip(hi - lo, 1e-12, None)
        norm = (Y - lo) / span
        l1 = norm.sum(axis=1)
        k = int(np.argmin(l1))
        return {"x": pf["X"][k], "objectives": pf["objectives"][k]}
