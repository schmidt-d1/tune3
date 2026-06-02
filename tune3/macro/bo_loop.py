# tune3/macro/bo_loop.py
"""Nivel MACRO do Tune3 -- BO multi-objetivo via BoTorch (GP + cEHVI).
ATUALIZADO (Fase 8): seleciona tanto o knee point (equilibrio) quanto o ponto
de Pareto de MELHOR CVaR (alinhado a metrica de deploy)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.sampling.normal import SobolQMCNormalSampler
import structlog

logger = structlog.get_logger()
SearchSpace = Dict[str, Tuple[float, float]]
EvalFn = Callable[[Dict[str, float]], Tuple[float, float]]


@dataclass
class MacroConfig:
    n_init: int = 8
    n_iter: int = 20
    batch_q: int = 1
    mc_samples: int = 128
    num_restarts: int = 10
    raw_samples: int = 256
    ref_point_eps: float = 0.1
    seed: int = 42
    device: str = "cpu"


class Tune3MacroLoop:
    def __init__(self, search_space, evaluate, config=None):
        self.space = search_space; self.evaluate = evaluate
        self.cfg = config or MacroConfig()
        self.names = list(search_space.keys()); self.dim = len(self.names)
        self._bounds = torch.tensor(
            [[search_space[n][0] for n in self.names],
             [search_space[n][1] for n in self.names]],
            dtype=torch.double, device=self.cfg.device)
        self.train_X = torch.empty((0, self.dim), dtype=torch.double, device=self.cfg.device)
        self.train_Y = torch.empty((0, 2), dtype=torch.double, device=self.cfg.device)
        torch.manual_seed(self.cfg.seed)

    def _x_to_dict(self, x): return {n: float(x[i]) for i, n in enumerate(self.names)}

    def _sobol_init(self):
        from torch.quasirandom import SobolEngine
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.cfg.seed)
        unit = sobol.draw(self.cfg.n_init).to(dtype=torch.double, device=self.cfg.device)
        lo, hi = self._bounds[0], self._bounds[1]
        return lo + (hi - lo) * unit

    def _observe(self, X):
        ys = []
        for i in range(X.shape[0]):
            xd = self._x_to_dict(X[i]); c, k = self.evaluate(xd); ys.append([c, k])
        Y = torch.tensor(ys, dtype=torch.double, device=self.cfg.device)
        self.train_X = torch.cat([self.train_X, X], 0); self.train_Y = torch.cat([self.train_Y, Y], 0)

    def _fit_model(self):
        Y_max = -self.train_Y; models = []
        for j in range(2):
            models.append(SingleTaskGP(self.train_X, Y_max[:, j:j+1],
                          input_transform=Normalize(d=self.dim, bounds=self._bounds),
                          outcome_transform=Standardize(m=1)))
        model = ModelListGP(*models)
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        return model

    def _ref_point(self):
        Y_max = -self.train_Y; worst = Y_max.min(0).values
        span = torch.clamp(Y_max.max(0).values - worst, min=1e-6)
        return worst - self.cfg.ref_point_eps * span

    def _suggest(self, model):
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=self._ref_point(), X_baseline=self.train_X,
            sampler=SobolQMCNormalSampler(torch.Size([self.cfg.mc_samples])), prune_baseline=True)
        cand, _ = optimize_acqf(acqf, bounds=self._bounds, q=self.cfg.batch_q,
                                num_restarts=self.cfg.num_restarts, raw_samples=self.cfg.raw_samples)
        return cand.detach()

    def run(self):
        self._observe(self._sobol_init())
        for it in range(self.cfg.n_iter):
            self._observe(self._suggest(self._fit_model()))
        return self.pareto_front()

    def pareto_front(self):
        Y_max = -self.train_Y; mask = is_non_dominated(Y_max); idx = torch.where(mask)[0]
        pf_X = self.train_X[idx]; pf_Y = self.train_Y[idx]
        return {"X": [self._x_to_dict(pf_X[i]) for i in range(pf_X.shape[0])],
                "objectives": pf_Y.cpu().numpy().tolist(),
                "names": ["cvar", "curvature"], "n_evaluations": int(self.train_X.shape[0])}

    def knee_point(self):
        pf = self.pareto_front(); Y = np.asarray(pf["objectives"])
        if len(Y) == 0: return {}
        lo, hi = Y.min(0), Y.max(0); span = np.clip(hi - lo, 1e-12, None)
        k = int(np.argmin(((Y - lo) / span).sum(1)))
        return {"x": pf["X"][k], "objectives": pf["objectives"][k]}

    def best_objective_point(self, obj_index: int = 0):
        """Ponto da fronteira de Pareto que MINIMIZA o objetivo obj_index
        (0=CVaR, 1=curvatura). Alinha a selecao a metrica de deploy."""
        pf = self.pareto_front(); Y = np.asarray(pf["objectives"])
        if len(Y) == 0: return {}
        k = int(np.argmin(Y[:, obj_index]))
        return {"x": pf["X"][k], "objectives": pf["objectives"][k]}
