# tune3/baselines/bo_mono.py
"""
B4 -- Otimizacao Bayesiana MONO-objetivo (GP + log-EI) apenas sobre CVaR.

E' o baseline decisivo da Linha 1 (E1): separa "o mecanismo geometrico
funciona" de "otimizacao bayesiana simplesmente explora melhor o espaco".

Desenho (para a comparacao ser limpa, B4 difere do Tune3 SOMENTE no objetivo):
  - MESMO GP (SingleTaskGP, Normalize + Standardize) e MESMO orcamento
    (n_init pontos Sobol + n_iter iteracoes) do nivel MACRO do Tune3;
  - MESMO espaco de busca e MESMA decodificacao (log_lr, log_wd, ...);
  - MESMA funcao de treino interna (`evaluate_fn`), que por padrao e' o
    `run_trial` do Tune3 (com DDKF/Cantelli/EoS), de modo que a unica
    diferenca em relacao ao Tune3 e' minimizar CVaR SOZINHO em vez do par
    (CVaR, Tr(H^2)). Se preferir um B4 "puro" (sem DDKF), passe
    evaluate_fn baseado em plain_trial.
  - Aquisicao: qLogExpectedImprovement (Ament et al. 2023), a versao
    numericamente estavel do EI classico.

Retorno: {"method": "bo_mono_cvar", "n_evaluations", "results", "best"}, no
mesmo formato de RandomSearchHPO/ASHA (best = menor CVaR de VALIDACAO).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import structlog

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

logger = structlog.get_logger()
SearchSpace = Dict[str, Tuple[float, float]]
EvalFn = Callable[[Dict[str, float]], Dict]   # hparams decodificados -> dict com "cvar"


def decode_hparams(raw: Dict[str, float]) -> Dict[str, float]:
    """Mesma decodificacao do runner do Tune3 (log_lr/log_wd -> lr/wd)."""
    return {
        "learning_rate": float(10.0 ** raw["log_lr"]),
        "weight_decay": float(10.0 ** raw["log_wd"]),
        "dropout": float(raw["dropout"]),
        "hidden_dim": int(round(raw["hidden_dim"])),
        "n_layers": int(round(raw["n_layers"])),
    }


@dataclass
class MonoObjectiveBO:
    space: SearchSpace
    evaluate_fn: EvalFn                 # recebe hparams DECODIFICADOS, devolve dict com "cvar"
    n_init: int = 8
    n_iter: int = 20
    mc_samples: int = 128
    num_restarts: int = 10
    raw_samples: int = 256
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        self.names = list(self.space.keys()); self.dim = len(self.names)
        self._bounds = torch.tensor(
            [[self.space[n][0] for n in self.names], [self.space[n][1] for n in self.names]],
            dtype=torch.double, device=self.device)
        self.train_X = torch.empty((0, self.dim), dtype=torch.double, device=self.device)
        self.train_Y = torch.empty((0, 1), dtype=torch.double, device=self.device)
        self.results: List[Dict] = []

    def _x_to_raw(self, x) -> Dict[str, float]:
        return {n: float(x[i]) for i, n in enumerate(self.names)}

    def _observe(self, X: torch.Tensor):
        ys = []
        for i in range(X.shape[0]):
            hp = decode_hparams(self._x_to_raw(X[i]))
            r = self.evaluate_fn(hp)
            c = float(r["cvar"])
            self.results.append({"hparams": hp, "cvar": c, "curvature": float(r.get("curvature", np.nan)),
                                 "aborted": bool(r.get("aborted", False))})
            ys.append([-c])                      # BoTorch MAXIMIZA -> usamos -CVaR
        Y = torch.tensor(ys, dtype=torch.double, device=self.device)
        self.train_X = torch.cat([self.train_X, X], 0)
        self.train_Y = torch.cat([self.train_Y, Y], 0)

    def _sobol_init(self) -> torch.Tensor:
        from torch.quasirandom import SobolEngine
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        unit = sobol.draw(self.n_init).to(dtype=torch.double, device=self.device)
        lo, hi = self._bounds[0], self._bounds[1]
        return lo + (hi - lo) * unit

    def _fit(self):
        model = SingleTaskGP(self.train_X, self.train_Y,
                             input_transform=Normalize(d=self.dim, bounds=self._bounds),
                             outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return model

    def _suggest(self, model) -> torch.Tensor:
        acqf = qLogExpectedImprovement(
            model=model, best_f=self.train_Y.max(),
            sampler=SobolQMCNormalSampler(torch.Size([self.mc_samples])))
        cand, _ = optimize_acqf(acqf, bounds=self._bounds, q=1,
                                num_restarts=self.num_restarts, raw_samples=self.raw_samples)
        return cand.detach()

    def run(self) -> Dict:
        torch.manual_seed(self.seed)
        self._observe(self._sobol_init())
        for it in range(self.n_iter):
            self._observe(self._suggest(self._fit()))
        best = min(self.results, key=lambda d: d["cvar"])
        logger.info("B4 (BO mono-objetivo em CVaR) concluido",
                    n_evaluations=len(self.results), best_cvar=round(best["cvar"], 4))
        return {"method": "bo_mono_cvar", "n_evaluations": len(self.results),
                "results": self.results, "best": best}
