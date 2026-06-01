# tune3/baselines/sam.py
"""
SAM -- Sharpness-Aware Minimization (Foret et al. 2021).

Baseline de PLANURA NO ESPACO DE PESOS -- o contraste direto com o Tune3, que
atua no espaco de HIPERPARAMETROS. SAM faz dois passos por minibatch:
  1) sobe para w + eps, onde eps = rho * g/||g||  (pior vizinhanca);
  2) calcula o gradiente em w+eps e aplica em w (desce a partir do topo).
Isso busca minimos planos. Comparar Tune3 vs SAM mostra que a contribuicao do
Tune3 (controle de curvatura via hiperparametros) e' complementar, nao redundante.
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        if rho < 0:
            raise ValueError("rho deve ser >= 0")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        norms = [p.grad.norm(2) for g in self.param_groups for p in g["params"]
                 if p.grad is not None]
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), 2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)                       # sobe para w + eps
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])      # volta para w
        self.base_optimizer.step()                # passo real do otimizador base
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        # SAM precisa de dois forwards; use first_step/second_step explicitamente.
        raise RuntimeError("Use first_step()/second_step() com SAM, nao step().")
