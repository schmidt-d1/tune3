# tune3/curvature/sharpness.py
"""
Sharpness ADAPTATIVA de pior caso (Kwon et al., ICML 2021; Andriushchenko et al., ICML 2023).

    S_rho(w) = max_{ ||u||_2 <= rho }  L(w + |w| (.) u)  -  L(w)

A perturbacao de cada parametro e' proporcional ao proprio tamanho do parametro. Consequencia: em
redes ReLU, reescalar uma camada por a e a seguinte por 1/a (a mesma funcao) NAO muda S_rho --
ao contrario de Tr(H^2), que muda (Dinh et al., 2017). Por isso ela entra na D2 ao lado da nossa
Tr(H^2): se a Tr(H^2) falhar e esta funcionar, o problema e' a metrica, nao a hipotese.

O maximo e' aproximado por subida de gradiente projetada (PGD) no espaco de u, com passos
normalizados e projecao na bola L2 de raio rho; devolve o maior aumento de perda visto.
Medida em model.eval() (sem dropout), num lote fixo -- a mesma convencao da Tr(H^2).
"""
from __future__ import annotations

from typing import Callable

import torch


def adaptive_sharpness(model: torch.nn.Module, loss_fn: Callable[[torch.nn.Module], torch.Tensor],
                       rho: float = 0.1, steps: int = 10, step_size: float | None = None) -> float:
    """S_rho adaptativa (L2) por PGD. `loss_fn(model)` devolve a perda escalar no lote fixo.
    Os parametros do modelo sao restaurados ao final, mesmo em caso de erro."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or rho <= 0:
        return 0.0
    alpha = step_size if step_size is not None else 2.5 * rho / max(steps, 1)
    w0 = [p.detach().clone() for p in params]
    scale = [w.abs() for w in w0]
    u = [torch.zeros_like(w) for w in w0]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base = float(loss_fn(model))
        best = base
        for _ in range(int(steps)):
            with torch.no_grad():
                for p, w, sc, ui in zip(params, w0, scale, u):
                    p.copy_(w + sc * ui)
            loss = loss_fn(model)
            best = max(best, float(loss.detach()))
            grads = torch.autograd.grad(loss, params)
            gu = [sc * g for sc, g in zip(scale, grads)]                  # dL/du (regra da cadeia)
            gnorm = torch.sqrt(sum((g * g).sum() for g in gu)) + 1e-12
            with torch.no_grad():
                u = [ui + alpha * g / gnorm for ui, g in zip(u, gu)]
                unorm = torch.sqrt(sum((ui * ui).sum() for ui in u))
                if unorm > rho:
                    u = [ui * (rho / unorm) for ui in u]
        with torch.no_grad():
            for p, w, sc, ui in zip(params, w0, scale, u):
                p.copy_(w + sc * ui)
            best = max(best, float(loss_fn(model)))
    finally:
        with torch.no_grad():
            for p, w in zip(params, w0):
                p.copy_(w)
        model.train(was_training)
    return float(best - base)
