# tune3/curvature/measures.py
"""
Medidas geometricas ADICIONAIS para o E0 (Cenarios 2 e 3 do plano de robustez, out/2026), todas no
modelo final, em model.eval(), num lote fixo de treino -- a mesma convencao da Tr(H^2) primaria:

  lambda_max(model, loss_fn)      maior autovalor da Hessiana da perda (iteracao de potencia com
                                  produtos Hessiana-vetor por autograd; Rayleigh quotient no fim).
  tr_h2_hutchinson(model, loss_fn, probes, seed)
                                  SEGUNDA estimativa independente de Tr(H^2) (Hutchinson, Rademacher),
                                  com gerador proprio -- serve para medir a confiabilidade do
                                  estimador (teste-reteste entre modelos) sem tocar no RNG global.

As duas funcoes usam um torch.Generator proprio: a curvatura primaria e os CVaRs de uma rodada
antiga sao reproduzidos bit a bit quando as medidas extras sao ligadas.
"""
from __future__ import annotations

from typing import Callable

import torch


def _gen(device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _dot(a, b) -> torch.Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


@torch.enable_grad()
def lambda_max(model: torch.nn.Module, loss_fn: Callable[[torch.nn.Module], torch.Tensor],
               iters: int = 20, seed: int = 0, tol: float = 1e-3) -> float:
    """Maior autovalor (em modulo) da Hessiana da perda no lote fixo, por iteracao de potencia.
    Num minimo a Hessiana e' ~semidefinida positiva e o resultado e' lambda_max. Devolve o
    quociente de Rayleigh v^T H v da ultima iteracao (||v|| = 1)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0
    was_training = model.training
    model.eval()
    try:
        loss = loss_fn(model)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        g = _gen(params[0].device, seed)
        v = [torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype) for p in params]
        nv = torch.sqrt(_dot(v, v)) + 1e-12
        v = [vi / nv for vi in v]
        lam = 0.0
        for it in range(int(iters)):
            Hv = torch.autograd.grad(_dot(grads, v), params, retain_graph=True)
            lam_new = float(_dot(Hv, v))
            nh = torch.sqrt(_dot(Hv, Hv)) + 1e-12
            v = [(h / nh).detach() for h in Hv]
            if it > 0 and abs(lam_new - lam) <= tol * max(abs(lam_new), 1e-12):
                lam = lam_new
                break
            lam = lam_new
        return float(lam)
    finally:
        model.train(was_training)


@torch.enable_grad()
def tr_h2_hutchinson(model: torch.nn.Module, loss_fn: Callable[[torch.nn.Module], torch.Tensor],
                     probes: int = 5, seed: int = 0) -> float:
    """Tr(H^2) = E||Hv||^2 com v Rademacher, M sondas, gerador proprio (mesma formula do
    HutchinsonEstimator; independente da estimativa primaria)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0
    was_training = model.training
    model.eval()
    try:
        loss = loss_fn(model)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        g = _gen(params[0].device, seed)
        acc = 0.0
        for i in range(int(probes)):
            v = [torch.randint(0, 2, p.shape, generator=g, device=p.device, dtype=p.dtype) * 2.0 - 1.0 for p in params]
            Hv = torch.autograd.grad(_dot(grads, v), params, retain_graph=(i < probes - 1))
            acc += float(_dot(Hv, Hv))
        return acc / max(int(probes), 1)
    finally:
        model.train(was_training)


def make_extra_measures(measures, sharp_rho=(0.1, 0.5), sharp_steps: int = 10, lmax_iters: int = 20,
                        probes: int = 5, second_batch=None, seed: int = 0):
    """Monta o gancho `extra_measures(model, xb, yb, crit) -> dict` do plain_trial.
    measures: subconjunto de {"sharp", "lmax", "curv2", "curvb"}; `second_batch` = (X2, y2) tensores
    no device (para "curvb": Tr(H^2) num SEGUNDO lote de treino, ruido de lote)."""
    from tune3.curvature.sharpness import adaptive_sharpness
    measures = list(measures or [])

    def extra(model, xb, yb, crit):
        out = {}
        lf = lambda m: crit(m(xb), yb)
        if "sharp" in measures:
            for r in sharp_rho:
                out[f"sharp_{r:g}"] = float(adaptive_sharpness(model, lf, rho=r, steps=sharp_steps))
        if "lmax" in measures:
            out["lmax"] = float(lambda_max(model, lf, iters=lmax_iters, seed=seed + 1))
        if "curv2" in measures:
            out["curv2"] = float(tr_h2_hutchinson(model, lf, probes=probes, seed=seed + 2))
        if "curvb" in measures and second_batch is not None:
            X2, y2 = second_batch
            out["curvb"] = float(tr_h2_hutchinson(model, lambda m: crit(m(X2), y2), probes=probes, seed=seed + 3))
        return out
    return extra
