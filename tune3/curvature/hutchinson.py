# tune3/curvature/hutchinson.py
"""Estimador de Tr(H^2) via Hutchinson com vetores de Rademacher.
Tr(H^2) = E[||Hv||^2] (sensor de explosao de curvatura). Inclui EMA, cooldown
e gatilho por mudanca relativa de loss para acionamento esparso (maybe_estimate)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import structlog
logger = structlog.get_logger()


@dataclass
class HutchinsonConfig:
    num_probes: int = 10
    ema_alpha: float = 0.10
    cooldown_steps: int = 3
    rel_change_trigger: float = 0.10


class HutchinsonEstimator:
    def __init__(self, config: Optional[HutchinsonConfig] = None):
        self.cfg = config or HutchinsonConfig()
        self.ema_trace: Optional[float] = None
        self._steps_since_fire = self.cfg.cooldown_steps

    @staticmethod
    def _rademacher_like(t):
        return torch.randint(0, 2, t.shape, device=t.device, dtype=t.dtype) * 2.0 - 1.0

    @torch.enable_grad()
    def estimate(self, model, loss) -> float:
        """Tr(H^2) = (1/M) sum_i ||H v_i||^2, v_i ~ Rademacher."""
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return 0.0
        if loss.grad_fn is None:
            raise RuntimeError("loss sem grafo (precisa de requires_grad)")
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        M = self.cfg.num_probes
        acc = 0.0
        for i in range(M):
            v = [self._rademacher_like(p) for p in params]
            gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v))
            Hv = torch.autograd.grad(gv, params, retain_graph=(i < M - 1))
            acc += float(sum(torch.sum(h * h) for h in Hv))
        tr = acc / M
        self.ema_trace = tr if self.ema_trace is None else \
            self.cfg.ema_alpha * tr + (1 - self.cfg.ema_alpha) * self.ema_trace
        return tr

    def maybe_estimate(self, model, loss, loss_rel_change=None):
        """Aciona o estimador apenas apos cooldown e/ou mudanca relevante de loss."""
        self._steps_since_fire += 1
        if self._steps_since_fire < self.cfg.cooldown_steps:
            return None
        if loss_rel_change is not None and abs(loss_rel_change) < self.cfg.rel_change_trigger:
            return None
        self._steps_since_fire = 0
        return self.estimate(model, loss)
