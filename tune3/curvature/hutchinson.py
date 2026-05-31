# tune3/curvature/hutchinson.py
"""
Estimador de Hutchinson para Tr(H^2) -- o proxy de PLANURA do Tune3.

CORREÇÃO CRÍTICA vs. a versão anterior (utils/curvature.py):
    A versão antiga calculava  v^T H v   -> estima Tr(H)   (ERRADO)
    Esta versão calcula        ||Hv||^2  -> estima Tr(H^2) (CORRETO, "Patch M1")

Por quê Tr(H^2) e não Tr(H)?
    - Tr(H)  = soma dos autovalores  -> pode ser NEGATIVO perto de selas,
               mascarando a magnitude real da curvatura.
    - Tr(H^2)= soma dos autovalores ao QUADRADO -> sempre >= 0, é o
               proxy de planura sobre o qual toda a teoria do Tune3 se
               apoia (Lema 2.2, Prop. 3.5, Prop. 6.2).

Identidade-chave (sondas com E[v v^T] = I):
    E[||Hv||^2] = E[v^T H^2 v] = Tr(H^2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger()


@dataclass
class HutchinsonConfig:
    num_probes: int = 10          # M -- número de sondas (manual: M=10)
    ema_alpha: float = 0.10       # suavização EMA do traço estimado
    cooldown_steps: int = 3       # T_cd -- dispara no máximo a cada T_cd chamadas
    rel_change_trigger: float = 0.10  # delta_check -- só dispara se a loss mudou o suficiente


class HutchinsonEstimator:
    """Estimador estocástico de Tr(H^2) via ||Hv||^2 com sondas de Rademacher."""

    def __init__(self, config: Optional[HutchinsonConfig] = None):
        self.cfg = config or HutchinsonConfig()
        self.ema_trace: Optional[float] = None
        # permite disparo na primeira chamada
        self._steps_since_fire: int = self.cfg.cooldown_steps

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rademacher_like(t: torch.Tensor) -> torch.Tensor:
        """Vetor i.i.d. Rademacher em {-1,+1} com a forma/dtype/device de t."""
        return torch.randint(0, 2, t.shape, device=t.device, dtype=t.dtype) * 2.0 - 1.0

    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def estimate(self, model: nn.Module, loss: torch.Tensor) -> float:
        """
        Estima Tr(H^2) onde H = Hessiana da `loss` em relação aos parâmetros.

        IMPORTANTE:
          - `loss` deve ter sido computada com o grafo ativo (model.train()),
            sobre dados de TREINO (nunca de teste -- evita data leak).
          - Atualiza self.ema_trace internamente.

        Retorna o valor instantâneo de Tr(H^2) (não o EMA).
        """
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            logger.warning("Sem parâmetros treináveis; Tr(H^2)=0.0")
            return 0.0

        # Guard: verificar se o grafo ainda existe
        if loss.grad_fn is None:
            raise RuntimeError(
                "HutchinsonEstimator.estimate() recebeu uma loss sem grafo computacional.\n"
                "Causas comuns:\n"
                "  1. loss.backward() foi chamado antes de estimate() -> grafo liberado\n"
                "  2. loss foi computada dentro de torch.no_grad()\n"
                "Solução: chame estimate() ANTES de loss.backward(), ou use retain_graph=True\n"
                "no seu backward anterior."
            )

        # 1ª derivada com create_graph=True (necessário para 2ª derivada)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        M = self.cfg.num_probes
        acc = 0.0
        for i in range(M):
            v = [self._rademacher_like(p) for p in params]
            # produto interno grad . v  (escalar diferenciável)
            gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v))
            # HVP: d(grad.v)/d(params) = H v
            Hv = torch.autograd.grad(gv, params, retain_graph=(i < M - 1))
            # ||Hv||^2 = v^T H^2 v  (estimador de Tr(H^2)) -- ESTA é a correção
            hv_norm_sq = sum(torch.sum(h * h) for h in Hv)
            acc += float(hv_norm_sq)

        trace_h2 = acc / M

        # EMA
        if self.ema_trace is None:
            self.ema_trace = trace_h2
        else:
            a = self.cfg.ema_alpha
            self.ema_trace = a * trace_h2 + (1.0 - a) * self.ema_trace

        logger.debug("Tr(H^2) estimado", instantaneo=trace_h2, ema=self.ema_trace, M=M)
        return trace_h2

    # ------------------------------------------------------------------ #
    def maybe_estimate(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        loss_rel_change: Optional[float] = None,
    ) -> Optional[float]:
        """
        Versão com 'gating' de telemetria esparsa (cooldown + gatilho por loss).

        Retorna Tr(H^2) se disparou; None caso contrário. Mantém o custo
        de Hutchinson desprezível (~ alguns disparos por trial).
        """
        self._steps_since_fire += 1
        if self._steps_since_fire < self.cfg.cooldown_steps:
            return None
        if (loss_rel_change is not None
                and abs(loss_rel_change) < self.cfg.rel_change_trigger):
            return None
        self._steps_since_fire = 0
        return self.estimate(model, loss)
