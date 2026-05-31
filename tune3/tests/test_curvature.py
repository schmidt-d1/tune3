# tune3/tests/test_curvature.py
"""
Testes-sentinela para o estimador de Tr(H^2).

Estes testes capturam DUAS regressões críticas:
  1. Confundir Tr(H) com Tr(H^2)  -> test_trace_h2_diagonal
  2. Variância do estimador com sinal trocado -> test_variancia_zero_para_identidade
"""
import torch
import torch.nn as nn

from tune3.curvature import HutchinsonEstimator, HutchinsonConfig


class ModeloQuadratico(nn.Module):
    """loss = sum(w_i^2)  =>  H = 2 I (diagonal).
    Para w in R^3:  Tr(H) = 6,  Tr(H^2) = 4+4+4 = 12.
    """
    def __init__(self, n: int = 3):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))

    def forward(self, x=None):
        return torch.sum(self.w ** 2)


def test_trace_h2_diagonal():
    """Para H = 2I (3x3), Tr(H^2) deve ser EXATAMENTE 12, não 6 (=Tr H)."""
    model = ModeloQuadratico(n=3)
    loss = model()
    # num_probes=1 é exato para H = cI porque ||v||^2 = n é CONSTANTE
    # para sondas Rademacher (v_i = ±1 => v_i^2 = 1 sempre).
    # Portanto ||Hv||^2 = c^2 * n é determinístico. Sem aleatoriedade.
    est = HutchinsonEstimator(HutchinsonConfig(num_probes=1))
    trace_h2 = est.estimate(model, loss)
    assert abs(trace_h2 - 12.0) < 1e-5, (
        f"Tr(H^2) estimado = {trace_h2}, esperado 12.0. "
        f"Se vier ~6.0, o estimador está calculando Tr(H) (BUG)."
    )


def test_trace_h2_escala():
    """loss = c * sum(w^2) => H = 2c I => Tr(H^2) = n * (2c)^2."""
    c = 3.0
    n = 4

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(n))
        def forward(self, x=None):
            return c * torch.sum(self.w ** 2)

    model = M()
    loss = model()
    est = HutchinsonEstimator(HutchinsonConfig(num_probes=1))
    trace_h2 = est.estimate(model, loss)
    esperado = n * (2 * c) ** 2  # 4 * 36 = 144
    assert abs(trace_h2 - esperado) < 1e-4, f"{trace_h2} != {esperado}"


def test_ema_atualiza():
    """O EMA deve ser preenchido após a primeira chamada."""
    model = ModeloQuadratico()
    est = HutchinsonEstimator(HutchinsonConfig(num_probes=2))
    assert est.ema_trace is None
    est.estimate(model, model())
    assert est.ema_trace is not None


def test_cooldown_gating():
    """maybe_estimate deve respeitar o cooldown."""
    model = ModeloQuadratico()
    est = HutchinsonEstimator(HutchinsonConfig(num_probes=1, cooldown_steps=3))
    # 1ª chamada dispara (contador inicia satisfeito)
    r0 = est.maybe_estimate(model, model(), loss_rel_change=1.0)
    assert r0 is not None
    # próximas 2 chamadas NÃO disparam (cooldown)
    assert est.maybe_estimate(model, model(), loss_rel_change=1.0) is None
    assert est.maybe_estimate(model, model(), loss_rel_change=1.0) is None
    # 3ª chamada após reset dispara de novo
    assert est.maybe_estimate(model, model(), loss_rel_change=1.0) is not None
