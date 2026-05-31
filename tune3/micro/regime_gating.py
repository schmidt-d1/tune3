# tune3/micro/regime_gating.py
"""
RegimeGate -- gating adaptativo do DDKF por regime de informacao (manual, secao
de caracterizacao de regimes).

A razao de informacao R^2 = S_u / S_inf (exposta pelo DDKF via information_ratio())
mede quanto da variancia estacionaria do estado e' explicada pela observacao.
Classificamos a epoca em tres regimes:

  Regime A (R^2 < r2_low):   observacao POUCO informativa
                             -> predict-only (NAO aplica correcao do filtro).
                             Aplicar a correcao aqui so adicionaria ruido.
  Regime B (r2_low <= R^2 < r2_high): regime de FILTRAGEM normal
                             -> aplica a correcao do DDKF (fator 1).
  Regime C (R^2 >= r2_high):  observacao MUITO informativa
                             -> aplica correcao (alta confianca).

Tambem expomos um fator suave (ramp linear) entre r2_low e r2_high, util para
evitar chaveamento abrupto entre predict-only e filtragem.

Framing honesto (secao de regimes do artigo): em treino estabilizado, o Regime A
tende a dominar -- o DDKF atua como REDE DE SEGURANCA, nao como otimizador
principal. O gating evita que o filtro injete ruido quando nao ha sinal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Regime = Literal["A", "B", "C"]


@dataclass
class RegimeGatingConfig:
    r2_low: float = 0.10    # fronteira A|B (limiar de gating; sensibilidade F.9)
    r2_high: float = 0.50   # fronteira B|C
    smooth: bool = False    # se True, gating_factor faz ramp linear em [r2_low, r2_high]

    def __post_init__(self):
        if not (0.0 <= self.r2_low <= self.r2_high <= 1.0):
            raise ValueError("requer 0 <= r2_low <= r2_high <= 1")


class RegimeGate:
    def __init__(self, config: RegimeGatingConfig | None = None):
        self.cfg = config or RegimeGatingConfig()
        self.counts = {"A": 0, "B": 0, "C": 0}  # telemetria: fracao de epocas/regime

    def classify(self, r2: float) -> Regime:
        if r2 < self.cfg.r2_low:
            return "A"
        if r2 < self.cfg.r2_high:
            return "B"
        return "C"

    def gating_factor(self, r2: float) -> float:
        """
        Fator multiplicativo aplicado a correcao u do DDKF:
          modo duro (smooth=False): 0.0 no Regime A, 1.0 em B e C.
          modo suave (smooth=True): ramp linear de 0->1 entre r2_low e r2_high.
        """
        if not self.cfg.smooth:
            return 0.0 if r2 < self.cfg.r2_low else 1.0
        # ramp suave
        if r2 <= self.cfg.r2_low:
            return 0.0
        if r2 >= self.cfg.r2_high:
            return 1.0
        return (r2 - self.cfg.r2_low) / (self.cfg.r2_high - self.cfg.r2_low)

    def observe(self, r2: float) -> Regime:
        """Classifica e atualiza a telemetria de contagem por regime."""
        reg = self.classify(r2)
        self.counts[reg] += 1
        return reg

    def fractions(self) -> dict:
        """Fracao de epocas em cada regime (para a tabela de caracterizacao)."""
        total = sum(self.counts.values())
        if total == 0:
            return {"A": 0.0, "B": 0.0, "C": 0.0}
        return {k: v / total for k, v in self.counts.items()}
