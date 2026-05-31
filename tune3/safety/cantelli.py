# tune3/safety/cantelli.py
"""
CantelliGuard -- aborto robusto de trial (manual, secao 2.4).

Limiar:
    Theta_t = loc_hat_t + C_gamma * scale_hat_t
    C_gamma     = sqrt(gamma / (1 - gamma))              # constante de Cantelli
    scale_hat   = max(fisher_const * MAD, floor)         # escala robusta com piso
    fisher_const = 1.4826 = 1 / Phi^{-1}(0.75)

Cantelli (unilateral): P[X - mu >= k*sigma] <= 1/(1+k^2).
Igualando 1/(1+k^2) = 1-gamma => k = sqrt(gamma/(1-gamma)).
gamma=0.95 => C_gamma = sqrt(19) ~= 4.3589.

LOCALIZACAO ROBUSTA (mediana, nao media):
    A media e' sensivel a outliers -- um unico spike historico a inflaria,
    derrotando o proposito de robustez. Usamos a MEDIANA como estimador de
    localizacao, par classico com o MAD. Sob flutuacoes ~simetricas da perda,
    mediana ~ media, recuperando a interpretacao de Cantelli em (mu, sigma).
    (Configuravel via location='mean' caso um revisor exija a notacao literal.)

PISO DE ESCALA (floor):
    Janelas degeneradas (>50% de valores identicos) dao MAD=0, colapsando o
    limiar. O piso scale = max(1.4826*MAD, floor_rel*|loc|, floor_abs) evita isso.

Escopo (secao 2.4 do artigo): MAD*1.4826 ~ sigma so sob normalidade aproximada.
Sob caudas pesadas, prioriza deteccao de outliers vs cobertura formal exata.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CantelliConfig:
    gamma: float = 0.95
    window_size: int = 50
    fisher_const: float = 1.4826
    location: str = "median"        # "median" (robusto) ou "mean" (literal)
    floor_rel: float = 0.05         # piso de escala relativo a |loc|
    floor_abs: float = 1e-6         # piso de escala absoluto
    min_samples: Optional[int] = None

    def __post_init__(self):
        if not (0.0 < self.gamma < 1.0):
            raise ValueError(f"gamma deve estar em (0,1), recebido {self.gamma}")
        if self.location not in ("median", "mean"):
            raise ValueError("location deve ser 'median' ou 'mean'")
        if self.min_samples is None:
            self.min_samples = self.window_size


class CantelliGuard:
    """Guarda de aborto robusto via Cantelli + MAD."""

    def __init__(self, config: Optional[CantelliConfig] = None):
        self.cfg = config or CantelliConfig()
        self.loss_buffer: deque = deque(maxlen=self.cfg.window_size)
        self.C_gamma: float = math.sqrt(self.cfg.gamma / (1.0 - self.cfg.gamma))
        logger.info("CantelliGuard inicializado",
                    gamma=self.cfg.gamma, C_gamma=round(self.C_gamma, 4),
                    window=self.cfg.window_size, location=self.cfg.location)

    # ------------------------------------------------------------------ #
    def _location(self, arr: np.ndarray) -> float:
        if self.cfg.location == "median":
            return float(np.median(arr))
        return float(np.mean(arr))

    def _scale(self, arr: np.ndarray, loc: float) -> float:
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        scale = self.cfg.fisher_const * float(mad)
        # piso para evitar colapso (MAD=0) em janelas degeneradas
        return max(scale, self.cfg.floor_rel * abs(loc), self.cfg.floor_abs)

    # ------------------------------------------------------------------ #
    def threshold(self) -> Optional[float]:
        if len(self.loss_buffer) < self.cfg.min_samples:
            return None
        arr = np.asarray(self.loss_buffer, dtype=float)
        loc = self._location(arr)
        scale = self._scale(arr, loc)
        return loc + self.C_gamma * scale

    def should_abort(self, loss: float) -> bool:
        """
        Testa a perda atual contra o limiar derivado do HISTORICO e entao a
        incorpora. Durante warmup, nunca aborta. O valor atual e' testado ANTES
        de entrar na janela (um outlier nao infla o proprio limiar).
        """
        loss = float(loss)
        if len(self.loss_buffer) < self.cfg.min_samples:
            self.loss_buffer.append(loss)
            return False
        thr = self.threshold()
        abort = loss > thr
        self.loss_buffer.append(loss)
        if abort:
            logger.warning("Cantelli: aborto disparado",
                           loss=round(loss, 4), threshold=round(thr, 4))
        return abort

    def reset(self) -> None:
        self.loss_buffer.clear()
