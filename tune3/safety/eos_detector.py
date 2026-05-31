# tune3/safety/eos_detector.py
"""
EoSDetector -- detector de Edge-of-Stability via "progressive sharpening"
(Cohen et al. 2021; manual, conexao com edge of stability).

Fenomeno: durante o treino, a curvatura (aqui o proxy Tr(H^2)) frequentemente
CRESCE monotonicamente ("progressive sharpening") ate o treino atingir o limite
de estabilidade, onde o maior autovalor da Hessiana ~ 2/eta. Passar desse limite
causa oscilacao/divergencia da loss.

Estrategia: monitorar a serie de Tr(H^2). Se houver crescimento sustentado
(tendencia monotonica de alta por >= patience passos E razao recente/baseline
acima de growth_threshold), sinalizar EoS e sugerir reducao do lr por lr_factor.

Diferente do CantelliGuard (que ABORTA em spikes catastroficos da LOSS), o
EoSDetector age preventivamente na CURVATURA, reduzindo o lr ANTES da divergencia.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class EoSConfig:
    window: int = 6              # janela da serie de curvatura
    patience: int = 4            # nº de passos de alta consecutiva p/ disparar
    growth_threshold: float = 1.5  # razao recente/baseline que caracteriza sharpening
    lr_factor: float = 0.7       # fator de reducao do lr ao detectar EoS
    cooldown: int = 5            # passos de espera apos um disparo (evita disparos repetidos)
    eps: float = 1e-12

    def __post_init__(self):
        if self.window < 2:
            raise ValueError("window deve ser >= 2")
        if not (0.0 < self.lr_factor < 1.0):
            raise ValueError("lr_factor deve estar em (0,1)")


class EoSDetector:
    def __init__(self, config: Optional[EoSConfig] = None):
        self.cfg = config or EoSConfig()
        self.buffer: deque = deque(maxlen=self.cfg.window)
        self._cooldown_left: int = 0
        self.n_triggers: int = 0

    def _monotonic_rise_len(self) -> int:
        """Comprimento da sequencia final estritamente crescente."""
        arr = list(self.buffer)
        run = 1
        for i in range(len(arr) - 1, 0, -1):
            if arr[i] > arr[i - 1]:
                run += 1
            else:
                break
        return run

    def update(self, curvature: float) -> bool:
        """
        Registra um novo valor de Tr(H^2) e retorna True se EoS for detectado.
        Em caso positivo, o chamador deve multiplicar o lr por suggested_lr_factor().
        """
        self.buffer.append(float(curvature))

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return False
        if len(self.buffer) < self.cfg.window:
            return False

        arr = np.asarray(self.buffer, dtype=float)
        baseline = float(arr[0]) + self.cfg.eps
        recent = float(arr[-1])
        growth = recent / baseline
        rise = self._monotonic_rise_len()

        triggered = (rise >= self.cfg.patience) and (growth >= self.cfg.growth_threshold)
        if triggered:
            self.n_triggers += 1
            self._cooldown_left = self.cfg.cooldown
            logger.warning("EoS: progressive sharpening detectado",
                           growth=round(growth, 3), rise=rise,
                           lr_factor=self.cfg.lr_factor)
        return triggered

    def suggested_lr_factor(self) -> float:
        return self.cfg.lr_factor

    def reset(self) -> None:
        self.buffer.clear()
        self._cooldown_left = 0
