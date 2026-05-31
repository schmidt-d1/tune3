# tune3/curvature/gsnr.py
"""
GSNREstimator -- Gradient Signal-to-Noise Ratio (terceira observacao do DDKF).

Definicao (Liu et al. 2020): para um parametro, GSNR = (E[g])^2 / Var[g], onde g
e' o gradiente por minibatch. Agregamos sobre os parametros:

    GSNR = ||g_bar||^2 / sum_i Var[g_i]

onde g_bar e' a media dos gradientes sobre K minibatches e Var[g_i] a variancia
empirica componente a componente.

Interpretacao: GSNR alto => sinal de gradiente consistente entre minibatches
(direcao confiavel) => pode-se usar lr maior. GSNR baixo => gradiente dominado
por ruido de amostragem => lr menor. O DDKF observa log(GSNR) como sinal de
controle do learning-rate.

Este modulo e' AGNOSTICO de arquitetura: recebe uma funcao que produz o gradiente
achatado de um minibatch, entao funciona para o TabularMLP do DREBIN ou qualquer
outro modelo PyTorch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class GSNRConfig:
    num_batches: int = 8     # K -- nº de minibatches para estimar media/variancia
    eps: float = 1e-12       # estabilidade numerica no denominador
    log_floor: float = 1e-8  # piso para log(GSNR)


class GSNREstimator:
    """Estima GSNR a partir de gradientes por minibatch (agnostico de framework)."""

    def __init__(self, config: Optional[GSNRConfig] = None):
        self.cfg = config or GSNRConfig()

    # ------------------------------------------------------------------ #
    def from_gradients(self, grads: List[np.ndarray]) -> float:
        """
        Calcula GSNR a partir de uma lista de K gradientes ACHATADOS (1-D),
        um por minibatch. Cada grad e' um vetor de dimensao D (mesma para todos).

        GSNR = ||mean||^2 / sum(var_componente)
        """
        if len(grads) < 2:
            raise ValueError("GSNR requer >= 2 minibatches")
        G = np.stack([np.asarray(g, dtype=float).reshape(-1) for g in grads], axis=0)  # (K, D)
        g_bar = G.mean(axis=0)                 # (D,)
        # variancia empirica por componente (ddof=1)
        var = G.var(axis=0, ddof=1)            # (D,)
        signal = float(g_bar @ g_bar)          # ||mean||^2
        noise = float(var.sum()) + self.cfg.eps
        return signal / noise

    def log_gsnr(self, grads: List[np.ndarray]) -> float:
        """log(GSNR) com piso, para uso direto como observacao do DDKF."""
        gsnr = self.from_gradients(grads)
        return math.log(max(gsnr, self.cfg.log_floor))

    # ------------------------------------------------------------------ #
    def estimate_torch(self, grad_fn: Callable[[], "object"]) -> float:
        """
        Versao PyTorch: grad_fn() deve retornar o gradiente achatado (torch.Tensor 1-D)
        de UM minibatch (tipicamente: zero_grad -> forward -> backward -> concat grads).
        Chamamos grad_fn() K vezes (minibatches diferentes) e computamos o GSNR.

        Mantido aqui apenas como conveniencia; a logica numerica esta em from_gradients.
        """
        grads = []
        for _ in range(self.cfg.num_batches):
            g = grad_fn()
            # aceita torch.Tensor ou np.ndarray
            arr = g.detach().cpu().numpy() if hasattr(g, "detach") else np.asarray(g)
            grads.append(arr.reshape(-1))
        return self.from_gradients(grads)
