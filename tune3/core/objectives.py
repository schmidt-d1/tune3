# tune3/core/objectives.py
"""
Objetivo CVaR (Conditional Value-at-Risk) -- o alvo de otimizacao do Tune3.

Em SecOps, minimizar a perda MEDIA ignora a CAUDA (variantes zero-day, falsos
negativos catastroficos). O CVaR_gamma foca exatamente nessa cauda: e' a perda
ESPERADA no pior (1-gamma) fracao dos casos.

Definicoes:
  VaR_gamma(L)  = gamma-quantil de L  (limiar de cauda)
  CVaR_gamma(L) = E[L | L >= VaR_gamma]  = media da pior (1-gamma) fracao

Formula de Rockafellar-Uryasev (otimizavel, convexa em tau):
  CVaR_gamma(L) = min_tau [ tau + 1/(1-gamma) * E[(L - tau)_+] ]
  com otimo em tau* = VaR_gamma.

Ambas as formas sao implementadas e devem concordar (teste de regressao).

NOTA: gamma alto (0.95) => foco numa cauda mais estreita e mais severa.
"""
from __future__ import annotations

import numpy as np


def value_at_risk(losses, gamma: float = 0.95) -> float:
    """VaR_gamma = gamma-quantil das perdas."""
    arr = np.asarray(losses, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("losses vazio")
    return float(np.quantile(arr, gamma))


def cvar_quantile(losses, gamma: float = 0.95) -> float:
    """
    CVaR empirico pela definicao direta: media das perdas >= VaR_gamma.
    Robusto a empates incluindo o proprio VaR na cauda quando necessario.
    """
    arr = np.asarray(losses, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("losses vazio")
    var = np.quantile(arr, gamma)
    tail = arr[arr >= var]
    if tail.size == 0:           # gamma muito alto p/ amostra pequena
        tail = arr[arr >= np.max(arr)]
    return float(tail.mean())


def cvar_rockafellar(losses, gamma: float = 0.95) -> float:
    """
    CVaR via formula de Rockafellar-Uryasev, com tau = VaR (otimo analitico):
      CVaR = tau + 1/(1-gamma) * mean((L - tau)_+)
    """
    arr = np.asarray(losses, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("losses vazio")
    tau = np.quantile(arr, gamma)
    excess = np.maximum(arr - tau, 0.0)
    return float(tau + excess.mean() / (1.0 - gamma))


def cvar(losses, gamma: float = 0.95, method: str = "quantile") -> float:
    """Interface principal. method in {'quantile','rockafellar'}."""
    if method == "quantile":
        return cvar_quantile(losses, gamma)
    if method == "rockafellar":
        return cvar_rockafellar(losses, gamma)
    raise ValueError("method deve ser 'quantile' ou 'rockafellar'")
