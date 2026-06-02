# tune3/data/imbalance.py
"""
Fase S1 -- desbalanceamento controlado (H2).

Subamostra a classe de malware do TREINO para uma razao-alvo, mantendo validacao
e teste numa razao fixa realista. Testa a hipotese de que a vantagem do Tune3 em
CVaR cresce conforme o malware fica mais raro (a cauda passa a dominar).

So mexe no TREINO. Validacao e teste preservam sua composicao original (ou uma
razao fixa) para que a avaliacao seja comparavel entre razoes de treino.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger()


def make_imbalanced(
    X: np.ndarray,
    y: np.ndarray,
    target_malware_frac: float,
    rng: Optional[np.random.Generator] = None,
    positive_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subamostra para atingir `target_malware_frac` de positivos.

    Mantem TODA a classe majoritaria e remove amostras da minoritaria necessaria
    para chegar a razao desejada. Se a razao-alvo for MAIOR que a atual, subamostra
    a classe negativa em vez disso.
    """
    if not 0.0 < target_malware_frac < 1.0:
        raise ValueError("target_malware_frac deve estar em (0,1)")
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y).astype(int)
    pos_idx = np.where(y == positive_label)[0]
    neg_idx = np.where(y != positive_label)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    cur_frac = n_pos / (n_pos + n_neg)

    if abs(cur_frac - target_malware_frac) < 1e-9:
        return X, y

    if target_malware_frac < cur_frac:
        # reduzir positivos: manter todos negativos, subamostrar positivos
        # f = p/(p+neg) => p = f*neg/(1-f)
        n_keep_pos = int(round(target_malware_frac * n_neg / (1 - target_malware_frac)))
        n_keep_pos = max(1, min(n_keep_pos, n_pos))
        keep_pos = rng.choice(pos_idx, size=n_keep_pos, replace=False)
        keep = np.concatenate([keep_pos, neg_idx])
    else:
        # aumentar fracao de positivos: subamostrar negativos
        # f = pos/(pos+n) => n = pos*(1-f)/f
        n_keep_neg = int(round(n_pos * (1 - target_malware_frac) / target_malware_frac))
        n_keep_neg = max(1, min(n_keep_neg, n_neg))
        keep_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
        keep = np.concatenate([pos_idx, keep_neg])

    rng.shuffle(keep)
    X_out, y_out = X[keep], y[keep]
    logger.info("desbalanceamento aplicado",
                alvo=round(target_malware_frac, 4),
                obtido=round(float((y_out == positive_label).mean()), 4),
                n=len(y_out))
    return X_out, y_out
