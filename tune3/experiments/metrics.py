# tune3/experiments/metrics.py
"""
Metricas operacionais de seguranca (alem do CVaR).

Venues de seguranca exigem metricas operacionais, nao so a metrica de otimizacao.
Todas recebem y_true (0/1) e y_score (probabilidade da classe positiva = malware).

- fpr_at_tpr   : falsos positivos a uma taxa de deteccao fixa (a metrica-chave de SecOps)
- auc_roc      : area sob ROC
- auc_pr       : area sob precision-recall (crucial sob desbalanceamento)
- f1_at        : F1 num limiar
- mcc_at       : Matthews num limiar
- detection_rate_at_fpr : TPR a um FPR fixo (visao complementar)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    f1_score, matthews_corrcoef,
)


def fpr_at_tpr(y_true, y_score, tpr_target: float = 0.95) -> float:
    """Menor FPR atingivel mantendo TPR >= tpr_target."""
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = tpr >= tpr_target
    if not ok.any():
        return 1.0
    return float(fpr[ok][0])


def detection_rate_at_fpr(y_true, y_score, fpr_target: float = 0.01) -> float:
    """Maior TPR atingivel mantendo FPR <= fpr_target."""
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= fpr_target
    if not ok.any():
        return 0.0
    return float(tpr[ok][-1])


def auc_roc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auc_pr(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def f1_at(y_true, y_score, threshold: float = 0.5) -> float:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def mcc_at(y_true, y_score, threshold: float = 0.5) -> float:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(matthews_corrcoef(y_true, y_pred))


def all_metrics(y_true, y_score, tpr_target: float = 0.95,
                fpr_target: float = 0.01, threshold: float = 0.5) -> Dict[str, float]:
    """Calcula todas de uma vez (para o protocolo experimental)."""
    return {
        "fpr_at_95tpr": fpr_at_tpr(y_true, y_score, tpr_target),
        "tpr_at_1fpr": detection_rate_at_fpr(y_true, y_score, fpr_target),
        "auc_roc": auc_roc(y_true, y_score),
        "auc_pr": auc_pr(y_true, y_score),
        "f1": f1_at(y_true, y_score, threshold),
        "mcc": mcc_at(y_true, y_score, threshold),
    }
