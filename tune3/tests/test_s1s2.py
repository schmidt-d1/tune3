# tune3/tests/test_s1s2.py
"""Testes-sentinela das Fases S1 (desbalanceamento) e S2 (deslocamento)."""
import numpy as np
import pytest


def _synthetic(n, frac=0.37, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 215)) < 0.15).astype(np.float32)
    w = np.zeros(215); w[:20] = 1.0
    s = X @ w + rng.normal(0, 1.5, n)
    y = (s > np.quantile(s, 1 - frac)).astype(int)
    return X, y


# ----------------------------- S1 ----------------------------- #
def test_imbalance_reduz_para_alvo():
    from tune3.data.imbalance import make_imbalanced
    X, y = _synthetic(3000)
    for target in [0.05, 0.02, 0.01]:
        Xi, yi = make_imbalanced(X, y, target, rng=np.random.default_rng(1))
        assert abs(yi.mean() - target) < 0.01


def test_imbalance_preserva_todos_negativos_ao_reduzir():
    from tune3.data.imbalance import make_imbalanced
    X, y = _synthetic(3000)
    n_neg = int((y == 0).sum())
    Xi, yi = make_imbalanced(X, y, 0.05, rng=np.random.default_rng(1))
    assert int((yi == 0).sum()) == n_neg  # negativos preservados


def test_imbalance_valida_fracao():
    from tune3.data.imbalance import make_imbalanced
    X, y = _synthetic(500)
    with pytest.raises(ValueError):
        make_imbalanced(X, y, 1.5)


# ----------------------------- S2 ----------------------------- #
def test_cluster_holdout_teste_tem_ambas_classes():
    from tune3.data.shift import cluster_holdout_split
    X, y = _synthetic(3000)
    for fold in range(5):
        Xtr, ytr, Xv, yv, Xte, yte = cluster_holdout_split(X, y, 5, fold, seed=0)
        assert len(np.unique(yte)) == 2
        assert ytr.sum() > 0 and yte.sum() > 0


def test_cluster_holdout_malware_disjunto():
    """O malware de teste vem de um cluster NAO visto no treino."""
    from tune3.data.shift import cluster_malware
    X, y = _synthetic(3000)
    clabels = cluster_malware(X, y, 5, seed=0)
    pos_idx = np.where(y == 1)[0]
    cl = clabels[pos_idx]
    for fold in range(5):
        test_mal = set(pos_idx[cl == fold])
        train_mal = set(pos_idx[(cl != fold) & (cl >= 0)])
        assert len(test_mal & train_mal) == 0


def test_cluster_holdout_cobre_todo_malware():
    """A uniao dos clusters cobre todo o malware."""
    from tune3.data.shift import n_malware_clusters
    X, y = _synthetic(3000)
    sizes = n_malware_clusters(X, y, 5, seed=0)
    assert sum(sizes.values()) == int(y.sum())


# --------------------------- metricas --------------------------- #
def test_metricas_classificador_perfeito():
    from tune3.experiments.metrics import all_metrics
    yt = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ys = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    m = all_metrics(yt, ys)
    assert m["auc_roc"] == 1.0
    assert m["auc_pr"] == 1.0
    assert m["fpr_at_95tpr"] == 0.0
    assert m["f1"] == 1.0


def test_metricas_aleatorio():
    from tune3.experiments.metrics import auc_roc
    yt = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    ys = np.array([0.5] * 8)
    assert auc_roc(yt, ys) == 0.5


def test_fpr_at_tpr_monotonia():
    """TPR alvo maior => FPR igual ou maior (mais permissivo)."""
    from tune3.experiments.metrics import fpr_at_tpr
    rng = np.random.default_rng(0)
    yt = np.array([0] * 100 + [1] * 100)
    ys = np.concatenate([rng.normal(0.3, 0.2, 100), rng.normal(0.7, 0.2, 100)])
    ys = np.clip(ys, 0, 1)
    assert fpr_at_tpr(yt, ys, 0.99) >= fpr_at_tpr(yt, ys, 0.80)
