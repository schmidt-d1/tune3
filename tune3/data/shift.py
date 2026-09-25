# tune3/data/shift.py
"""
Fase S2 -- deslocamento de distribuicao / zero-day por clusterizacao (H3).

Como o DREBIN-215 nao tem rotulos de familia, simulamos "variantes nunca vistas"
clusterizando o malware no espaco de features (k-means) e fazendo leave-one-cluster-out:
o cluster retido representa uma familia/variante que NAO aparece no treino.

E' um proxy CONSERVADOR de zero-day: se o Tune3 generaliza melhor para um cluster
de malware nao visto, a vantagem deve ser ainda maior em familias reais (CICMalDroid).

cluster_holdout_split(X, y, n_clusters, fold) devolve treino/val/teste onde o
malware de teste vem inteiramente do cluster `fold`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import structlog
# k-means proprio (numpy), sem sklearn.cluster (25/09/2026): importar sklearn.cluster carrega
# sklearn.neighbors._kd_tree (extensao compilada), que o Controle Inteligente de Aplicativos do
# Windows bloqueia desde 23/09/2026. Com a implementacao propria o S2 e o E0 por cluster rodam em
# qualquer maquina, e a atribuicao de clusters e' IDENTICA entre maquinas (so' numpy + semente).
# Nao reproduz bit a bit o KMeans do sklearn -- nenhum resultado S2 anterior e' reportavel de
# qualquer forma (vazamento de teste, auditoria de set/2026).

logger = structlog.get_logger()


def cluster_malware(X: np.ndarray, y: np.ndarray, n_clusters: int,
                    seed: int = 0, positive_label: int = 1) -> np.ndarray:
    """Atribui cada amostra de malware a um cluster (k-means). Negativos = -1."""
    y = np.asarray(y).astype(int)
    labels = np.full(len(y), -1, dtype=int)
    pos_idx = np.where(y == positive_label)[0]
    labels[pos_idx] = kmeans(X[pos_idx], n_clusters, seed=seed, n_init=10)
    return labels


def kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 10, max_iter: int = 300,
           tol: float = 1e-6) -> np.ndarray:
    """k-means (Lloyd) com inicializacao k-means++; devolve o rotulo de cada linha.
    Roda `n_init` vezes e fica com a de menor inercia. Deterministico dada a semente.
    Rotulos reordenados por tamanho decrescente de cluster (0 = maior), para que o
    numero do fold nao dependa de detalhes da inicializacao."""
    X = np.asarray(X, dtype=np.float64); n = len(X)
    if not 1 <= k <= n:
        raise ValueError(f"k={k} invalido para {n} pontos")
    rng = np.random.default_rng(seed)
    sq = (X * X).sum(1)

    def d2(C):                                    # distancias quadradas ponto x centro
        return np.maximum(sq[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :], 0.0)

    best_lab, best_in = None, np.inf
    for _ in range(n_init):
        C = X[[rng.integers(n)]]                  # k-means++
        for _j in range(1, k):
            dmin = d2(C).min(1)
            p = dmin / dmin.sum() if dmin.sum() > 0 else np.full(n, 1.0 / n)
            C = np.vstack([C, X[rng.choice(n, p=p)]])
        prev = np.inf
        for _it in range(max_iter):
            D = d2(C); lab = D.argmin(1); inertia = float(D[np.arange(n), lab].sum())
            for j in range(k):
                m = lab == j
                C[j] = X[m].mean(0) if m.any() else X[rng.integers(n)]   # cluster vazio: reinicia
            if prev - inertia <= tol * max(prev, 1.0):
                break
            prev = inertia
        D = d2(C); lab = D.argmin(1); inertia = float(D[np.arange(n), lab].sum())
        if inertia < best_in:
            best_in, best_lab = inertia, lab
    order = np.argsort(-np.bincount(best_lab, minlength=k), kind="stable")
    remap = np.empty(k, dtype=int); remap[order] = np.arange(k)
    return remap[best_lab]


def cluster_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    fold: int,
    val_frac: float = 0.2,
    seed: int = 0,
    positive_label: int = 1,
) -> Tuple[np.ndarray, ...]:
    """
    Leave-one-cluster-out. Retorna (X_tr, y_tr, X_val, y_val, X_te, y_te).

    - Malware do cluster `fold` -> TESTE (variante nao vista).
    - Malware dos demais clusters -> treino/val.
    - Benignos -> divididos aleatoriamente entre treino/val/teste nas mesmas
      proporcoes (para o teste ter ambas as classes).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    clabels = cluster_malware(X, y, n_clusters, seed=seed, positive_label=positive_label)

    mal_test_idx = np.where(clabels == fold)[0]
    mal_trainval_idx = np.where((clabels != fold) & (clabels >= 0))[0]
    neg_idx = np.where(y != positive_label)[0]

    # benignos: dividir proporcionalmente. Teste recebe benignos na mesma proporcao
    # que o malware de teste representa do total de malware.
    n_mal_total = (clabels >= 0).sum()
    test_share = len(mal_test_idx) / max(1, n_mal_total)
    rng.shuffle(neg_idx)
    n_neg_test = int(round(test_share * len(neg_idx)))
    neg_test = neg_idx[:n_neg_test]
    neg_trainval = neg_idx[n_neg_test:]

    # split treino/val dentro de trainval (malware dos outros clusters + benignos)
    def split_trainval(idx):
        idx = idx.copy(); rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        return idx[n_val:], idx[:n_val]

    mal_tr, mal_val = split_trainval(mal_trainval_idx)
    neg_tr, neg_val = split_trainval(neg_trainval)

    tr = np.concatenate([mal_tr, neg_tr]); rng.shuffle(tr)
    val = np.concatenate([mal_val, neg_val]); rng.shuffle(val)
    te = np.concatenate([mal_test_idx, neg_test]); rng.shuffle(te)

    logger.info("cluster_holdout_split", fold=fold, n_clusters=n_clusters,
                treino=len(tr), val=len(val), teste=len(te),
                malware_teste=len(mal_test_idx),
                frac_mal_treino=round(float((y[tr] == positive_label).mean()), 3))
    return X[tr], y[tr], X[val], y[val], X[te], y[te]


def n_malware_clusters(X, y, n_clusters, seed=0, positive_label=1) -> dict:
    """Diagnostico: tamanho de cada cluster de malware."""
    clabels = cluster_malware(X, y, n_clusters, seed=seed, positive_label=positive_label)
    return {int(c): int((clabels == c).sum()) for c in range(n_clusters)}
