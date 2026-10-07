# tune3/data/cifar.py
"""
CIFAR-10 + CIFAR-10.1 para o controle positivo da etapa D2 (out/2026).

Sem torchvision: le diretamente os arquivos oficiais (pickle do CIFAR-10, .npy do CIFAR-10.1),
para nao acrescentar dependencia nem passar por modulos que o Controle Inteligente de Aplicativos
do Windows bloqueie. Os arquivos sao baixados por `scripts/download_cifar.py`.

  data/cifar10/cifar-10-batches-py/{data_batch_1..5, test_batch}   (CIFAR-10, Krizhevsky 2009)
  data/cifar10_1/cifar10.1_v6_data.npy, cifar10.1_v6_labels.npy    (CIFAR-10.1 v6, Recht et al. 2018)

Divisoes:
  - treino: `n_train` imagens sorteadas (estratificadas por classe) do treino oficial de 50 000;
  - validacao: `n_val` imagens das RESTANTES (i.i.d. com o treino);
  - teste: as 10 000 do teste oficial do CIFAR-10 SEGUIDAS das 2 000 do CIFAR-10.1;
    `test_novel_mask` marca as do CIFAR-10.1 -- o alvo deslocado (mesmo protocolo de coleta,
    imagens novas; os modelos erram mais nelas).
Normalizacao por canal com media e desvio do TREINO SORTEADO apenas (nada de validacao ou teste).
Saida em float32, formato NCHW.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class CIFARConfig:
    data_dir: str = "data/cifar10"
    cifar101_dir: str = "data/cifar10_1"
    n_train: int = 10000
    n_val: int = 5000
    random_state: int = 0


def _read_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)   # planos R, G, B de 1024
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10_raw(data_dir):
    base = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"{base} nao encontrado -- rode scripts/download_cifar.py")
    tr = [_read_batch(os.path.join(base, f"data_batch_{i}")) for i in range(1, 6)]
    Xtr = np.concatenate([a for a, _ in tr]); ytr = np.concatenate([b for _, b in tr])
    Xte, yte = _read_batch(os.path.join(base, "test_batch"))
    return Xtr, ytr, Xte, yte


def load_cifar101_raw(cifar101_dir):
    X = np.load(os.path.join(cifar101_dir, "cifar10.1_v6_data.npy"))      # (2000, 32, 32, 3) uint8
    y = np.load(os.path.join(cifar101_dir, "cifar10.1_v6_labels.npy")).astype(np.int64)
    if X.ndim != 4 or X.shape[1:] != (32, 32, 3) or len(X) != len(y):
        raise ValueError(f"CIFAR-10.1 com formato inesperado: {X.shape}, {y.shape}")
    return np.ascontiguousarray(X.transpose(0, 3, 1, 2)), y                  # -> NCHW


def _stratified(y, n, rng):
    """Indices de `n` amostras com a mesma proporcao de classes de y."""
    classes = np.unique(y); per = n // len(classes); out = []
    for c in classes:
        out.append(rng.permutation(np.flatnonzero(y == c))[:per])
    idx = np.concatenate(out)
    if len(idx) < n:                       # resto, se n nao for multiplo do numero de classes
        rest = np.setdiff1d(np.arange(len(y)), idx)
        idx = np.concatenate([idx, rng.permutation(rest)[: n - len(idx)]])
    return np.sort(idx)


class CIFARLoader:
    def __init__(self, config: CIFARConfig | None = None):
        self.cfg = config or CIFARConfig()

    def load_splits_with_meta(self):
        c = self.cfg; rng = np.random.default_rng(c.random_state)
        Xall, yall, Xte, yte = load_cifar10_raw(c.data_dir)
        if c.n_train + c.n_val > len(yall):
            raise ValueError("n_train + n_val maior que o treino oficial (50 000)")
        itr = _stratified(yall, c.n_train, rng)
        rest = np.setdiff1d(np.arange(len(yall)), itr)
        iv = rest[_stratified(yall[rest], c.n_val, rng)]
        X1, y1 = load_cifar101_raw(c.cifar101_dir)

        Xtr = Xall[itr].astype(np.float32) / 255.0
        mean = Xtr.mean(axis=(0, 2, 3), keepdims=True); std = Xtr.std(axis=(0, 2, 3), keepdims=True)
        norm = lambda X: ((X.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)
        X_test = np.concatenate([Xte, X1]); y_test = np.concatenate([yte, y1])
        novel = np.concatenate([np.zeros(len(yte), bool), np.ones(len(y1), bool)])
        return {"X_train": norm(Xall[itr]), "y_train": yall[itr],
                "X_val": norm(Xall[iv]), "y_val": yall[iv],
                "X_test": norm(X_test), "y_test": y_test, "test_novel_mask": novel,
                "train_idx": itr, "val_idx": iv,
                "channel_mean": mean.ravel().tolist(), "channel_std": std.ravel().tolist()}

    def load_splits(self):
        m = self.load_splits_with_meta()
        return m["X_train"], m["X_val"], m["X_test"], m["y_train"], m["y_val"], m["y_test"]
