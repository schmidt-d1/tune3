# tune3/models/factory.py
"""
Fábrica de modelos AGNOSTICA -- permite que o Tune3 otimize hiperparametros
arquiteturais (largura, profundidade, dropout) e prova que o framework NAO e'
especifico de uma rede.

build_model(config) recebe um dicionario e devolve um nn.Module. Para o DREBIN
tabular usamos um MLP; para outros dominios, basta registrar outra fabrica aqui.
"""
from __future__ import annotations

from typing import Dict

import torch.nn as nn


def build_mlp(input_dim: int, num_classes: int, hidden_dim: int = 128,
              n_layers: int = 2, dropout: float = 0.1) -> nn.Module:
    """MLP configuravel: n_layers camadas ocultas de largura hidden_dim."""
    layers = []
    d = input_dim
    for _ in range(int(n_layers)):
        layers += [nn.Linear(d, int(hidden_dim)), nn.ReLU(), nn.Dropout(float(dropout))]
        d = int(hidden_dim)
    layers += [nn.Linear(d, int(num_classes))]
    return nn.Sequential(*layers)


def build_model(input_dim: int, num_classes: int, arch: str = "mlp",
                hparams: Dict | None = None) -> nn.Module:
    """
    Interface unica. arch in {'mlp'} por enquanto (extensivel a 'cnn', etc.).
    hparams: dict com hidden_dim, n_layers, dropout (os arquiteturais que o BO
    pode otimizar).
    """
    hp = hparams or {}
    if arch == "mlp":
        return build_mlp(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hp.get("hidden_dim", 128),
            n_layers=hp.get("n_layers", 2),
            dropout=hp.get("dropout", 0.1),
        )
    raise ValueError(f"arquitetura desconhecida: {arch}")
