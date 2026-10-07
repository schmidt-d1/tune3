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


class SmallCNN(nn.Module):
    """CNN pequena para o controle positivo no CIFAR-10 (etapa D2, out/2026).

    `depth` blocos; o bloco i tem duas conv 3x3 com `width * 2**i` canais, ReLU e max-pool 2x2.
    Depois: achatamento, dropout e uma camada linear. Sem batch-norm, de proposito: a BN torna a
    rede invariante a escala dos pesos anteriores e mistura essa invariancia com a pergunta sobre
    curvatura. Profundidade e largura variam de forma CONTROLADA na grade da D2, porque nos tres
    cenarios da D1 a curvatura acompanhou sobretudo a profundidade."""

    def __init__(self, num_classes: int, width: int = 32, depth: int = 3, dropout: float = 0.0,
                 in_channels: int = 3, image_size: int = 32):
        super().__init__()
        layers, c_in, size = [], int(in_channels), int(image_size)
        for i in range(int(depth)):
            c = int(width) * 2 ** i
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(),
                       nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c_in, size = c, size // 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(float(dropout)),
                                  nn.Linear(c_in * size * size, int(num_classes)))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(input_dim: int, num_classes: int, arch: str = "mlp",
                hparams: Dict | None = None) -> nn.Module:
    """
    Interface unica. arch in {'mlp', 'cnn'}.
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
    if arch == "cnn":      # input_dim = numero de canais; hidden_dim = largura; n_layers = blocos
        return SmallCNN(num_classes=num_classes, width=hp.get("hidden_dim", 32),
                        depth=hp.get("n_layers", 3), dropout=hp.get("dropout", 0.0),
                        in_channels=input_dim)
    raise ValueError(f"arquitetura desconhecida: {arch}")
