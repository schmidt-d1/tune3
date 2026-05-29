# tune3/models/mlp.py
import torch
import torch.nn as nn
import structlog
from typing import List

logger = structlog.get_logger()


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.2):
        """
        Multilayer Perceptron parametrizável para Otimização de Hiperparâmetros.
        """
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Camada de saída
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

        logger.info(
            "Modelo TabularMLP inicializado",
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)