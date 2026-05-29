# tune3/tests/test_curvature.py
import torch
import torch.nn as nn
from tune3.utils.curvature import hutchinson_trace_estimator


class ModeloQuadraticoExato(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 parâmetros inicializados a 1
        self.w = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # A nossa "Loss" será w_1^2 + w_2^2 + w_3^2
        # A primeira derivada (gradiente) é 2*w
        # A segunda derivada (Hessiana) é uma matriz diagonal [2, 2, 2]
        # Portanto, o Traço da Hessiana tem de ser EXATAMENTE 6.
        return torch.sum(self.w ** 2)


def test_hutchinson_trace_exatidao():
    """
    Verifica se o estimador de Hutchinson calcula o traço exato
    para uma função com matriz Hessiana estritamente diagonal.
    """
    model = ModeloQuadraticoExato()

    # O input não importa para esta função de custo artificial
    loss = model(None)

    # Como usamos vetores de Rademacher, 1 único vetor é suficiente
    # para obter a resposta exata numa Hessiana diagonal.
    trace = hutchinson_trace_estimator(model, loss, num_vectors=1)

    # Tolerância de ponto flutuante
    assert abs(trace - 6.0) < 1e-5, f"O traço estimado foi {trace}, mas esperava-se 6.0"