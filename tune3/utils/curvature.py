# tune3/utils/curvature.py
import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger()


def _rademacher_like(tensor: torch.Tensor) -> torch.Tensor:
    """Gera um vetor aleatório com valores {-1, 1} com a mesma forma do tensor de entrada."""
    return torch.randint_like(tensor, low=0, high=2).float() * 2 - 1.0


def hutchinson_trace_estimator(
        model: nn.Module,
        loss: torch.Tensor,
        num_vectors: int = 5
) -> float:
    """
    Calcula a estimativa do traço da matriz Hessiana usando o método de Hutchinson.

    Args:
        model: O modelo PyTorch atual.
        loss: O valor escalar da função de custo (precisa ter `create_graph=True` no forward se necessário,
              ou calculamos os gradientes mantendo o grafo aqui).
        num_vectors: Número de amostras aleatórias (v) para a estimativa.

    Returns:
        O valor estimado do traço (float).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    if not params:
        logger.warning("Nenhum parâmetro com requires_grad encontrado. Retornando traço 0.0.")
        return 0.0

    # 1. Primeira derivada: Jacobiano (Gradiente da Loss em relação aos parâmetros)
    # create_graph=True é OBRIGATÓRIO aqui para permitir a segunda derivada
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    trace_estimate = 0.0

    for _ in range(num_vectors):
        # 2. Amostrar o vetor aleatório v (Distribuição de Rademacher)
        v = [_rademacher_like(p) for p in params]

        # 3. Produto interno do gradiente com o vetor aleatório v
        grad_v_prod = sum(torch.sum(g * v_i) for g, v_i in zip(grads, v))

        # 4. Segunda derivada: Produto Hessiana-Vetor (HVP)
        # Como estamos derivando um escalar (grad_v_prod) em relação a params, obtemos H*v
        hvp = torch.autograd.grad(grad_v_prod, params, retain_graph=True)

        # 5. Fechar a equação quadrática: v^T * H * v
        v_hvp_prod = sum(torch.sum(h_i * v_i) for h_i, v_i in zip(hvp, v))

        trace_estimate += v_hvp_prod.item()

    # Média de todas as amostras
    final_trace = trace_estimate / num_vectors

    logger.debug(
        "Traço da Hessiana estimado",
        estimativa=final_trace,
        num_vetores=num_vectors
    )

    return final_trace