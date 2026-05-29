# tune3/data/validation.py
import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger()


def validate_dataset(df: pd.DataFrame, target_column: str) -> bool:
    """
    Verifica a sanidade do dataset (NaNs e Infinitos) para garantir
    estabilidade nos cálculos de gradiente e Hessiana.
    """
    logger.info("Validando integridade do dataset", shape=df.shape)

    if target_column not in df.columns:
        logger.error("Falha: Coluna alvo não encontrada.", target=target_column)
        raise ValueError(f"Coluna alvo '{target_column}' ausente do dataset.")

    # Busca por valores nulos (NaN)
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.warning("Valores nulos detectados no dataset", total_nans=nan_count)
        raise ValueError(f"O dataset contém {nan_count} valores NaN. Limpeza necessária.")

    # Busca por infinitos, que quebram o estimador Hutchinson
    features = df.drop(columns=[target_column])
    inf_count = np.isinf(features.select_dtypes(include=[np.number]).values).sum()
    if inf_count > 0:
        logger.error("Valores infinitos (inf) detectados no dataset.")
        raise ValueError("Valores numéricos infinitos inviabilizam o treinamento.")

    logger.info("Validação concluída: Dataset estruturalmente apto.")
    return True