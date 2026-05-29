# tune3/data/preprocessing.py
import structlog
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold

logger = structlog.get_logger()


class DataPreprocessor:
    def __init__(self, scaling_method: str = "robust", variance_threshold: float = 0.0):
        self.scaling_method = scaling_method
        self.variance_threshold = variance_threshold

        if scaling_method == "robust":
            self.scaler = RobustScaler()
        elif scaling_method == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Método de escalonamento '{scaling_method}' não suportado.")

        self.var_filter = VarianceThreshold(threshold=variance_threshold)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando pré-processamento", shape=X_train.shape, scaler=self.scaling_method)

        # 1. Filtra colunas sem variância (evita matrizes singulares)
        X_train_var = self.var_filter.fit_transform(X_train)
        kept_features = self.var_filter.get_feature_names_out(X_train.columns)

        # 2. Aplica escalonamento
        X_train_scaled = self.scaler.fit_transform(X_train_var)

        features_removidas = X_train.shape[1] - len(kept_features)
        if features_removidas > 0:
            logger.info("Features de variância zero removidas", count=features_removidas)

        return pd.DataFrame(X_train_scaled, columns=kept_features, index=X_train.index)

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        # Aplica exatamente a mesma transformação dos dados de treino ao teste
        X_test_var = self.var_filter.transform(X_test)
        kept_features = self.var_filter.get_feature_names_out(X_test.columns)
        X_test_scaled = self.scaler.transform(X_test_var)

        return pd.DataFrame(X_test_scaled, columns=kept_features, index=X_test.index)