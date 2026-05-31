# tune3/data/loader.py
import structlog
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.datasets import load_breast_cancer


logger = structlog.get_logger()


class DataLoader:
    def __init__(self, data_path: str, target_column: str, random_state: int = 42):
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state

    def load_and_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Carrega os dados tabulares e realiza o split em treino e teste.
        """
        logger.info("Carregando dataset tabular", path=self.data_path)

        try:
            df = pd.read_csv(self.data_path)

            if self.target_column not in df.columns:
                raise ValueError(f"Coluna alvo '{self.target_column}' não encontrada no dataset.")

            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            logger.info(
                "Dataset dividido com sucesso",
                train_size=len(X_train),
                test_size=len(X_test)
            )

            return X_train, X_test, y_train, y_test

        except FileNotFoundError:
            logger.error("Arquivo de dados não encontrado", path=self.data_path)
            raise
        except Exception as e:
            logger.error("Erro inesperado ao carregar dados", erro=str(e))
            raise


class BreastCancerLoader:
    def __init__(self):
        self.target_column = "target"

    def load_and_split(self, test_size=0.2):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df[self.target_column] = data.target

        # Separar features (X) e target (y)
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Split padrão
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test