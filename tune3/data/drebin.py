# tune3/data/drebin.py
"""
Loader do dataset DREBIN-215 (Android malware).

DREBIN-215:
    15.036 aplicações (5.560 malware + 9.476 benignas), cada uma representada
    por um vetor de 215 features binárias (permissões, chamadas de API, intents,
    comandos do sistema) extraídas por análise estática.

NOTA sobre a escala "129.013":
    O artigo menciona ~129.013 aplicações -- esse é o DREBIN ORIGINAL completo
    (5.560 malware + 123.453 benignas) com ~545.000 features esparsas, que exige
    pedido de acesso aos autores (uni-goettingen). Para reprodutibilidade e
    fidelidade ao modelo de features de 215 dimensões do artigo, usamos o
    DREBIN-215 (CSV público). Recomendo declarar isto explicitamente na seção
    de dados do artigo.

COMO OBTER O CSV (escolha UMA opção):
    (a) figshare (recomendado, fonte do paper DroidFusion):
        https://figshare.com/articles/dataset/Android_malware_dataset_for_machine_learning_2/5854653
        Baixe o arquivo e salve como: data/drebin215.csv
    (b) Kaggle: procure "Drebin-215 dataset"
    (c) Mirror GitHub (CSV direto):
        github.com/.../drebin-215-dataset-5560malware-9476-benign.csv

PECULIARIDADES TRATADAS POR ESTE LOADER:
    - A coluna de classe pode se chamar 'class' e conter 'S'/'B' (malware/benigno)
      OU já vir como 1/0. Tratamos ambos.
    - Algumas células contêm '?' (faltante) e fazem o pandas ler colunas como
      string. Coercemos para numérico e imputamos.
    - Convencão: y = 1 para MALWARE (classe positiva = o que queremos detectar).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

logger = structlog.get_logger()


@dataclass
class DrebinConfig:
    csv_path: str = "data/drebin215.csv"
    test_size: float = 0.20
    val_size: float = 0.20         # fração do TREINO reservada para validação
    random_state: int = 42
    class_column_candidates: tuple = ("class", "Class", "label", "Label", "target")
    malware_tokens: tuple = ("S", "s", "1", 1, "malware", "Malware")


class DrebinLoader:
    """Carrega e prepara o DREBIN-215 com split estratificado treino/val/teste."""

    def __init__(self, config: DrebinConfig | None = None):
        self.cfg = config or DrebinConfig()
        self.feature_names_: list[str] | None = None

    # ------------------------------------------------------------------ #
    def _find_class_column(self, df: pd.DataFrame) -> str:
        for cand in self.cfg.class_column_candidates:
            if cand in df.columns:
                return cand
        # fallback: última coluna
        logger.warning("Coluna de classe não encontrada por nome; usando a última coluna.",
                       ultima=df.columns[-1])
        return df.columns[-1]

    def _encode_labels(self, y_raw: pd.Series) -> np.ndarray:
        """Mapeia para {0,1} com 1 = malware."""
        # caso já seja numérico 0/1
        if pd.api.types.is_numeric_dtype(y_raw):
            uniq = set(pd.unique(y_raw.dropna()))
            if uniq <= {0, 1}:
                return y_raw.astype(int).to_numpy()
        # caso S/B (ou variantes)
        tokens = set(str(t) for t in self.cfg.malware_tokens)
        y = y_raw.astype(str).str.strip().map(lambda v: 1 if v in tokens else 0)
        return y.to_numpy().astype(int)

    # ------------------------------------------------------------------ #
    def load_raw(self) -> pd.DataFrame:
        if not os.path.exists(self.cfg.csv_path):
            raise FileNotFoundError(
                f"DREBIN-215 não encontrado em '{self.cfg.csv_path}'.\n"
                "Baixe o CSV (figshare/Kaggle/GitHub -- ver docstring de drebin.py)\n"
                "e salve nesse caminho, ou ajuste DrebinConfig.csv_path."
            )
        # low_memory=False evita inferência de dtype por chunk (importante com '?')
        df = pd.read_csv(self.cfg.csv_path, low_memory=False)
        logger.info("DREBIN-215 carregado", shape=df.shape, path=self.cfg.csv_path)
        return df

    def _clean_features(self, X: pd.DataFrame) -> np.ndarray:
        """Coerce '?' -> NaN -> 0, e garante dtype float."""
        X = X.replace("?", np.nan)
        X = X.apply(pd.to_numeric, errors="coerce")
        n_nan = int(X.isna().sum().sum())
        if n_nan > 0:
            logger.warning("Valores faltantes imputados com 0", total=n_nan)
            X = X.fillna(0.0)
        return X.to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------ #
    def load_splits(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna (X_train, X_val, X_test, y_train, y_val, y_test) já como np.float32/int.
        Splits ESTRATIFICADOS (preservam a proporção malware/benigno).
        """
        df = self.load_raw()
        class_col = self._find_class_column(df)

        y_all = self._encode_labels(df[class_col])
        X_df = df.drop(columns=[class_col])
        self.feature_names_ = list(X_df.columns)
        X_all = self._clean_features(X_df)

        frac_mal = float(y_all.mean())
        logger.info("Distribuição de classes",
                    n=len(y_all), frac_malware=round(frac_mal, 4),
                    n_features=X_all.shape[1])

        # 1º split: treino+val vs teste (estratificado)
        X_tv, X_test, y_tv, y_test = train_test_split(
            X_all, y_all,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y_all,
        )
        # 2º split: treino vs val (estratificado)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=self.cfg.val_size,
            random_state=self.cfg.random_state,
            stratify=y_tv,
        )
        logger.info("Splits criados",
                    train=len(y_train), val=len(y_val), test=len(y_test))
        return X_train, X_val, X_test, y_train, y_val, y_test
