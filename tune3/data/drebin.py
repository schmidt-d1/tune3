# tune3/data/drebin.py
"""Loader do DREBIN-215 (Android malware, 215 features binarias).
Classe 'S'->1, 'B'->0; '?'->NaN->0; split estratificado treino/val/teste.
load_splits() -> (X_train, X_val, X_test, y_train, y_val, y_test)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

logger = structlog.get_logger()


@dataclass
class DrebinConfig:
    csv_path: str = "data/drebin215.csv"
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    class_column: Optional[str] = None


class DrebinLoader:
    def __init__(self, config: Optional[DrebinConfig] = None):
        self.cfg = config or DrebinConfig()

    def _load_dataframe(self):
        df = pd.read_csv(self.cfg.csv_path)
        logger.info("DREBIN-215 carregado", path=self.cfg.csv_path, shape=df.shape)
        col = self.cfg.class_column
        if col is None:
            for cand in ["class", "Class", "label", "Label"]:
                if cand in df.columns:
                    col = cand; break
            if col is None:
                col = df.columns[-1]
        y_raw = df[col]; X = df.drop(columns=[col])
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.astype(int)
        else:
            mapping = {"S": 1, "B": 0, "s": 1, "b": 0,
                       "malware": 1, "benign": 0, "1": 1, "0": 0}
            y = y_raw.astype(str).str.strip().map(mapping)
            if y.isna().any():
                raise ValueError(f"classe nao mapeada na coluna '{col}': {y_raw.unique()[:10]}")
            y = y.astype(int)
        X = X.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
        n_missing = int(X.isna().sum().sum())
        if n_missing > 0:
            logger.warning("Valores faltantes imputados com 0", total=n_missing)
        X = X.fillna(0.0)
        logger.info("Distribuicao de classes", n=len(y), n_features=X.shape[1],
                    frac_malware=round(float(y.mean()), 4))
        return X, y

    def load_splits(self):
        X, y = self._load_dataframe()
        Xv = X.values.astype(np.float32); yv = y.values.astype(np.int64)
        X_tmp, X_te, y_tmp, y_te = train_test_split(
            Xv, yv, test_size=self.cfg.test_size, stratify=yv,
            random_state=self.cfg.random_state)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tmp, y_tmp, test_size=self.cfg.val_size, stratify=y_tmp,
            random_state=self.cfg.random_state)
        return X_tr, X_val, X_te, y_tr, y_val, y_te
