# tune3/data/nslkdd.py
"""Loader do NSL-KDD (deteccao de intrusao em rede) -- trilha de VALIDADE EXTERNA do Tune3.

POR QUE ESTE DATASET (decisao de 22/09/2026, ver docs do projeto):
  O DREBIN-215 tem 215 atributos BINARIOS: superficie de perda excepcionalmente bem
  condicionada. A tese do Tune3 e' sobre curvatura e nunca tinha sido testada onde a
  curvatura se comporta mal. O NSL-KDD tem 37 atributos numericos em escalas muito
  diferentes (src_bytes/dst_bytes chegam a ~1,4e9) e 3 categoricos, e seu conjunto de
  teste oficial contem 17 tipos de ataque AUSENTES do treino -- um cenario zero-day
  publicado, independente da nossa construcao por cluster (H3/S2).
  Ressalva a declarar no texto: o NSL-KDD (2009, derivado do KDD'99) e' criticado como
  benchmark de IDS. Aqui ele e' ESTRESSE DO OTIMIZADOR, nao alegacao de estado da arte.

FORMATO (conferido nos arquivos reais em 23/09/2026):
  KDDTrain+.txt  125 973 linhas | KDDTest+.txt  22 544 linhas | 43 colunas, sem cabecalho
  col 0..40 : 41 atributos; categoricos nas colunas 1 (protocol_type, 3 valores),
              2 (service, 70) e 3 (flag, 11); coluna 19 (num_outbound_cmds) e' constante
  col 41    : rotulo textual ('normal' ou o nome do ataque; 23 classes no treino, 38 no teste)
  col 42    : "dificuldade" (0..21) -- METADADO do dataset, NAO e' atributo: descartada
  Binario: normal -> 0, qualquer ataque -> 1 (ataque = classe positiva, como malware no DREBIN).

SEM VAZAMENTO: categorias do one-hot e parametros de escala sao ajustados SO' no treino.

SPLITS
  split="official" (padrao): treino/validacao estratificados a partir do KDDTrain+; teste =
      KDDTest+ inteiro (16,6 % das linhas de teste sao ataques NUNCA vistos). E' o analogo
      zero-day do H3.
  split="random": treino/validacao/teste estratificados SO' dentro do KDDTrain+ -- analogo
      in-distribution do H1 (ataques do teste sao todos conhecidos).

ESCALA ("scaling")
  "standard"     (padrao) z-score por coluna, ajustado no treino. PRESERVA a cauda pesada
                  de src_bytes/dst_bytes (valores a centenas de desvios-padrao), que e'
                  justamente o estresse que queremos -- mas continua treinavel.
  "log_standard" log1p nas colunas numericas nao-negativas, depois z-score. E' a escolha
                  "bem-comportada" convencional; serve de controle (curvatura domada).
  "none"         cru. Com valores de 1e9 o SGD diverge; existe so' para demonstrar isso.

load_splits() -> (X_train, X_val, X_test, y_train, y_val, y_test), mesma interface do DREBIN.
load_splits_with_meta() -> dict com os arrays + test_novel_mask + nomes das features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

logger = structlog.get_logger()

# nomes canonicos das 41 features (ordem das colunas 0..40)
FEATURES: List[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
LABEL, DIFFICULTY = "label", "difficulty"
COLUMNS = FEATURES + [LABEL, DIFFICULTY]


@dataclass
class NSLKDDConfig:
    data_dir: str = "data/nsl_kdd"
    train_file: str = "KDDTrain+.txt"
    test_file: str = "KDDTest+.txt"
    random_state: int = 42
    val_size: float = 0.2
    test_size: float = 0.2            # so' usado em split="random"
    split: str = "official"           # "official" (zero-day) | "random" (in-distribution)
    scaling: str = "standard"         # "standard" | "log_standard" | "none"
    max_train: Optional[int] = None   # subamostra estratificada do treino (pilotos rapidos)

    def __post_init__(self):
        if self.split not in ("official", "random"):
            raise ValueError(f"split deve ser 'official' ou 'random'; veio {self.split!r}")
        if self.scaling not in ("standard", "log_standard", "none"):
            raise ValueError(f"scaling invalido: {self.scaling!r}")


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    if df.shape[1] != 43:
        raise ValueError(f"{path}: esperadas 43 colunas, encontradas {df.shape[1]}")
    for c in CATEGORICAL + [LABEL]:
        df[c] = df[c].astype(str).str.strip()
    return df


class NSLKDDLoader:
    def __init__(self, config: Optional[NSLKDDConfig] = None):
        self.cfg = config or NSLKDDConfig()

    # ------------------------------------------------------------------ preprocess
    def _fit_transform(self, tr: pd.DataFrame, others: List[pd.DataFrame]):
        """Ajusta one-hot e escala no TREINO e aplica aos demais."""
        num_cols = [c for c in FEATURES if c not in CATEGORICAL]
        const = [c for c in num_cols if tr[c].nunique() <= 1]
        num_cols = [c for c in num_cols if c not in const]
        cats = {c: sorted(tr[c].unique()) for c in CATEGORICAL}

        def encode(df):
            parts = [df[num_cols].astype(np.float64)]
            for c in CATEGORICAL:
                # categoria nunca vista no treino -> vetor de zeros (nao quebra, nao vaza)
                parts.append(pd.DataFrame(
                    {f"{c}={v}": (df[c].values == v).astype(np.float64) for v in cats[c]},
                    index=df.index))
            return pd.concat(parts, axis=1)

        Xtr = encode(tr); Xo = [encode(d) for d in others]
        s = self.cfg.scaling
        if s == "log_standard":
            nonneg = [c for c in num_cols if (tr[c] >= 0).all()]
            for X in [Xtr] + Xo:
                X[nonneg] = np.log1p(X[nonneg].clip(lower=0))
        if s in ("standard", "log_standard"):
            mu = Xtr[num_cols].mean(); sd = Xtr[num_cols].std(ddof=0).replace(0.0, 1.0)
            for X in [Xtr] + Xo:
                X[num_cols] = (X[num_cols] - mu) / sd
        names = list(Xtr.columns)
        return (Xtr.values.astype(np.float32), [X.values.astype(np.float32) for X in Xo],
                names, const)

    # ------------------------------------------------------------------ public
    def load_splits_with_meta(self) -> Dict:
        d = Path(self.cfg.data_dir)
        tr_full = _read(d / self.cfg.train_file)
        y_full = (tr_full[LABEL] != "normal").astype(np.int64).values
        rs = self.cfg.random_state

        if self.cfg.max_train is not None and self.cfg.max_train < len(tr_full):
            keep, _ = train_test_split(np.arange(len(tr_full)), train_size=self.cfg.max_train,
                                       stratify=y_full, random_state=rs)
            tr_full = tr_full.iloc[np.sort(keep)].reset_index(drop=True)
            y_full = y_full[np.sort(keep)]

        if self.cfg.split == "official":
            te = _read(d / self.cfg.test_file)
            tr_idx, va_idx = train_test_split(np.arange(len(tr_full)), test_size=self.cfg.val_size,
                                              stratify=y_full, random_state=rs)
            tr, va = tr_full.iloc[tr_idx], tr_full.iloc[va_idx]
        else:
            tmp_idx, te_idx = train_test_split(np.arange(len(tr_full)), test_size=self.cfg.test_size,
                                               stratify=y_full, random_state=rs)
            tmp = tr_full.iloc[tmp_idx]
            tr_idx, va_idx = train_test_split(np.arange(len(tmp)), test_size=self.cfg.val_size,
                                              stratify=y_full[tmp_idx], random_state=rs)
            tr, va, te = tmp.iloc[tr_idx], tmp.iloc[va_idx], tr_full.iloc[te_idx]

        Xtr, (Xva, Xte), names, const = self._fit_transform(tr, [va, te])
        ytr = (tr[LABEL] != "normal").astype(np.int64).values
        yva = (va[LABEL] != "normal").astype(np.int64).values
        yte = (te[LABEL] != "normal").astype(np.int64).values
        known = set(tr[LABEL].unique())
        novel = (~te[LABEL].isin(known)).values          # ataque cujo TIPO nunca foi visto
        logger.info("NSL-KDD carregado", split=self.cfg.split, scaling=self.cfg.scaling,
                    treino=Xtr.shape, val=Xva.shape, teste=Xte.shape,
                    frac_ataque_treino=round(float(ytr.mean()), 4),
                    frac_ataque_teste=round(float(yte.mean()), 4),
                    frac_teste_ataque_novo=round(float(novel.mean()), 4),
                    colunas_constantes_descartadas=const)
        return {"X_train": Xtr, "X_val": Xva, "X_test": Xte, "y_train": ytr, "y_val": yva,
                "y_test": yte, "test_novel_mask": novel, "feature_names": names,
                "dropped_constant": const, "test_labels": te[LABEL].values}

    def load_splits(self):
        m = self.load_splits_with_meta()
        return m["X_train"], m["X_val"], m["X_test"], m["y_train"], m["y_val"], m["y_test"]
