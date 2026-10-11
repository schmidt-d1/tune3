# tune3/data/kronodroid.py
"""Loader do KronoDroid (Guerra-Manzanares, Bahsi & Nomm, Computers & Security 110, 2021) com
DIVISAO TEMPORAL -- etapa F3 do Tune3 (regra fixada em claude/TUNE3_PREREGISTRO_FASE_DIRECAO.md,
secao "F3 -- regra fixada", ANTES de qualquer modelo).

POR QUE ESTE DATASET. E' o unico dos nossos com carimbo de tempo por amostra (2008-2020), o que
permite o deslocamento REAL de producao em deteccao de malware: treinar no passado, testar no
futuro (Pendlebury et al., TESSERACT, 2019). NSL-KDD e DREBIN por cluster sao deslocamentos
construidos (tipo de ataque / familia retida); aqui o tempo faz a divisao.

FORMATO (conferido nos arquivos reais em 11/10/2026, inspecao em results/inspect/*kronodroid-real*):
  real_device/real_malware_v1.zip    -> real_malware_v1.csv     41 382 linhas x 484 colunas
  real_device/real_legitimate_v1.zip -> real_legitimate_v1.csv  36 755 linhas x 484 colunas
  colunas: Package | Malware (rotulo) | 288 chamadas de sistema (contagens; execve ... SYS_369) |
           nr_syscalls | 166 permissoes (0/1; ACCEPT_HANDOVER ... WRITE_VOICEMAIL) | nr_permissions,
           normal, dangerous, signature, custom_yes, nr_custom, total_perm | sha256 | CFileSize,
           UFileSize | EarliestModDate, HighestModDate (texto M/D/AAAA) | FilesInsideAPK |
           TimesSubmitted | NrContactedIps | Activities, NrIntServices, NrIntServicesActions,
           NrIntActivities, NrIntActivitiesActions, NrIntReceivers, NrIntReceiversActions,
           TotalIntentFilters, NrServices | Scanners | Detection_Ratio | MalFamily
  (o emulador usa FirstModDate/LastModDate e o arquivo de malware nao tem FilesInsideAPK; o loader
   aceita esses nomes, mas a F3 usa so' o real device.)

EIXO DE TEMPO: HighestModDate (ultima modificacao entre os arquivos do APK). Preferido ao
EarliestModDate porque este tem 6 945 datas-placeholder em 1980 (epoca do DOS) contra 24; os dois sao
interpretaveis em 98 %. Linhas com data nao interpretavel ou ano fora de [2008, 2020] sao descartadas.

DIVISAO: treino e validacao sorteados (estratificados por rotulo) do periodo ano <= cut_year; teste =
tudo apos o corte, dividido em "proximo" (cut < ano < far_year) e "distante" (ano >= far_year); a
mascara test_novel_mask marca o DISTANTE (e' o papel de "ataques novos"/"cluster novo" dos outros E0s).
A fracao de malware do teste e' CASADA com a do periodo de treino, separadamente no proximo e no
distante, subamostrando a classe majoritaria (sorteio com semente): assim validacao e teste diferem
no TEMPO, nao na mistura de classes. match_ratio=False mantem a proporcao natural (exploratorio).

ATRIBUTOS (474 candidatos): chamadas de sistema + nr_syscalls, permissoes (0/1, sem
escala), contagens de permissoes, tamanhos de arquivo, contagens do manifesto. EXCLUIDOS: Package,
sha256 (identificadores), as duas datas (eixo de tempo), MalFamily / Scanners / Detection_Ratio /
TimesSubmitted (metadados do VirusTotal: vazamento de rotulo) e NrContactedIps (proveniencia incerta).
Escala: log1p nas contagens e tamanhos, depois z-score ajustado SO' no treino, limitado a +-10 (o futuro
traz contagens a centenas de desvios do treino; o limite preserva o deslocamento sem alavancagem numerica);
faltantes -> 0. Colunas constantes ou com menos de 50 linhas nao-nulas no treino (nao aprendiveis; 46 delas
sao permissoes que so' existem depois de 2014) sao descartadas: ficam 165 atributos.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

logger = structlog.get_logger()

LABEL = "Malware"
TIME_COLS = ("HighestModDate", "LastModDate")          # real device | emulador
EXCLUDE = ["Package", "sha256", "EarliestModDate", "HighestModDate", "FirstModDate", "LastModDate",
           "TimesSubmitted", "NrContactedIps", "Scanners", "Detection_Ratio", "MalFamily", LABEL]
SYSCALL_END = "nr_syscalls"          # ultima coluna do bloco de chamadas de sistema (inclusive)
PERM_END = "nr_permissions"          # primeira coluna DEPOIS do bloco de permissoes 0/1


@dataclass
class KronoDroidConfig:
    data_dir: str = "data/kronodroid/real_device"
    malware_file: str = "real_malware_v1.zip"
    legit_file: str = "real_legitimate_v1.zip"
    cut_year: int = 2014          # treino/validacao: ano <= cut; teste: ano > cut
    far_year: int = 2018          # teste distante: ano >= far (mascara 'novel'); proximo: cut < ano < far
    year_min: int = 2008
    year_max: int = 2020
    max_train: Optional[int] = 25000
    n_val: int = 5000
    match_ratio: bool = True      # casa a fracao de malware do teste (proximo e distante) com a do periodo de treino
    random_state: int = 0
    scaling: str = "log_standard"  # "log_standard" | "standard" | "none"
    min_support: int = 50         # coluna com menos de min_support linhas nao-nulas no TREINO e' descartada (nao aprendivel)
    clip_z: Optional[float] = 10.0  # valores padronizados limitados a [-clip_z, clip_z] em todos os conjuntos (None = sem limite)

    def __post_init__(self):
        if self.scaling not in ("log_standard", "standard", "none"):
            raise ValueError(f"scaling invalido: {self.scaling!r}")
        if not (self.year_min <= self.cut_year < self.far_year <= self.year_max):
            raise ValueError("esperado year_min <= cut_year < far_year <= year_max")


def _read_zip_csv(path: Path) -> pd.DataFrame:
    """Le o primeiro CSV dentro do ZIP (ou um CSV solto) sem extrair para o disco."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"{path}: nenhum CSV dentro do ZIP")
            with z.open(names[0]) as f:
                return pd.read_csv(io.BytesIO(f.read()), low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _time_col(df: pd.DataFrame) -> str:
    for c in TIME_COLS:
        if c in df.columns:
            return c
    raise ValueError(f"nenhuma coluna de tempo entre {TIME_COLS}")


def feature_groups(columns: List[str]) -> Dict[str, List[str]]:
    """Separa as colunas em grupos pela estrutura do CSV (separadores nr_syscalls / nr_permissions)."""
    cols = list(columns)
    i_sys = cols.index(SYSCALL_END); i_perm = cols.index(PERM_END)
    syscalls = [c for c in cols[: i_sys + 1] if c not in EXCLUDE]
    perms = [c for c in cols[i_sys + 1: i_perm] if c not in EXCLUDE]
    counts = [c for c in cols[i_perm:] if c not in EXCLUDE]
    return {"syscalls": syscalls, "permissions": perms, "counts": counts}


def _subsample_to_ratio(y: np.ndarray, p_mal: float, rng: np.random.Generator) -> np.ndarray:
    """Indices (ordenados) de uma subamostra com fracao de malware p_mal, descartando so' da classe
    majoritaria em relacao a p_mal. Mantem todas as linhas da classe limitante."""
    mal = np.flatnonzero(y == 1); ben = np.flatnonzero(y == 0)
    if len(mal) == 0 or len(ben) == 0:
        return np.arange(len(y))
    n_total = min(len(mal) / p_mal, len(ben) / (1.0 - p_mal))
    n_mal = min(len(mal), int(round(n_total * p_mal))); n_ben = min(len(ben), int(round(n_total * (1.0 - p_mal))))
    keep = np.concatenate([rng.choice(mal, n_mal, replace=False), rng.choice(ben, n_ben, replace=False)])
    return np.sort(keep)


class KronoDroidLoader:
    def __init__(self, config: Optional[KronoDroidConfig] = None):
        self.cfg = config or KronoDroidConfig()

    # ------------------------------------------------------------------ leitura
    def _load_frame(self) -> Tuple[pd.DataFrame, np.ndarray, str]:
        d = Path(self.cfg.data_dir)
        mal = _read_zip_csv(d / self.cfg.malware_file); leg = _read_zip_csv(d / self.cfg.legit_file)
        if LABEL in mal.columns and not (mal[LABEL] == 1).all():
            raise ValueError("arquivo de malware com rotulo != 1")
        if LABEL in leg.columns and not (leg[LABEL] == 0).all():
            raise ValueError("arquivo de benignos com rotulo != 0")
        common = [c for c in mal.columns if c in set(leg.columns)]     # intersecao: coluna so' de um lado vazaria o rotulo
        dropped = sorted((set(mal.columns) | set(leg.columns)) - set(common))
        df = pd.concat([mal[common], leg[common]], ignore_index=True)
        y = np.concatenate([np.ones(len(mal), np.int64), np.zeros(len(leg), np.int64)])
        tcol = _time_col(df)
        year = pd.to_datetime(df[tcol].astype(str).str.strip(), format="%m/%d/%Y", errors="coerce").dt.year
        ok = year.between(self.cfg.year_min, self.cfg.year_max).values
        self._dropped_cols = dropped; self._n_bad_year = int((~ok).sum())
        df = pd.concat([df.loc[ok].reset_index(drop=True), pd.Series(year[ok].astype(int).values, name="_year")], axis=1)
        return df, y[ok], tcol

    # ------------------------------------------------------------------ preprocess
    def _fit_transform(self, tr: pd.DataFrame, others: List[pd.DataFrame]):
        g = feature_groups([c for c in tr.columns if c != "_year"])
        count_cols = g["syscalls"] + g["counts"]; bin_cols = g["permissions"]
        allc = count_cols + bin_cols
        tr = tr[allc].astype(np.float64).fillna(0.0); others = [o[allc].astype(np.float64).fillna(0.0) for o in others]
        # descarte: constantes no treino e colunas com suporte (linhas nao-nulas no treino) < min_support --
        # com 1-49 exemplos em 25 000 nada se aprende, e o z-score dessas colunas explode no teste
        support = (tr != 0).sum()
        const = [c for c in allc if tr[c].nunique() <= 1 or support[c] < self.cfg.min_support]
        count_cols = [c for c in count_cols if c not in const]; bin_cols = [c for c in bin_cols if c not in const]
        keep = count_cols + bin_cols
        Xtr = tr[keep].copy(); Xo = [o[keep].copy() for o in others]
        if self.cfg.scaling == "log_standard":
            for X in [Xtr] + Xo:
                X[count_cols] = np.log1p(X[count_cols].clip(lower=0))
        if self.cfg.scaling in ("log_standard", "standard"):
            mu = Xtr[count_cols].mean(); sd = Xtr[count_cols].std(ddof=0).replace(0.0, 1.0)
            for X in [Xtr] + Xo:
                X[count_cols] = (X[count_cols] - mu) / sd
                if self.cfg.clip_z is not None:      # o deslocamento temporal leva contagens a centenas de
                    X[count_cols] = X[count_cols].clip(-self.cfg.clip_z, self.cfg.clip_z)   # desvios: limitado
        groups = {"syscalls": [c for c in g["syscalls"] if c in keep], "permissions": bin_cols,
                  "counts": [c for c in g["counts"] if c in keep]}
        return (Xtr.values.astype(np.float32), [X.values.astype(np.float32) for X in Xo], keep, const, groups)

    # ------------------------------------------------------------------ public
    def load_splits_with_meta(self) -> Dict:
        cfg = self.cfg; rng = np.random.default_rng(cfg.random_state)
        df, y, tcol = self._load_frame()
        yr = df["_year"].values
        pool = np.flatnonzero(yr <= cfg.cut_year); test = np.flatnonzero(yr > cfg.cut_year)
        if len(pool) == 0 or len(test) == 0:
            raise ValueError("corte temporal deixa treino ou teste vazio")
        p_mal = float(y[pool].mean())

        # treino + validacao: estratificados dentro do periodo <= corte
        n_tv = len(pool) if cfg.max_train is None else min(len(pool), cfg.max_train + cfg.n_val)
        if n_tv < len(pool):
            keep, _ = train_test_split(pool, train_size=n_tv, stratify=y[pool], random_state=cfg.random_state)
            pool = np.sort(keep)
        n_val = min(cfg.n_val, len(pool) - 1)
        tr_idx, va_idx = train_test_split(pool, test_size=n_val, stratify=y[pool], random_state=cfg.random_state)
        tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)

        # teste: proximo e distante, cada um casado com p_mal (ou natural)
        near = test[yr[test] < cfg.far_year]; far = test[yr[test] >= cfg.far_year]
        if cfg.match_ratio:
            near = near[_subsample_to_ratio(y[near], p_mal, rng)] if len(near) else near
            far = far[_subsample_to_ratio(y[far], p_mal, rng)] if len(far) else far
        te_idx = np.sort(np.concatenate([near, far]))
        novel = yr[te_idx] >= cfg.far_year

        Xtr, (Xva, Xte), names, const, groups = self._fit_transform(df.iloc[tr_idx], [df.iloc[va_idx], df.iloc[te_idx]])
        ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]
        meta = {"time_col": tcol, "cut_year": cfg.cut_year, "far_year": cfg.far_year, "match_ratio": cfg.match_ratio,
                "n_rows_valid": int(len(df)), "n_rows_bad_year": self._n_bad_year, "dropped_columns": self._dropped_cols,
                "p_mal_train_period": p_mal, "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
                "n_test": int(len(te_idx)), "n_test_near": int((~novel).sum()), "n_test_far": int(novel.sum()),
                "frac_mal_train": float(ytr.mean()), "frac_mal_val": float(yva.mean()),
                "frac_mal_test_near": float(yte[~novel].mean()) if (~novel).any() else None,
                "frac_mal_test_far": float(yte[novel].mean()) if novel.any() else None,
                "n_features": int(Xtr.shape[1]), "n_syscalls": len(groups["syscalls"]),
                "n_permissions": len(groups["permissions"]), "n_counts": len(groups["counts"]),
                "dropped_constant": const}
        logger.info("KronoDroid carregado", **{k: v for k, v in meta.items() if k != "dropped_constant"})
        return {"X_train": Xtr, "X_val": Xva, "X_test": Xte, "y_train": ytr, "y_val": yva, "y_test": yte,
                "test_novel_mask": novel, "test_year": yr[te_idx], "feature_names": names,
                "feature_groups": groups, "dropped_constant": const, "meta": meta}

    def load_splits(self):
        m = self.load_splits_with_meta()
        return m["X_train"], m["X_val"], m["X_test"], m["y_train"], m["y_val"], m["y_test"]
