# tune3/data/registry.py
"""Ponto unico de carga de datasets para os scripts.

    load_dataset("drebin", "data/drebin215.csv", seed)         -> 6-tupla
    load_dataset("nslkdd", "data/nsl_kdd", seed, split="official", scaling="standard")

Retorna sempre (X_train, X_val, X_test, y_train, y_val, y_test), a interface do DREBIN.
"""
from __future__ import annotations

DATASETS = ("drebin", "nslkdd")
DEFAULT_PATH = {"drebin": "data/drebin215.csv", "nslkdd": "data/nsl_kdd"}


def load_dataset(name: str, path: str | None = None, seed: int = 0, **kw):
    if name == "drebin":
        from tune3.data.drebin import DrebinLoader, DrebinConfig
        return DrebinLoader(DrebinConfig(csv_path=path or DEFAULT_PATH[name],
                                         random_state=seed)).load_splits()
    if name == "nslkdd":
        from tune3.data.nslkdd import NSLKDDLoader, NSLKDDConfig
        return NSLKDDLoader(NSLKDDConfig(data_dir=path or DEFAULT_PATH[name],
                                         random_state=seed, **kw)).load_splits()
    raise ValueError(f"dataset desconhecido: {name!r}; validos: {DATASETS}")
