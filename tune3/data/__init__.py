"""Dados do Tune3. Os simbolos de `shift` (S2, clusterizacao) sao carregados sob demanda:
importar o pacote nao deve puxar `sklearn.cluster` -> `sklearn.neighbors._kd_tree`, extensao
compilada que o Controle Inteligente de Aplicativos do Windows bloqueia em algumas maquinas."""
from .drebin import DrebinLoader, DrebinConfig
from .imbalance import make_imbalanced

_LAZY = {"cluster_holdout_split", "cluster_malware", "n_malware_clusters"}
__all__ = ["DrebinLoader", "DrebinConfig", "make_imbalanced", *sorted(_LAZY)]


def __getattr__(name):
    if name in _LAZY:
        from . import shift
        return getattr(shift, name)
    raise AttributeError(f"module 'tune3.data' has no attribute {name!r}")
