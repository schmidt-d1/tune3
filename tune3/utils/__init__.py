# tune3/utils/__init__.py
# `ModelTrainer` (pipeline legado Optuna/Breast-Cancer) foi movido para legacy/.
from .curvature import hutchinson_trace_estimator

__all__ = ["hutchinson_trace_estimator"]
