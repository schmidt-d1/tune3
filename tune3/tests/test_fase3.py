# tune3/tests/test_fase3.py
"""Testes-sentinela da Fase 3: objetivo CVaR e loop MACRO (BoTorch)."""
import warnings

import numpy as np
import pytest

from tune3.core.objectives import (
    cvar, value_at_risk, cvar_quantile, cvar_rockafellar,
)


# ------------------------------ CVaR ------------------------------ #
def test_cvar_decila_pior():
    """CVaR_0.90 de 0..99 = média da pior decila (~94.5)."""
    L = np.arange(100.0)
    assert abs(cvar(L, 0.90) - 94.5) < 1.0


def test_ordenacao_media_var_cvar():
    """media <= VaR <= CVaR (foco progressivo na cauda)."""
    rng = np.random.default_rng(0)
    L = rng.lognormal(0, 1, 5000)
    assert L.mean() <= value_at_risk(L, 0.95) <= cvar(L, 0.95)


def test_cvar_formulas_concordam():
    """quantile e rockafellar concordam."""
    rng = np.random.default_rng(0)
    L = rng.lognormal(0, 1, 5000)
    for g in [0.80, 0.90, 0.95, 0.99]:
        a, b = cvar_quantile(L, g), cvar_rockafellar(L, g)
        assert abs(a - b) / abs(b) < 0.02


def test_cvar_penaliza_cauda():
    """Mesma média, cauda mais pesada => CVaR maior."""
    A = np.full(100, 1.0)
    B = np.concatenate([np.full(95, 0.5), np.full(5, 10.5)])
    assert cvar(B, 0.95) > cvar(A, 0.95)


def test_cvar_vazio_levanta():
    with pytest.raises(ValueError):
        cvar([], 0.95)


# --------------------------- MACRO BO loop --------------------------- #
@pytest.mark.slow
def test_macro_bo_pareto_nao_dominado():
    """Smoke test do loop MACRO: produz frente de Pareto não-dominada."""
    warnings.filterwarnings("ignore")
    from tune3.macro.bo_loop import Tune3MacroLoop, MacroConfig

    def evaluate(x):
        x1, x2 = x["lr"], x["wd"]
        f1 = (x1 - 0.2) ** 2 + 0.3 * x2
        f2 = (x2 - 0.8) ** 2 + 0.3 * (1.0 - x1)
        return float(f1), float(f2)

    space = {"lr": (0.0, 1.0), "wd": (0.0, 1.0)}
    loop = Tune3MacroLoop(space, evaluate,
                          MacroConfig(n_init=6, n_iter=4, mc_samples=32,
                                      num_restarts=3, raw_samples=64, seed=0))
    pf = loop.run()
    assert pf["n_evaluations"] == 10
    Y = np.asarray(pf["objectives"])

    def dominates(a, b):
        return all(a <= b) and any(a < b)

    for i in range(len(Y)):
        for k in range(len(Y)):
            if i != k:
                assert not dominates(Y[k], Y[i]), "frente contém ponto dominado"


@pytest.mark.slow
def test_macro_knee_point():
    warnings.filterwarnings("ignore")
    from tune3.macro.bo_loop import Tune3MacroLoop, MacroConfig

    def evaluate(x):
        return float(x["a"] ** 2), float((1 - x["a"]) ** 2)

    loop = Tune3MacroLoop({"a": (0.0, 1.0)}, evaluate,
                          MacroConfig(n_init=5, n_iter=3, mc_samples=32,
                                      num_restarts=3, raw_samples=64, seed=1))
    loop.run()
    knee = loop.knee_point()
    assert "x" in knee and "objectives" in knee
