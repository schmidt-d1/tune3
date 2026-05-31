# tune3/tests/test_cantelli.py
"""Testes-sentinela do CantelliGuard (seção 2.4 do manual)."""
import math

import numpy as np
import pytest

from tune3.safety import CantelliGuard, CantelliConfig


def test_c_gamma_095():
    """C_gamma(0.95) = sqrt(19) ~= 4.3589."""
    g = CantelliGuard(CantelliConfig(gamma=0.95, window_size=10))
    assert abs(g.C_gamma - math.sqrt(19)) < 1e-9


def test_c_gamma_outros_niveis():
    assert abs(CantelliGuard(CantelliConfig(gamma=0.90)).C_gamma - 3.0) < 1e-9
    assert abs(CantelliGuard(CantelliConfig(gamma=0.99)).C_gamma - math.sqrt(99)) < 1e-9


def test_fisher_const_default():
    """Constante de consistência MAD->sigma."""
    assert abs(CantelliConfig().fisher_const - 1.4826) < 1e-12


def test_no_abort_durante_warmup():
    """Não aborta enquanto a janela não atinge min_samples."""
    g = CantelliGuard(CantelliConfig(window_size=10))
    aborts = [g.should_abort(1.0) for _ in range(9)]
    assert sum(aborts) == 0


def test_abort_em_spike_catastrofico():
    """Após janela estável, um spike enorme dispara aborto."""
    g = CantelliGuard(CantelliConfig(window_size=20, gamma=0.95))
    rng = np.random.default_rng(0)
    for _ in range(20):
        g.should_abort(1.0 + rng.normal(0, 0.01))
    assert g.should_abort(100.0)


def test_mad_robusto_a_outlier_unico():
    """Um único outlier gigante na janela NÃO deve inflar nem colapsar o limiar."""
    g = CantelliGuard(CantelliConfig(window_size=20))
    for L in [1.0] * 19 + [1000.0]:
        g.should_abort(L)
    thr = g.threshold()
    assert thr is not None and thr < 10.0


def test_consistencia_mad_sigma():
    """1.4826 * MAD ~= sigma sob normalidade."""
    rng = np.random.default_rng(0)
    x = rng.normal(100.0, 5.0, 100000)
    mad = np.median(np.abs(x - np.median(x)))
    assert abs(1.4826 * mad - 5.0) < 0.05


def test_location_mean_configuravel():
    """location='mean' deve ser aceito."""
    g = CantelliGuard(CantelliConfig(window_size=10, location="mean"))
    for _ in range(10):
        g.should_abort(1.0)
    assert g.threshold() is not None


def test_gamma_invalido_levanta():
    with pytest.raises(ValueError):
        CantelliConfig(gamma=1.5)
