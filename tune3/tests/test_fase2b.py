# tune3/tests/test_fase2b.py
"""Testes-sentinela da Fase 2b: RegimeGate, EoSDetector, GSNREstimator."""
import math

import numpy as np
import pytest

from tune3.micro.regime_gating import RegimeGate, RegimeGatingConfig
from tune3.safety.eos_detector import EoSDetector, EoSConfig
from tune3.curvature.gsnr import GSNREstimator, GSNRConfig


# ----------------------------- RegimeGate ----------------------------- #
def test_regime_classify():
    g = RegimeGate(RegimeGatingConfig(r2_low=0.10, r2_high=0.50))
    assert g.classify(0.05) == "A"
    assert g.classify(0.30) == "B"
    assert g.classify(0.80) == "C"


def test_regime_gating_duro():
    g = RegimeGate(RegimeGatingConfig(r2_low=0.10, r2_high=0.50))
    assert g.gating_factor(0.05) == 0.0   # Regime A: predict-only
    assert g.gating_factor(0.30) == 1.0   # Regime B: filtra
    assert g.gating_factor(0.90) == 1.0   # Regime C: filtra


def test_regime_gating_suave():
    g = RegimeGate(RegimeGatingConfig(r2_low=0.10, r2_high=0.50, smooth=True))
    assert g.gating_factor(0.10) == 0.0
    assert g.gating_factor(0.50) == 1.0
    assert abs(g.gating_factor(0.30) - 0.5) < 1e-9


def test_regime_fractions_somam_um():
    g = RegimeGate()
    for r in [0.05, 0.06, 0.30, 0.80]:
        g.observe(r)
    fr = g.fractions()
    assert abs(sum(fr.values()) - 1.0) < 1e-9


def test_regime_config_invalida():
    with pytest.raises(ValueError):
        RegimeGatingConfig(r2_low=0.6, r2_high=0.3)


# ----------------------------- EoSDetector ----------------------------- #
def test_eos_nao_dispara_estavel():
    d = EoSDetector(EoSConfig(window=6, patience=4, growth_threshold=1.5))
    rng = np.random.default_rng(0)
    fired = [d.update(1.0 + rng.normal(0, 0.01)) for _ in range(12)]
    assert sum(fired) == 0


def test_eos_dispara_sharpening():
    d = EoSDetector(EoSConfig(window=6, patience=4, growth_threshold=1.5))
    fired = [d.update(v) for v in [1.0, 1.2, 1.5, 1.9, 2.4, 3.0]]
    assert any(fired)


def test_eos_cooldown():
    d = EoSDetector(EoSConfig(window=6, patience=4, growth_threshold=1.5, cooldown=5))
    for v in [1.0, 1.2, 1.5, 1.9, 2.4, 3.0]:
        d.update(v)
    extra = [d.update(v) for v in [3.5, 4.0, 4.5]]
    assert sum(extra) == 0


def test_eos_lr_factor():
    d = EoSDetector(EoSConfig(lr_factor=0.7))
    assert abs(d.suggested_lr_factor() - 0.7) < 1e-12


def test_eos_config_invalida():
    with pytest.raises(ValueError):
        EoSConfig(lr_factor=1.5)


# ----------------------------- GSNREstimator ----------------------------- #
def test_gsnr_alto_vs_baixo():
    est = GSNREstimator()
    rng = np.random.default_rng(0)
    D = 50
    true_g = rng.normal(0, 1, D)
    grads_hi = [true_g + rng.normal(0, 0.05, D) for _ in range(8)]
    grads_lo = [rng.normal(0, 1, D) for _ in range(8)]
    assert est.from_gradients(grads_hi) > est.from_gradients(grads_lo) * 10


def test_gsnr_formula():
    est = GSNREstimator()
    rng = np.random.default_rng(1)
    grads = [rng.normal(0, 1, 30) for _ in range(6)]
    G = np.stack(grads)
    expected = (G.mean(0) @ G.mean(0)) / (G.var(0, ddof=1).sum() + 1e-12)
    assert abs(est.from_gradients(grads) - expected) < 1e-6


def test_gsnr_log_finito():
    est = GSNREstimator()
    rng = np.random.default_rng(2)
    grads = [rng.normal(0, 1, 20) for _ in range(8)]
    assert np.isfinite(est.log_gsnr(grads))


def test_gsnr_min_batches():
    est = GSNREstimator()
    with pytest.raises(ValueError):
        est.from_gradients([np.ones(10)])  # só 1 minibatch
