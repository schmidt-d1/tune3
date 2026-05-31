# tune3/tests/test_ddkf.py
"""Testes-sentinela do DDKFController (filtro de Kalman data-driven, nível MICRO)."""
import numpy as np
import pytest

from tune3.micro import DDKFController, DDKFConfig


def test_S_inf_estacionaria():
    """S_inf = sigma_eta^2 / (1 - rho^2)."""
    d = DDKFController(0.0, DDKFConfig(rho=0.95, sigma_eta=0.05))
    assert abs(d.S_inf - 0.05**2 / (1 - 0.95**2)) < 1e-12


def test_predict_ar1():
    """predict() = x_nom + rho*(x_hat - x_nom)."""
    d = DDKFController(0.0, DDKFConfig(rho=0.9, exploration_std=0.0))
    d.x_hat = 1.0
    assert abs(d.predict() - 0.9) < 1e-12


def test_warmup_predict_only():
    """Durante warmup (janela < min_samples), não há correção (u=0)."""
    d = DDKFController(0.0, DDKFConfig(min_samples=30, exploration_std=0.0))
    for _ in range(10):
        d.update([1.0, 1.0, 0.0])
    assert d.last_u == 0.0


def test_sinal_correcao_loss_positiva():
    """Cov(lr, loss) > 0 e loss atual alta => u > 0 (reduz lr)."""
    d = DDKFController(0.0, DDKFConfig(min_samples=20, window_size=60,
                                       exploration_std=0.0, max_correction=100.0))
    rng = np.random.default_rng(0)
    for _ in range(40):
        lr = rng.normal(0, 0.3)
        d.x_hat = lr
        d.update([2.0 * lr + rng.normal(0, 0.05), 2.0 * lr, -lr])
    d.x_hat = 0.0
    d.update([10.0, 10.0, 0.0])
    assert d.last_u > 0


def test_sinal_correcao_loss_negativa():
    """Cov(lr, loss) < 0 (loss cai com lr) e loss alta => u < 0 (aumenta lr)."""
    d = DDKFController(0.0, DDKFConfig(min_samples=20, window_size=60,
                                       exploration_std=0.0, max_correction=100.0))
    rng = np.random.default_rng(1)
    for _ in range(40):
        lr = rng.normal(0, 0.3)
        d.x_hat = lr
        d.update([-2.0 * lr + rng.normal(0, 0.05), -2.0 * lr, lr])
    d.x_hat = 0.0
    d.update([10.0, 10.0, 0.0])
    assert d.last_u < 0


def test_identificacao_r2_positivo():
    """Com exploração, o sistema é identificado (R^2 > 0)."""
    d = DDKFController(0.0, DDKFConfig(min_samples=30, window_size=80,
                                       exploration_std=0.05),
                       rng=np.random.default_rng(2))
    for _ in range(60):
        lr = d.x_hat
        d.update([2.0 * lr + np.random.randn() * 0.02, 2.0 * lr, -2.0 * lr])
    assert d.information_ratio() > 0


def test_clip_de_seguranca():
    """|u| <= max_correction * sqrt(S_inf)."""
    d = DDKFController(0.0, DDKFConfig(min_samples=5, window_size=20,
                                       max_correction=2.0, exploration_std=0.0),
                       rng=np.random.default_rng(0))
    for _ in range(20):
        d.update([np.random.randn() * 100, 0.0, 0.0])
    assert abs(d.last_u) <= 2.0 * np.sqrt(d.S_inf) + 1e-9


def test_rho_invalido_levanta():
    with pytest.raises(ValueError):
        DDKFConfig(rho=1.0)


def test_obs_dim_incorreta_levanta():
    d = DDKFController(0.0, DDKFConfig(obs_dim=3))
    with pytest.raises(ValueError):
        d.update([1.0, 2.0])  # dim 2, esperado 3


def test_learning_rate_property():
    """learning_rate = exp(x_hat)."""
    d = DDKFController(np.log(0.01), DDKFConfig())
    assert abs(d.learning_rate - 0.01) < 1e-9
