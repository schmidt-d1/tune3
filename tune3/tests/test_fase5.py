# tune3/tests/test_fase5.py
"""Testes-sentinela da Fase 5: baselines SAM, RandomSearch, ASHA."""
import warnings

import numpy as np
import pytest


def _synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 215)) < 0.15).astype(np.float32)
    w = np.zeros(215); w[:20] = 1.0
    s = X @ w + rng.normal(0, 1.5, n)
    y = (s > np.quantile(s, 0.63)).astype(int)
    return X, y


def _data():
    Xtr, ytr = _synthetic(1000, 0)
    Xv, yv = _synthetic(400, 1)
    return (Xtr, ytr, Xv, yv)


def _space():
    return {"log_lr": (-3.0, -1.0), "log_wd": (-6.0, -3.0),
            "dropout": (0.0, 0.4), "hidden_dim": (32.0, 128.0), "n_layers": (1.0, 3.0)}


# ------------------------------ SAM ------------------------------ #
def test_sam_dois_passos():
    """SAM exige first_step/second_step; step() direto deve falhar."""
    import torch
    from tune3.baselines.sam import SAM
    p = [torch.nn.Parameter(torch.randn(3))]
    opt = SAM(p, torch.optim.SGD, rho=0.05, lr=0.1)
    with pytest.raises(RuntimeError):
        opt.step()


def test_sam_trial_roda():
    from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
    r = plain_trial(
        {"learning_rate": 0.05, "weight_decay": 1e-4, "dropout": 0.1,
         "hidden_dim": 64, "n_layers": 2},
        _data(),
        PlainTrialConfig(max_epochs=6, optimizer="sam", device="cpu", sam_rho=0.05),
    )
    assert np.isfinite(r["cvar"]) and r["curvature"] >= 0.0


# --------------------------- Random Search --------------------------- #
def test_random_search():
    from tune3.baselines.hpo import RandomSearchHPO
    from tune3.baselines.plain_trial import PlainTrialConfig
    rs = RandomSearchHPO(_space(), _data(), n_trials=5,
                         trial_cfg=PlainTrialConfig(max_epochs=5, optimizer="sgd", device="cpu"),
                         seed=0)
    out = rs.run()
    assert out["n_evaluations"] == 5
    assert np.isfinite(out["best"]["cvar"])
    assert "learning_rate" in out["best"]["hparams"]


# ------------------------------- ASHA ------------------------------- #
def test_asha_economiza_orcamento():
    """ASHA deve gastar menos épocas que treinar todos os configs por completo."""
    from tune3.baselines.hpo import ASHA
    from tune3.baselines.plain_trial import PlainTrialConfig
    n_configs, max_ep = 8, 8
    asha = ASHA(_space(), _data(), n_configs=n_configs, r0=2, eta=2,
                trial_cfg=PlainTrialConfig(max_epochs=max_ep, optimizer="sgd", device="cpu"),
                seed=0)
    out = asha.run()
    assert out["total_epochs"] < n_configs * max_ep  # economia via early-stopping
    assert np.isfinite(out["best"]["cvar"])


def test_asha_retorna_um_vencedor():
    from tune3.baselines.hpo import ASHA
    from tune3.baselines.plain_trial import PlainTrialConfig
    asha = ASHA(_space(), _data(), n_configs=4, r0=2, eta=2,
                trial_cfg=PlainTrialConfig(max_epochs=4, optimizer="sgd", device="cpu"),
                seed=1)
    out = asha.run()
    assert "hparams" in out["best"]
