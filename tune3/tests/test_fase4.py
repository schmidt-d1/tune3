# tune3/tests/test_fase4.py
"""Testes-sentinela da Fase 4: integração (trial) e Tune3 completo."""
import warnings

import numpy as np
import pytest


def _synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 215)) < 0.15).astype(np.float32)
    w = np.zeros(215); w[:20] = 1.0
    score = X @ w + rng.normal(0, 1.5, n)
    y = (score > np.quantile(score, 0.63)).astype(int)
    return X, y


def test_trial_roda_e_retorna_objetivos():
    """Um trial retorna CVaR finito e curvatura >= 0."""
    from tune3.integration.trial import run_trial, TrialConfig
    Xtr, ytr = _synthetic(800, 0)
    Xv, yv = _synthetic(300, 1)
    res = run_trial(
        {"learning_rate": 0.05, "weight_decay": 1e-4, "dropout": 0.1,
         "hidden_dim": 64, "n_layers": 2},
        (Xtr, ytr, Xv, yv),
        TrialConfig(max_epochs=8, optimizer="sgd", device="cpu", curvature_every=4),
    )
    assert not res["aborted"]
    assert np.isfinite(res["cvar"]) and res["cvar"] < 1e3
    assert res["curvature"] >= 0.0  # Tr(H^2) >= 0 sempre


def test_trial_aborta_lr_explosivo():
    """lr absurdo com SGD => overflow numerico (loss nao-finita) OU spike detectado
    pelo Cantelli => trial devolve a penalidade de aborto.

    Nota (auditoria set/2026): com lr=50 o MLP nao diverge -- ele COLAPSA em ReLUs
    mortas e a loss fica constante (sem spike, Cantelli nao dispara). O teste
    antigo passava por acaso do caminho aleatorio. Usamos lr=1e3, que estoura
    para inf/NaN deterministicamente ja na 1a epoca.
    """
    from tune3.integration.trial import run_trial, TrialConfig
    from tune3.safety import CantelliConfig
    Xtr, ytr = _synthetic(800, 0)
    Xv, yv = _synthetic(300, 1)
    res = run_trial(
        {"learning_rate": 1e3, "weight_decay": 0.0, "dropout": 0.1,
         "hidden_dim": 64, "n_layers": 2},
        (Xtr, ytr, Xv, yv),
        TrialConfig(max_epochs=20, optimizer="sgd", device="cpu", batch_size=32,
                    cantelli=CantelliConfig(gamma=0.95, window_size=5)),
    )
    assert res["aborted"]
    assert res["cvar"] >= 100 and res["curvature"] >= 100   # penalidade de aborto


def test_trial_curvatura_nao_negativa_sempre():
    """Robustez: curvatura Tr(H^2) nunca negativa em vários hparams."""
    from tune3.integration.trial import run_trial, TrialConfig
    Xtr, ytr = _synthetic(600, 0)
    Xv, yv = _synthetic(200, 1)
    for lr in [0.01, 0.1]:
        res = run_trial(
            {"learning_rate": lr, "weight_decay": 1e-4, "dropout": 0.0,
             "hidden_dim": 32, "n_layers": 1},
            (Xtr, ytr, Xv, yv),
            TrialConfig(max_epochs=5, optimizer="sgd", device="cpu", curvature_every=2),
        )
        if not res["aborted"]:
            assert res["curvature"] >= 0.0


@pytest.mark.slow
def test_tune3_completo_ponta_a_ponta():
    """Tune3 completo: MACRO + trials -> frente de Pareto não-vazia."""
    warnings.filterwarnings("ignore")
    from tune3.integration.runner import run_tune3, Tune3RunConfig
    from tune3.integration.trial import TrialConfig
    from tune3.macro.bo_loop import MacroConfig
    Xtr, ytr = _synthetic(800, 0)
    Xv, yv = _synthetic(300, 1)
    out = run_tune3(
        (Xtr, ytr, Xv, yv),
        Tune3RunConfig(
            macro=MacroConfig(n_init=4, n_iter=3, mc_samples=16,
                              num_restarts=2, raw_samples=32, seed=0),
            trial=TrialConfig(max_epochs=5, optimizer="sgd", device="cpu", curvature_every=3),
        ),
    )
    assert out["pareto"]["n_evaluations"] == 7
    assert len(out["pareto"]["X"]) >= 1
    assert "learning_rate" in out["knee_hparams"]
