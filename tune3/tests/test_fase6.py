# tune3/tests/test_fase6.py
"""Testes-sentinela da Fase 6: protocolo estatístico."""
import numpy as np
import pytest

from tune3.experiments.stats import (
    cohens_dz, bootstrap_ci, min_achievable_p, compare_paired, summarize,
)


def test_cohens_dz_efeito_grande():
    """Diferença estável e positiva => d_z grande."""
    diffs = [0.10, 0.12, 0.08, 0.11, 0.09]
    assert cohens_dz(diffs) > 2.0


def test_cohens_dz_zero_quando_sem_variacao_ou_media_nula():
    assert cohens_dz([0.0, 0.0, 0.0]) == 0.0


def test_bootstrap_ci_exclui_zero_com_efeito_claro():
    lo, hi = bootstrap_ci([0.10, 0.12, 0.08, 0.11, 0.09], n_boot=3000, seed=0)
    assert lo > 0  # efeito real => IC não contém 0


def test_bootstrap_ci_inclui_zero_sem_efeito():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0, 0.1, 30)  # diferenças centradas em 0
    lo, hi = bootstrap_ci(diffs, n_boot=3000, seed=0)
    assert lo < 0 < hi


def test_min_achievable_p():
    """A lição do poder: Wilcoxon two-sided."""
    assert min_achievable_p(3) == 0.25
    assert min_achievable_p(5) == 0.0625
    assert min_achievable_p(6) < 0.05  # 0.03125


def test_compare_paired_holm_mais_conservador():
    """p_holm >= p_raw sempre (correção conservadora)."""
    rng = np.random.default_rng(0)
    n = 8
    tune3 = rng.normal(0.80, 0.02, n)
    base = {
        "b1": tune3 + rng.normal(0.10, 0.02, n),
        "b2": tune3 + rng.normal(0.08, 0.02, n),
        "b3": tune3 + rng.normal(0.05, 0.02, n),
    }
    res = compare_paired(list(tune3), {k: list(v) for k, v in base.items()},
                         lower_is_better=True)
    for r in res["comparisons"].values():
        assert r["p_holm"] >= r["p_raw"] - 1e-12


def test_compare_paired_detecta_tune3_melhor():
    """Com efeito claro e 8 seeds, Tune3 vence e é significativo."""
    rng = np.random.default_rng(1)
    n = 8
    tune3 = rng.normal(0.80, 0.02, n)
    base = {"b1": tune3 + rng.normal(0.15, 0.02, n)}
    res = compare_paired(list(tune3), {"b1": list(base["b1"])}, lower_is_better=True)
    assert res["comparisons"]["b1"]["win"]
    assert res["comparisons"]["b1"]["dz"] > 0
    assert res["comparisons"]["b1"]["significant"]


def test_compare_paired_valida_n_seeds():
    with pytest.raises(ValueError):
        compare_paired([1, 2, 3], {"b": [1, 2]})  # tamanhos diferentes


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["median"] == 3.0
    assert s["n"] == 5


@pytest.mark.slow
def test_protocolo_ponta_a_ponta():
    """Smoke test: protocolo completo (4 métodos x 2 seeds) em dados sintéticos."""
    import warnings
    warnings.filterwarnings("ignore")
    from tune3.experiments.protocol import run_protocol, ProtocolConfig

    def loader_fn(seed):
        rng = np.random.default_rng(seed)
        def make(n):
            X = (rng.random((n, 215)) < 0.15).astype(np.float32)
            w = np.zeros(215); w[:20] = 1.0
            s = X @ w + rng.normal(0, 1.5, n)
            y = (s > np.quantile(s, 0.63)).astype(int)
            return X, y
        Xtr, ytr = make(600); Xv, yv = make(200); Xte, yte = make(200)
        return (Xtr, ytr, Xv, yv, Xte, yte)

    cfg = ProtocolConfig(seeds=[0, 1], epochs=4, n_init=3, n_iter=1,
                         asha_configs=4, device="cpu")
    res = run_protocol(loader_fn, cfg)
    assert set(res["methods"]) == {"tune3", "tune3_bestcvar", "random_search", "asha", "sam"}
    assert all(len(v) == 2 for v in res["cvar_test_by_method"].values())
