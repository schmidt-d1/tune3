# tune3/tests/test_auditoria_2026_09.py
"""
Testes-sentinela da auditoria de set/2026. Cada teste captura UMA regressao
encontrada no repositorio e corrigida:

  A1  vazamento: teste usado como validacao na avaliacao final
  A2  EoSDetector inerte (recebia curvatura repetida, nunca disparava)
  A3  Hutchinson em model.train() (dropout ativo durante a estimacao)
  A4  Hutchinson: estimador vs Hessiana EXATA por caminho independente
  A5  ablacao DDKF (E2): com ddkf_enabled=False o lr nao muda
  A6  estatistica: BCa + regra pre-registrada (p_holm<0.05 E |d_z|>=0.30)
  A7  agregacao por seed em S2 (anti pseudo-replicacao)
  A8  registro padronizado de resultados (results_io)
  A9  B4 e placebo existem e sao configuraveis (E1)
"""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn


def _synthetic(n, seed=0, flip=False):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 215)) < 0.15).astype(np.float32)
    w = np.zeros(215); w[:20] = 1.0
    s = X @ w + rng.normal(0, 1.5, n)
    y = (s > np.quantile(s, 0.63)).astype(int)
    if flip:
        y = 1 - y
    return X, y


HP = {"learning_rate": 0.05, "weight_decay": 1e-4, "dropout": 0.1, "hidden_dim": 32, "n_layers": 2}


# ------------------------------------------------------------------ A1
def test_A1_teste_nao_influencia_escolha_de_epoca():
    """A melhor epoca e o CVaR de VALIDACAO nao podem depender do conjunto de teste.
    Rodamos o mesmo trial com dois testes diferentes (um com rotulos INVERTIDOS):
    val/best_epoch identicos; cvar_test muito diferente."""
    from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
    Xtr, ytr = _synthetic(600, 0); Xv, yv = _synthetic(200, 1)
    Xte, yte = _synthetic(200, 2); Xte_bad, yte_bad = _synthetic(200, 2, flip=True)
    cfg = PlainTrialConfig(max_epochs=8, optimizer="sgd", device="cpu", seed=0)
    r_ok = plain_trial(HP, (Xtr, ytr, Xv, yv), cfg, test_data=(Xte, yte))
    r_bad = plain_trial(HP, (Xtr, ytr, Xv, yv), cfg, test_data=(Xte_bad, yte_bad))
    r_none = plain_trial(HP, (Xtr, ytr, Xv, yv), cfg)
    assert r_ok["best_epoch"] == r_bad["best_epoch"] == r_none["best_epoch"]
    assert abs(r_ok["cvar"] - r_bad["cvar"]) < 1e-6 and abs(r_ok["cvar"] - r_none["cvar"]) < 1e-6
    assert "cvar_test" in r_ok and "cvar_test" not in r_none
    assert r_bad["cvar_test"] > r_ok["cvar_test"]          # teste invertido e' pior


def test_A1_protocolo_avalia_no_teste_com_val_separada():
    """_eval_on_test deve devolver cvar_test E cvar_val (val != teste)."""
    from tune3.experiments.protocol import _eval_on_test, ProtocolConfig
    Xtr, ytr = _synthetic(600, 0); Xv, yv = _synthetic(200, 1); Xte, yte = _synthetic(200, 2)
    cfg = ProtocolConfig(seeds=[0], epochs=6, device="cpu")
    r = _eval_on_test(HP, (Xtr, ytr, Xv, yv, Xte, yte), 0, cfg)
    assert {"cvar_test", "cvar_val", "best_epoch", "test_loss"} <= set(r)
    assert np.isfinite(r["cvar_test"]) and np.isfinite(r["cvar_val"])


# ------------------------------------------------------------------ A2
def test_A2_eos_recebe_apenas_curvaturas_novas(monkeypatch):
    """EoSDetector.update deve ser chamado exatamente uma vez por RE-ESTIMATIVA
    de curvatura (nao uma vez por epoca com valor repetido)."""
    from tune3.integration import trial as trial_mod
    calls = []
    orig = trial_mod.EoSDetector.update
    def spy(self, c):
        calls.append(c); return orig(self, c)
    monkeypatch.setattr(trial_mod.EoSDetector, "update", spy)
    Xtr, ytr = _synthetic(400, 0); Xv, yv = _synthetic(150, 1)
    res = trial_mod.run_trial(HP, (Xtr, ytr, Xv, yv),
                              trial_mod.TrialConfig(max_epochs=10, patience=100, device="cpu",
                                                    curvature_every=5))
    assert len(calls) == res["n_curvature_estimates"] == 3      # epocas 0, 5 e 9 (ultima)


def test_A2_eos_bug_original_documentado():
    """Reproduz o bug: valor repetido 5x entre estimativas => nunca dispara,
    mesmo com a curvatura dobrando a cada estimativa."""
    from tune3.safety.eos_detector import EoSDetector, EoSConfig
    stale = EoSDetector(EoSConfig()); fresh = EoSDetector(EoSConfig())
    last = 1.0
    for ep in range(60):
        if ep % 5 == 0:
            last = 2.0 ** (ep // 5)
            fresh.update(last)
        stale.update(last)
    assert stale.n_triggers == 0
    assert fresh.n_triggers >= 1


# ------------------------------------------------------------------ A3
def test_A3_hutchinson_roda_em_eval(monkeypatch):
    """Durante estimate() o modelo deve estar em eval() (dropout desligado)."""
    import importlib
    trial_mod = importlib.import_module("tune3.integration.trial")
    pt_mod = importlib.import_module("tune3.baselines.plain_trial")
    from tune3.curvature.hutchinson import HutchinsonEstimator
    seen = []
    orig = HutchinsonEstimator.estimate
    def spy(self, model, loss):
        seen.append(model.training); return orig(self, model, loss)
    monkeypatch.setattr(HutchinsonEstimator, "estimate", spy)   # classe unica, compartilhada
    Xtr, ytr = _synthetic(400, 0); Xv, yv = _synthetic(150, 1)
    trial_mod.run_trial(HP, (Xtr, ytr, Xv, yv), trial_mod.TrialConfig(max_epochs=4, device="cpu", curvature_every=2))
    pt_mod.plain_trial(HP, (Xtr, ytr, Xv, yv), pt_mod.PlainTrialConfig(max_epochs=3, device="cpu"))
    assert len(seen) >= 3 and not any(seen)


# ------------------------------------------------------------------ A4
def test_A4_hutchinson_vs_hessiana_exata():
    """Independencia: lado A = estimador do repositorio; lado B = ||H||_F^2 com a
    Hessiana EXATA (torch.autograd.functional.hessian, sem usar o estimador).
    (i) HVP do estimador == H v  (mesmas sondas)  a 1e-5;
    (ii) media de M=3000 sondas dentro de 4 erros-padrao do exato."""
    from tune3.models.factory import build_model
    from tune3.curvature import HutchinsonEstimator, HutchinsonConfig
    torch.manual_seed(0)
    model = build_model(10, 2, hparams={"hidden_dim": 6, "n_layers": 2, "dropout": 0.0}).eval()
    X = torch.randn(40, 10); y = (X[:, 0] > 0).long(); crit = nn.CrossEntropyLoss()
    params = list(model.parameters())

    def flat_loss(th):
        idx = 0; out = {}
        for k, p in model.named_parameters():
            out[k] = th[idx:idx + p.numel()].view_as(p); idx += p.numel()
        return crit(torch.func.functional_call(model, out, (X,)), y)

    theta = torch.cat([p.detach().reshape(-1) for p in params]).clone().requires_grad_(True)
    H = torch.autograd.functional.hessian(flat_loss, theta).detach()
    exact = float((H * H).sum())

    # (i) HVP identico
    grads = torch.autograd.grad(crit(model(X), y), params, create_graph=True)
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        v = (torch.randint(0, 2, (theta.numel(),), generator=g) * 2 - 1).float()
        vs = []; idx = 0
        for p in params:
            vs.append(v[idx:idx + p.numel()].view_as(p)); idx += p.numel()
        gv = sum((gi * vi).sum() for gi, vi in zip(grads, vs))
        Hv = torch.cat([h.reshape(-1) for h in torch.autograd.grad(gv, params, retain_graph=True)])
        assert float((Hv - H @ v).abs().max()) < 1e-5

    # (ii) estimador converge para o exato
    est = HutchinsonEstimator(HutchinsonConfig(num_probes=3000))
    val = est.estimate(model, crit(model(X), y))
    # variancia por sonda estimada a partir da Hessiana exata (independente do estimador)
    vals = []
    for _ in range(2000):
        v = (torch.randint(0, 2, (theta.numel(),)) * 2 - 1).float()
        vals.append(float(((H @ v) ** 2).sum()))
    se = np.std(vals, ddof=1) / math.sqrt(3000)
    assert abs(val - exact) < 4 * se + 1e-9, f"est={val} exato={exact} se={se}"


# ------------------------------------------------------------------ A5
def test_A5_ablacao_ddkf_desligado_nao_mexe_no_lr():
    from tune3.integration.trial import run_trial, TrialConfig
    from tune3.safety import EoSConfig
    Xtr, ytr = _synthetic(400, 0); Xv, yv = _synthetic(150, 1)
    # EoS com patience alto para nao interferir no lr neste teste
    res = run_trial(HP, (Xtr, ytr, Xv, yv),
                    TrialConfig(max_epochs=6, device="cpu", ddkf_enabled=False,
                                eos=EoSConfig(patience=1000)))
    assert res["n_ddkf_active_epochs"] == 0
    assert res["lr_change_fraction"] == 0.0
    assert abs(res["lr_final"] - HP["learning_rate"]) < 1e-12


def test_A5_telemetria_presente():
    from tune3.integration.trial import run_trial, TrialConfig
    Xtr, ytr = _synthetic(400, 0); Xv, yv = _synthetic(150, 1)
    res = run_trial(HP, (Xtr, ytr, Xv, yv), TrialConfig(max_epochs=3, device="cpu"))
    for k in ["regime_fractions", "n_eos_triggers", "n_epochs_run", "n_ddkf_active_epochs",
              "n_curvature_estimates", "lr0", "lr_final", "lr_change_fraction"]:
        assert k in res


# ------------------------------------------------------------------ A6
def test_A6_bca_e_regra_pre_registrada():
    from tune3.experiments.stats import bootstrap_ci, compare_paired
    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 0.1, 20)
    lo, hi = bootstrap_ci(d, n_boot=2000, statistic="dz", ci_method="bca")
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    lo_m, hi_m = bootstrap_ci(d, n_boot=2000, statistic="mean", ci_method="bca")
    assert lo_m < d.mean() < hi_m
    # degenerado (todas as diferencas iguais) nao explode -- com n suficiente
    lo_d, hi_d = bootstrap_ci([1.0] * 8, statistic="dz")
    assert lo_d == hi_d and np.isfinite(lo_d)
    # rev. 4 (23/09/2026): com n < 6 o bootstrap nao e' inferencia -> IC NaN, nao um numero enganoso
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        assert all(np.isnan(v) for v in bootstrap_ci([1.0, 1.0, 1.0], statistic="dz"))
        assert all(np.isnan(v) for v in bootstrap_ci([0.1, 0.2, 0.15], statistic="mean"))

    t = rng.normal(0.8, 0.02, 10)
    base = {"grande": list(t + 0.15 + rng.normal(0, 0.01, 10)),     # efeito claro
            "minusculo": list(t + 0.0005 + rng.normal(0, 0.0001, 10))}  # signif. mas |dz| pode ser grande...
    r = compare_paired(list(t), base, lower_is_better=True, n_boot=1000)
    g = r["comparisons"]["grande"]
    assert g["prereg_win"] and g["significant"] and g["practical"] and g["win"]
    assert "dz_ci95" in g and r["ci_method"] == "bca" and r["dz_min"] == 0.30
    # regra: sem significancia nao ha vitoria, por maior que seja o d_z
    r3 = compare_paired([1.0, 0.9, 1.1], {"b": [1.5, 1.4, 1.6]}, lower_is_better=True, n_boot=200)
    c = r3["comparisons"]["b"]
    assert r3["underpowered"] and not c["significant"] and not c["prereg_win"]


# ------------------------------------------------------------------ A7
def test_A7_aggregate_per_seed():
    from tune3.experiments.stats import aggregate_per_seed
    rbf = {"0": {"seeds": [0, 1], "by_metric": {"m": {"cvar_test": [1.0, 3.0]}}},
           "1": {"seeds": [0, 1], "by_metric": {"m": {"cvar_test": [2.0, 5.0]}}}}
    assert aggregate_per_seed(rbf, "m") == [1.5, 4.0]


# ------------------------------------------------------------------ A8
def test_A8_results_io_roundtrip(tmp_path):
    from tune3.experiments.results_io import save_result, load_results
    p = save_result({"x": [1, 2, 3], "arr": np.arange(3)}, experiment="exp_teste",
                    tag="pytest", args={"a": 1}, out_dir=str(tmp_path))
    docs = list(load_results("exp_teste", str(tmp_path)))
    assert len(docs) == 1 and docs[0][0] == p
    meta, payload = docs[0][1]["meta"], docs[0][1]["payload"]
    assert meta["experiment"] == "exp_teste" and meta["tag"] == "pytest" and meta["args"] == {"a": 1}
    assert "git" in meta and "env" in meta and "python" in meta["env"]
    assert payload["arr"] == [0, 1, 2]


# ------------------------------------------------------------------ A9
def test_A9_config_placebo_valida():
    from tune3.integration.runner import Tune3RunConfig
    Tune3RunConfig(objective2="random")
    with pytest.raises(ValueError):
        Tune3RunConfig(objective2="ruido")


def test_A9_metodos_validados():
    from tune3.experiments.protocol import ProtocolConfig, ALL_METHODS
    assert {"bo_mono_cvar", "tune3_placebo_bestcvar", "tune3_noddkf_bestcvar"} <= set(ALL_METHODS)
    with pytest.raises(ValueError):
        ProtocolConfig(methods=["inexistente"])


@pytest.mark.slow
def test_A9_b4_e_placebo_ponta_a_ponta():
    """B4 (BO mono-objetivo) e Tune3-placebo rodam com orcamento identico ao Tune3."""
    import warnings; warnings.filterwarnings("ignore")
    from tune3.baselines.bo_mono import MonoObjectiveBO
    from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
    from tune3.integration.runner import run_tune3, Tune3RunConfig
    from tune3.integration.trial import TrialConfig
    from tune3.macro.bo_loop import MacroConfig
    Xtr, ytr = _synthetic(500, 0); Xv, yv = _synthetic(200, 1)
    space = {"log_lr": (-3.0, -1.0), "log_wd": (-6.0, -3.0), "dropout": (0.0, 0.4),
             "hidden_dim": (32.0, 64.0), "n_layers": (1.0, 2.0)}
    tc = PlainTrialConfig(max_epochs=3, device="cpu")
    b4 = MonoObjectiveBO(space, evaluate_fn=lambda hp: plain_trial(hp, (Xtr, ytr, Xv, yv), tc),
                         n_init=3, n_iter=2, mc_samples=16, num_restarts=2, raw_samples=32, seed=0).run()
    assert b4["n_evaluations"] == 5 and np.isfinite(b4["best"]["cvar"])
    out = run_tune3((Xtr, ytr, Xv, yv),
                    Tune3RunConfig(macro=MacroConfig(n_init=3, n_iter=2, mc_samples=16, num_restarts=2,
                                                     raw_samples=32, seed=0),
                                   trial=TrialConfig(max_epochs=3, device="cpu"),
                                   search_space=space, objective2="random"))
    assert out["objective2"] == "random" and out["pareto"]["n_evaluations"] == 5
    assert len(out["trials"]) == 5 and "n_eos_triggers" in out["trials"][0]


# ---------------------------------------------------------------- A10 (rev. 2)
def test_A10_curvature_every_atravessa_ate_o_trial(monkeypatch):
    """A10: a cadencia de curvatura escolhida no script precisa chegar ao TrialConfig.

    Sem isso, `--curvature-every 1` seria aceito na linha de comando e silenciosamente
    ignorado -- e a medicao de custo (decisao D-C) compararia duas execucoes identicas.
    """
    from tune3.experiments import protocol as P

    visto = {}

    def fake_run_tune3(train_val, run_cfg):
        visto["every"] = run_cfg.trial.curvature_every
        return {"ok": True}

    monkeypatch.setattr(P, "run_tune3", fake_run_tune3)
    P._tune3_variant(None, P.ProtocolConfig(curvature_every=1), seed=0)
    assert visto["every"] == 1, "ProtocolConfig.curvature_every nao chegou ao TrialConfig"
    P._tune3_variant(None, P.ProtocolConfig(), seed=0)
    assert visto["every"] == 5, "o padrao do projeto (5) mudou sem aviso"


# ---------------------------------------------------------------- A11 (rev. 3)
def test_A11_telemetria_r2_chega_ao_runner():
    """A11: r2_max/n_ddkf_observations precisam SAIR do trial E CHEGAR ao JSON.

    O runner copia uma lista fixa de chaves; uma chave nova no trial que nao seja
    acrescentada ali some silenciosamente -- foi o que aconteceu na primeira versao.
    Sem r2_max nao da' para distinguir "buffer curto" de "gate fechado" (pergunta do E2).
    """
    import warnings; warnings.filterwarnings("ignore")
    from tune3.integration.runner import run_tune3, Tune3RunConfig
    from tune3.integration.trial import TrialConfig, run_trial
    from tune3.macro.bo_loop import MacroConfig
    Xtr, ytr = _synthetic(400, 0); Xv, yv = _synthetic(200, 1)

    r = run_trial({"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.1,
                   "hidden_dim": 32, "n_layers": 1}, (Xtr, ytr, Xv, yv),
                  TrialConfig(max_epochs=4, device="cpu"))
    for k in ("r2_max", "r2_mean", "n_ddkf_observations"):
        assert k in r, f"trial nao expoe {k}"
    assert r["n_ddkf_observations"] == r["n_epochs_run"], "1 observacao por epoca"

    out = run_tune3((Xtr, ytr, Xv, yv),
                    Tune3RunConfig(macro=MacroConfig(n_init=2, n_iter=1, mc_samples=8,
                                                     num_restarts=2, raw_samples=16, seed=0),
                                   trial=TrialConfig(max_epochs=4, device="cpu"),
                                   search_space={"log_lr": (-3.0, -2.0), "log_wd": (-5.0, -4.0),
                                                 "dropout": (0.0, 0.2), "hidden_dim": (32.0, 64.0),
                                                 "n_layers": (1.0, 2.0)}))
    t0 = out["trials"][0]
    for k in ("r2_max", "r2_mean", "n_ddkf_observations"):
        assert k in t0 and t0[k] is not None, f"runner nao propaga {k} para a telemetria"


# ---------------------------------------------------------------- A12 (rev. 3)
def test_A12_obs_vector_muda_o_que_o_ddkf_observa():
    """A12: o braco D1 precisa MESMO trocar a 2a componente do vetor de observacao.

    Uma flag que e' aceita mas nao muda nada produziria um "resultado" de ablacao
    falso -- exatamente o tipo de erro que o placebo do E1 existe para pegar.
    Aqui capturamos os z_t entregues ao filtro nos dois modos.
    """
    import warnings; warnings.filterwarnings("ignore")
    from tune3.integration import trial as T
    from tune3.micro import DDKFController

    Xtr, ytr = _synthetic(400, 0); Xv, yv = _synthetic(200, 1)
    hp = {"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.0,
          "hidden_dim": 32, "n_layers": 1}

    def observados(obs_vector):
        capturado = []
        original = DDKFController.update

        def espiao(self, z, *a, **kw):
            capturado.append(np.asarray(z, dtype=float).copy())
            return original(self, z, *a, **kw)

        T.DDKFController.update = espiao
        try:
            r = T.run_trial(hp, (Xtr, ytr, Xv, yv),
                            T.TrialConfig(max_epochs=4, device="cpu", curvature_every=1,
                                          obs_vector=obs_vector))
        finally:
            T.DDKFController.update = original
        return np.array(capturado), r

    z_loss, r_loss = observados("loss")
    z_curv, r_curv = observados("curvature")

    assert z_loss.shape == z_curv.shape and z_loss.shape[1] == 3
    # 1a e 3a componentes sao as mesmas (train_loss, log GSNR); a 2a muda
    assert not np.allclose(z_loss[:, 1], z_curv[:, 1]), "obs_vector nao trocou a observacao"
    # e a 2a componente do braco 'curvature' e' log(1+Tr(H^2)) >= 0
    assert np.all(z_curv[:, 1] >= 0.0)
    assert r_loss["obs_vector"] == "loss" and r_curv["obs_vector"] == "curvature"

    with pytest.raises(ValueError):
        T.TrialConfig(obs_vector="inexistente")


# ---------------------------------------------------------------- A13 (rev. 4)
def _nsl_fake(path, rows):
    """Escreve um arquivo no formato NSL-KDD (43 colunas, sem cabecalho)."""
    lines = []
    for proto, serv, flag, sbytes, label, const in rows:
        f = [0, proto, serv, flag, sbytes, 0] + [0] * 13 + [const] + [0] * 21
        assert len(f) == 41
        lines.append(",".join(str(v) for v in f + [label, 21]))
    path.write_text("\n".join(lines) + "\n")


def test_A13_nslkdd_loader_sem_vazamento(tmp_path):
    """A13: loader NSL-KDD -- formato, rotulo binario, split oficial zero-day, sem vazamento.

    Garante: (i) a coluna 'dificuldade' (metadado do dataset) NAO vira atributo;
    (ii) coluna constante descartada; (iii) one-hot e z-score ajustados SO' no treino --
    categoria que so' existe no teste vira vetor de zeros; (iv) test_novel_mask marca
    exatamente os tipos de ataque ausentes do treino.
    """
    from tune3.data.nslkdd import NSLKDDLoader, NSLKDDConfig
    from tune3.data.registry import load_dataset
    rng = np.random.default_rng(0)
    tr = [("tcp" if i % 2 else "udp", "http" if i % 3 else "ftp", "SF",
           int(rng.integers(0, 10**6)), "normal" if i % 2 else "neptune", 0) for i in range(200)]
    te = [("icmp", "http", "SF", 5, "normal", 0), ("tcp", "ftp", "SF", 7, "neptune", 0),
          ("tcp", "http", "SF", 9, "apache2", 0), ("udp", "ftp", "SF", 3, "mailbomb", 0)]
    _nsl_fake(tmp_path / "KDDTrain+.txt", tr); _nsl_fake(tmp_path / "KDDTest+.txt", te)

    m = NSLKDDLoader(NSLKDDConfig(data_dir=str(tmp_path), random_state=0)).load_splits_with_meta()
    names = m["feature_names"]
    assert "difficulty" not in names and "label" not in names
    assert "num_outbound_cmds" in m["dropped_constant"] and "num_outbound_cmds" not in names
    assert m["X_test"].shape[0] == 4 and m["X_train"].shape[1] == m["X_test"].shape[1]
    assert list(m["y_test"]) == [0, 1, 1, 1]                        # normal->0, ataque->1
    assert list(m["test_novel_mask"]) == [False, False, True, True]  # apache2, mailbomb: novos
    # 'icmp' nao existe no treino: todas as colunas protocol_type=* zeradas na 1a linha de teste
    proto_cols = [i for i, n in enumerate(names) if n.startswith("protocol_type=")]
    assert "protocol_type=icmp" not in names and m["X_test"][0, proto_cols].sum() == 0
    # z-score ajustado no treino: media ~0 no treino
    j = names.index("src_bytes")
    assert abs(m["X_train"][:, j].mean()) < 1e-5 and abs(m["X_train"][:, j].std() - 1) < 1e-3
    # registry devolve a mesma 6-tupla
    t = load_dataset("nslkdd", str(tmp_path), 0)
    assert len(t) == 6 and np.allclose(t[2], m["X_test"])


# ---------------------------------------------------------------- A14 (rev. 4)
def test_A14_partial_spearman_recupera_sinal_e_supressao():
    """A14: a estatistica do E0 precisa acertar o SINAL -- e' ele que decide se H_G e' sustentada.

    Dados com efeito direto conhecido de x sobre y, com x tambem correlacionado com o controle.
    Inclui os dois casos em que a correlacao BRUTA engana: redundancia (bruta > 0, parcial ~0)
    e supressao (bruta ~0, parcial < 0).
    """
    from tune3.experiments.stats import partial_spearman
    rng = np.random.default_rng(1); n = 600

    def gera(efeito, acopl=0.6):
        x = rng.normal(size=n); c = acopl * x + rng.normal(size=n)
        return x, c, 0.9 * c + efeito * x + rng.normal(size=n)

    x, c, y = gera(+0.8); assert partial_spearman(y, x, [c]) > 0.3
    x, c, y = gera(-0.8); assert partial_spearman(y, x, [c]) < -0.3
    x, c, y = gera(0.0)                                     # redundancia
    assert np.corrcoef(x, y)[0, 1] > 0.2 and abs(partial_spearman(y, x, [c])) < 0.12
    x, c, y = gera(-0.6, acopl=0.8)                         # supressao
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.2 and partial_spearman(y, x, [c]) < -0.2


def test_A14b_plain_trial_devolve_perdas_de_teste_coerentes():
    """As perdas por amostra do teste reproduzem o cvar_test (base do CVaR em ataques novos)."""
    from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
    from tune3.core.objectives import cvar
    Xtr, ytr = _synthetic(300, 0); Xv, yv = _synthetic(150, 1); Xte, yte = _synthetic(120, 2)
    r = plain_trial({"learning_rate": 1e-2, "weight_decay": 1e-4, "dropout": 0.0, "hidden_dim": 16,
                     "n_layers": 1}, (Xtr, ytr, Xv, yv), PlainTrialConfig(max_epochs=3, device="cpu"),
                    test_data=(Xte, yte), return_test_losses=True)
    assert r["test_losses"].shape == (120,)
    assert abs(cvar(r["test_losses"], 0.95) - r["cvar_test"]) < 1e-9


# ---------------------------------------------------------------------------
# A15/A16 (24/09/2026): holdout de validacao e parada por perda de treino
# ---------------------------------------------------------------------------
def test_A15_holdout_nao_escolhe_nada_e_reproduz_cvar():
    """O holdout e' so' medido: trocar o holdout nao muda a epoca escolhida nem o CVaR de
    validacao/teste, e com holdout = validacao o cvar_holdout coincide com o cvar."""
    from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
    Xtr, ytr = _synthetic(300, 0); Xv, yv = _synthetic(150, 1); Xh, yh = _synthetic(150, 3)
    cfg = PlainTrialConfig(max_epochs=6, device="cpu", batch_size=64)
    a = plain_trial(HP, (Xtr, ytr, Xv, yv), cfg, holdout_data=(Xh, yh))
    b = plain_trial(HP, (Xtr, ytr, Xv, yv), cfg, holdout_data=(Xh[::-1].copy(), 1 - yh[::-1]))
    c = plain_trial(HP, (Xtr, ytr, Xv, yv), cfg)
    assert a["best_epoch"] == b["best_epoch"] == c["best_epoch"]
    assert a["cvar"] == b["cvar"] == c["cvar"] and a["curvature"] == c["curvature"]
    assert "cvar_holdout" not in c and a["cvar_holdout"] != b["cvar_holdout"]
    d = plain_trial(HP, (Xtr, ytr, Xv, yv), cfg, holdout_data=(Xv, yv))
    assert abs(d["cvar_holdout"] - d["cvar"]) < 1e-9
    assert c["stop_mode"] == "val_early_stopping" and "reached_train_loss" not in c


def test_A16_parada_por_perda_de_treino():
    """Com stop_train_loss: para na 1a epoca com perda de treino <= T, o modelo final e' o dessa
    epoca (a validacao nao escolhe), e um T inatingivel roda ate' o teto com reached=False."""
    from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
    Xtr, ytr = _synthetic(400, 0); Xv, yv = _synthetic(150, 1)
    hp = {"learning_rate": 0.05, "weight_decay": 0.0, "dropout": 0.0, "hidden_dim": 32, "n_layers": 2}
    base = PlainTrialConfig(max_epochs=4, device="cpu", batch_size=64)
    r0 = plain_trial(hp, (Xtr, ytr, Xv, yv), base)
    T = r0["train_loss_final"] * 1.05          # atingivel em poucas epocas
    r = plain_trial(hp, (Xtr, ytr, Xv, yv),
                    PlainTrialConfig(max_epochs=200, device="cpu", batch_size=64, stop_train_loss=T))
    assert r["stop_mode"] == "train_loss" and r["reached_train_loss"]
    assert r["train_loss_final"] <= T + 1e-12
    assert r["best_epoch"] == r["epochs_run"] - 1          # modelo final = epoca de parada
    # a validacao nao escolhe: trocar a validacao nao muda epoca de parada nem curvatura
    r2 = plain_trial(hp, (Xtr, ytr, Xv[::-1].copy(), 1 - yv[::-1]),
                     PlainTrialConfig(max_epochs=200, device="cpu", batch_size=64, stop_train_loss=T))
    assert r2["epochs_run"] == r["epochs_run"] and r2["curvature"] == r["curvature"]
    ru = plain_trial(hp, (Xtr, ytr, Xv, yv),
                     PlainTrialConfig(max_epochs=5, device="cpu", batch_size=64, stop_train_loss=1e-12))
    assert not ru["reached_train_loss"] and ru["epochs_run"] == 5


def test_A16b_split_half_estratificado_e_disjunto():
    """split_half do E0: metades disjuntas, cobrem tudo, proporcao de classes preservada."""
    import importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "e0_generalization.py"
    spec = importlib.util.spec_from_file_location("e0g", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng = np.random.default_rng(0); X = rng.normal(size=(1001, 3)); y = (rng.random(1001) < 0.3).astype(int)
    Xa, ya, Xb, yb = m.split_half(X, y, 7)
    assert len(ya) + len(yb) == 1001
    sa = {tuple(r) for r in Xa}; sb = {tuple(r) for r in Xb}
    assert not (sa & sb)
    assert abs(ya.mean() - yb.mean()) < 0.01


def test_A17_val_losses_reproduzem_cvar_de_validacao():
    """return_val_losses: as perdas por amostra da validacao reproduzem o 'cvar' devolvido (base
    da checagem de artefato com n pareado do E0), nos dois modos de parada."""
    from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
    from tune3.core.objectives import cvar
    Xtr, ytr = _synthetic(300, 0); Xv, yv = _synthetic(150, 1)
    for cfg in (PlainTrialConfig(max_epochs=4, device="cpu", batch_size=64),
                PlainTrialConfig(max_epochs=6, device="cpu", batch_size=64, stop_train_loss=0.5)):
        r = plain_trial(HP, (Xtr, ytr, Xv, yv), cfg, return_val_losses=True)
        assert r["val_losses"].shape == (150,)
        assert abs(cvar(r["val_losses"], 0.95) - r["cvar"]) < 1e-9


def test_A18_kmeans_proprio_sem_sklearn_deterministico():
    """k-means numpy do S2/E0-cluster: recupera blocos bem separados, e' deterministico dada a
    semente, rotula por tamanho decrescente e NAO importa sklearn.cluster (bloqueado pelo
    Controle Inteligente de Aplicativos do Windows)."""
    import subprocess, sys
    from tune3.data.shift import kmeans
    rng = np.random.default_rng(0)
    centros = np.array([[0, 0], [10, 0], [0, 10]], float); tam = [300, 200, 100]
    X = np.vstack([c + rng.normal(scale=0.5, size=(t, 2)) for c, t in zip(centros, tam)])
    verdade = np.repeat([0, 1, 2], tam)
    lab = kmeans(X, 3, seed=1)
    assert np.array_equal(lab, kmeans(X, 3, seed=1))
    assert np.array_equal(lab, verdade)                 # rotulos ordenados por tamanho
    code = ("import sys; import numpy as np; from tune3.data.shift import cluster_malware; "
            "X=np.random.default_rng(0).random((200,5)); y=(X[:,0]>.5).astype(int); "
            "cluster_malware(X,y,3); print('sklearn.cluster' in sys.modules)")
    import pathlib
    raiz = pathlib.Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(raiz))
    assert out.stdout.strip() == "False"
