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


# ---------------------------------------------------------------------------
# A19-A23 (out/2026): etapa D2 -- CNN, sharpness adaptativa, CIFAR, Kendall granulado
# ---------------------------------------------------------------------------
def _fake_cifar(root, n_per_batch=200, n101=100, seed=0):
    """Arquivos no formato oficial (pickle do CIFAR-10, .npy do CIFAR-10.1), com dados falsos."""
    import os, pickle
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "cifar10", "cifar-10-batches-py"); os.makedirs(base)
    for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
        y = np.arange(n_per_batch) % 10
        X = rng.integers(0, 256, size=(n_per_batch, 3072), dtype=np.uint8)
        with open(os.path.join(base, name), "wb") as f:
            pickle.dump({b"data": X, b"labels": list(map(int, y))}, f)
    d1 = os.path.join(root, "cifar10_1"); os.makedirs(d1)
    np.save(os.path.join(d1, "cifar10.1_v6_data.npy"), rng.integers(0, 256, size=(n101, 32, 32, 3), dtype=np.uint8))
    np.save(os.path.join(d1, "cifar10.1_v6_labels.npy"), (np.arange(n101) % 10).astype(np.int64))
    return os.path.join(root, "cifar10"), d1


def test_A19_small_cnn_formas_e_fabrica():
    from tune3.models import build_model, SmallCNN
    x = torch.randn(4, 3, 32, 32)
    for depth in (2, 3, 4):
        m = build_model(3, 10, arch="cnn", hparams={"hidden_dim": 8, "n_layers": depth, "dropout": 0.25})
        assert isinstance(m, SmallCNN) and m(x).shape == (4, 10)
    n = lambda d, w: sum(p.numel() for p in SmallCNN(10, width=w, depth=d).parameters())
    assert n(2, 8) < n(3, 8) and n(3, 8) < n(3, 16)


def test_A20_sharpness_adaptativa_invariante_a_reescala_e_tr_h2_nao():
    """Rede ReLU: multiplicar a 1a camada por a e dividir a 2a por a da' a MESMA funcao. A sharpness
    adaptativa nao muda; a Tr(H^2) muda (Dinh et al., 2017). E os pesos voltam ao original."""
    from tune3.curvature import adaptive_sharpness, HutchinsonEstimator, HutchinsonConfig
    torch.manual_seed(0)
    X = torch.randn(256, 6); y = (X[:, 0] > 0).long()
    net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for _ in range(200):
        opt.zero_grad(); nn.functional.cross_entropy(net(X), y).backward(); opt.step()
    import copy
    net2 = copy.deepcopy(net); a = 4.0
    with torch.no_grad():
        net2[0].weight.mul_(a); net2[0].bias.mul_(a); net2[2].weight.div_(a)
    assert torch.allclose(net(X), net2(X), atol=1e-5)
    lf = lambda m: nn.functional.cross_entropy(m(X), y)
    w_before = [p.detach().clone() for p in net.parameters()]
    s1 = adaptive_sharpness(net, lf, rho=0.1, steps=10); s2 = adaptive_sharpness(net2, lf, rho=0.1, steps=10)
    assert all(torch.equal(p, w) for p, w in zip(net.parameters(), w_before))
    assert s1 > 0 and abs(s1 - s2) / s1 < 1e-3
    tr = lambda m: HutchinsonEstimator(HutchinsonConfig(num_probes=50)).estimate(m, lf(m))
    torch.manual_seed(1); t1 = tr(net); torch.manual_seed(1); t2 = tr(net2)
    assert abs(t1 - t2) / t1 > 0.5


def test_A21_cifar_loader_formato_divisao_e_normalizacao(tmp_path):
    from tune3.data.cifar import CIFARLoader, CIFARConfig, load_cifar101_raw
    d10, d101 = _fake_cifar(str(tmp_path))
    m = CIFARLoader(CIFARConfig(data_dir=d10, cifar101_dir=d101, n_train=300, n_val=200)).load_splits_with_meta()
    assert m["X_train"].shape == (300, 3, 32, 32) and m["X_train"].dtype == np.float32
    assert m["X_val"].shape == (200, 3, 32, 32)
    assert len(np.intersect1d(m["train_idx"], m["val_idx"])) == 0          # validacao disjunta do treino
    assert np.bincount(m["y_train"]).tolist() == [30] * 10                   # estratificado
    assert abs(float(m["X_train"].mean(axis=(0, 2, 3)).max())) < 1e-4       # normalizado pelo treino
    assert m["X_test"].shape[0] == 200 + 100 and m["test_novel_mask"].sum() == 100
    assert not m["test_novel_mask"][:200].any() and m["test_novel_mask"][200:].all()
    raw = np.load(f"{d101}/cifar10.1_v6_data.npy"); X1, _ = load_cifar101_raw(d101)
    assert X1[5, 2, 7, 9] == raw[5, 7, 9, 2]                                 # NHWC -> NCHW correto


def test_A22_kendall_granulado():
    from tune3.experiments.generalization_measures import granulated_kendall, kendall_tau
    import itertools
    rng = np.random.default_rng(0)
    rows = []
    for a, b, c in itertools.product([1, 2, 3], [1, 2, 3], [1, 2, 3]):
        g = 2.0 * a - 1.0 * b + 0.0 * c + rng.normal(scale=0.01)
        rows.append({"hparams": {"a": a, "b": b, "c": c}, "m_bom": g, "m_a": float(a), "g": g})
    gk = granulated_kendall(rows, "m_bom", "g", ["a", "b", "c"])
    assert gk["per_factor"]["a"]["psi"] == 1.0 and gk["per_factor"]["b"]["psi"] == 1.0
    # uma medida que so' acompanha o fator a acerta psi_a, mas nao ganha nada em b (nao varia la')
    gka = granulated_kendall(rows, "m_a", "g", ["a", "b", "c"])
    assert gka["per_factor"]["a"]["psi"] == 1.0 and np.isnan(gka["per_factor"]["b"]["psi"])
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_A23_plain_trial_com_imagens_erros_e_medidas_extras():
    from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 3, 32, 32)).astype(np.float32); y = (np.arange(240) % 3).astype(np.int64)
    X[y == 1, 0] += 1.0; X[y == 2, 1] += 1.0
    seen = {}

    def extra(model, xb, yb, crit):
        seen["n"] = xb.shape[0]; return {"medida_x": 1.5}
    r = plain_trial({"learning_rate": 0.01, "weight_decay": 0.0, "dropout": 0.0, "hidden_dim": 4, "n_layers": 2},
                    (X[:160], y[:160], X[160:200], y[160:200]),
                    PlainTrialConfig(max_epochs=3, batch_size=32, arch="cnn", device="cpu", curvature_batch=64,
                                     eval_chunk=16, use_class_weight=False),
                    test_data=(X[200:], y[200:]), return_errors=True, extra_measures=extra)
    assert seen["n"] == 64 and r["medida_x"] == 1.5
    assert 0.0 <= r["train_error"] <= 1.0 and 0.0 <= r["val_error"] <= 1.0
    assert r["test_correct"].shape == (40,) and abs(r["test_error"] - (1 - r["test_correct"].mean())) < 1e-12


def test_A24_regra_A_nao_aceita_medida_que_so_acompanha_a_profundidade():
    """Uma medida que so' acompanha a profundidade (que por sua vez move o gap) tem tau alto e
    Psi > 0, mas NAO passa na regra (A) da D2: so' 1 dos 5 fatores com psi > 0. Uma medida que
    acompanha o gap em todos os fatores passa."""
    from tune3.experiments.generalization_measures import granulated_kendall, kendall_tau, replication_positive
    import itertools
    rng = np.random.default_rng(0); F = ["d", "w", "lr", "wd", "do"]; rows = []
    for v in itertools.product(*[[1, 2, 3]] * 5):
        hp = dict(zip(F, v))
        gap = 0.5 * hp["d"] + 0.2 * hp["w"] - 0.3 * hp["lr"] + 0.1 * hp["wd"] + 0.15 * hp["do"]
        rows.append({"hparams": hp, "gap": gap, "prof": hp["d"] + rng.normal(scale=0.05),
                     "boa": gap + rng.normal(scale=0.02)})
    g = np.array([r["gap"] for r in rows])
    for m, esperado in (("prof", False), ("boa", True)):
        x = np.array([r[m] for r in rows]); tau = kendall_tau(x, g)
        gk = granulated_kendall(rows, m, "gap", F)
        assert tau > 0.3 and gk["Psi"] > 0
        assert replication_positive([tau - 0.05, tau + 0.05], gk) is esperado


# ---------------------------------------------------------------------------
# A25-A27 (out/2026): fase F -- fisher_combine em stats, cluster_seed, escassez de validacao
# ---------------------------------------------------------------------------
def test_A25_fisher_combine_valores_conhecidos():
    from tune3.experiments.stats import fisher_combine
    c = fisher_combine([0.3, 0.3, 0.3], [100, 100, 100], 1)
    assert abs(c["r"] - 0.3) < 1e-9 and c["I2"] == 0.0 and abs(c["r_re"] - 0.3) < 1e-9
    assert c["ci95"][0] < 0.3 < c["ci95"][1]
    h = fisher_combine([0.6, -0.4, 0.5, -0.3], [200, 200, 200, 200], 1)
    assert h["I2"] > 0.5 and (h["ci95_re"][1] - h["ci95_re"][0]) > (h["ci95"][1] - h["ci95"][0])
    assert fisher_combine([float("nan")], [50], 1)["k"] == 0


def test_A26_cluster_seed_fixa_os_clusters_e_muda_so_a_divisao():
    """F2: com cluster_seed fixo, o malware de teste do fold e' o MESMO conjunto para qualquer seed de
    divisao; so' treino/validacao (e os benignos do teste) mudam. Sem cluster_seed, seed muda tudo."""
    from tune3.data.shift import cluster_holdout_split
    X, y = _synthetic(2000, 0)
    def mal_test(seed, cs):
        _, _, _, _, Xte, yte = cluster_holdout_split(X, y, 4, 1, seed=seed, cluster_seed=cs)
        return {tuple(r) for r in Xte[yte == 1]}
    def train_set(seed, cs):
        Xtr, *_ = cluster_holdout_split(X, y, 4, 1, seed=seed, cluster_seed=cs)
        return {tuple(r) for r in Xtr}
    assert mal_test(1, 0) == mal_test(2, 0) == mal_test(0, 0)
    assert train_set(1, 0) != train_set(2, 0)
    a, b = cluster_holdout_split(X, y, 4, 1, seed=0), cluster_holdout_split(X, y, 4, 1, seed=0, cluster_seed=0)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))        # padrao None == seed: D1 reproduzivel


def test_A27_escassez_de_validacao_recupera_estrutura_conhecida():
    """scan(): num cenario em que a medida acompanha a QUALIDADE do modelo (redundante com a validacao
    completa), a parcial dada a validacao escassa e' positiva e cai com n_val -> veredito SUSTENTA.
    Com medida independente de tudo -> NAO SUSTENTA. Valida a logica antes dos dados reais."""
    from tune3.experiments.val_scarcity import scan
    rng = np.random.default_rng(0); M, N = 150, 2000
    q = rng.normal(size=M)                                    # qualidade latente do modelo
    V = np.exp(0.4 * q[:, None] + rng.normal(scale=0.9, size=(M, N)))   # perdas por amostra (cauda pesada)
    target = q + 0.3 * rng.normal(size=M)                      # alvo: qualidade + ruido
    x_bom = q + 0.5 * rng.normal(size=M)                       # "curvatura" que acompanha a qualidade
    x_nulo = rng.normal(size=M)
    r = scan(V, target, x_bom, sizes=(50, 100, 200, 500), stat="cvar", R=30, B=150, seed=0)
    P = [row["P"] for row in r["per_size"]]
    assert P[0] > P[-1] + 0.2 and r["verdict"].startswith("ESCASSEZ SUSTENTA")
    r0 = scan(V, target, x_nulo, sizes=(50, 100, 200, 500), stat="cvar", R=30, B=150, seed=0)
    assert r0["verdict"].startswith("NAO SUSTENTA")


# ---------------------------------------------------------------- F3 (out/2026): KronoDroid temporal
def _krono_fake(dir_, seed=0, leak_col=False):
    """Dois ZIPs com a ESTRUTURA do KronoDroid real (separadores nr_syscalls / nr_permissions, metadados no
    fim, datas M/D/AAAA), em escala pequena. Permissoes P_A..P_D; 'P_RARE' com suporte < 50; 'P_FUTURE'
    so' aparece depois do corte; chamada 'sys_rare' com 3 linhas nao-nulas; 'sys_const' constante."""
    import zipfile, pandas as pd
    rng = np.random.default_rng(seed); rows = []
    for lab, n in ((1, 900), (0, 700)):
        for i in range(n):
            year = int(rng.choice([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
                                  p=[.14, .14, .12, .1, .1, .08, .08, .08, .06, .05, .05]))
            rows.append({"Package": f"pkg{lab}{i}", "Malware": lab,
                         "read": int(rng.poisson(50 + 40 * lab)), "write": int(rng.poisson(20)),
                         "sys_rare": int(i < 3 and lab == 1), "sys_const": 0, "ptrace": int(rng.poisson(0.02)) * (1 + 500 * (year > 2014)),
                         "nr_syscalls": int(rng.poisson(90)),
                         "P_A": int(rng.random() < 0.3 + 0.4 * lab), "P_B": int(rng.random() < 0.5), "P_C": int(rng.random() < 0.2),
                         "P_RARE": int(i < 10), "P_FUTURE": int(year > 2014 and rng.random() < 0.5), "P_D": int(rng.random() < 0.1),
                         "nr_permissions": 3, "normal": 2, "dangerous": 1, "signature": 0, "custom_yes": 0, "nr_custom": 0, "total_perm": 3,
                         "sha256": f"{seed:02d}{lab}{i:06d}", "CFileSize": int(rng.integers(10**4, 10**7)), "UFileSize": int(rng.integers(10**4, 10**7)),
                         "EarliestModDate": f"1/1/{year}", "HighestModDate": f"{int(rng.integers(1, 13))}/{int(rng.integers(1, 28))}/{year}",
                         "FilesInsideAPK": 10, "TimesSubmitted": 3, "NrContactedIps": 0, "Activities": 2.0, "NrIntServices": 0.0,
                         "NrIntServicesActions": 0.0, "NrIntActivities": 1.0, "NrIntActivitiesActions": 1.0, "NrIntReceivers": 0.0,
                         "NrIntReceiversActions": 0.0, "TotalIntentFilters": 1.0, "NrServices": 0.0, "Scanners": 60,
                         "Detection_Ratio": 0.5 * lab, "MalFamily": "Fam" if lab else float("nan")})
    df = pd.DataFrame(rows)
    df.loc[0, "HighestModDate"] = "1/1/1980"                     # placeholder: deve ser descartado
    for lab, name in ((1, "real_malware_v1"), (0, "real_legitimate_v1")):
        part = df[df["Malware"] == lab]
        if leak_col and lab == 0:
            part = part.assign(ONLY_BENIGN=1)
        with zipfile.ZipFile(dir_ / f"{name}.zip", "w") as z:
            z.writestr(f"{name}.csv", part.to_csv(index=False))
    return df


def test_A28_kronodroid_divisao_temporal_sem_vazamento(tmp_path):
    """F3: (i) treino/validacao so' ate' o corte, teste so' depois; mascara 'novel' = futuro distante;
    (ii) fracao de malware do teste casada com a do periodo de treino, no proximo e no distante;
    (iii) identificadores, datas e metadados do VirusTotal NAO viram atributo; permissoes ficam 0/1;
    (iv) colunas constantes ou com suporte < min_support descartadas (inclui permissao que so' existe
    no futuro); z-score ajustado no treino e limitado a +-clip_z; (v) data-placeholder descartada;
    (vi) deterministico; proporcao natural mantem todas as linhas do futuro."""
    from tune3.data.kronodroid import KronoDroidLoader, KronoDroidConfig, EXCLUDE
    _krono_fake(tmp_path)
    cfg = KronoDroidConfig(data_dir=str(tmp_path), cut_year=2014, far_year=2018, max_train=600, n_val=200,
                           min_support=50, clip_z=10.0, random_state=0)
    m = KronoDroidLoader(cfg).load_splits_with_meta(); meta = m["meta"]; names = m["feature_names"]
    assert meta["n_rows_bad_year"] == 1 and meta["n_train"] == 600 and meta["n_val"] == 200
    yr = m["test_year"]; assert yr.min() == 2015 and list(m["test_novel_mask"]) == list(yr >= 2018)
    p = meta["p_mal_train_period"]
    assert abs(meta["frac_mal_test_near"] - p) < 0.02 and abs(meta["frac_mal_test_far"] - p) < 0.02
    assert abs(m["y_train"].mean() - p) < 0.02 and abs(m["y_val"].mean() - p) < 0.02
    assert not (set(names) & set(EXCLUDE)) and "ONLY_BENIGN" not in names
    for c in ("sys_const", "sys_rare", "P_RARE", "P_FUTURE"):
        assert c in meta["dropped_constant"] and c not in names
    assert "read" in names and "P_A" in names and "CFileSize" in names
    perm = [i for i, n in enumerate(names) if n in ("P_A", "P_B", "P_C", "P_D")]
    assert set(np.unique(m["X_train"][:, perm])) <= {0.0, 1.0} and set(np.unique(m["X_test"][:, perm])) <= {0.0, 1.0}
    j = names.index("read")
    assert abs(m["X_train"][:, j].mean()) < 1e-4 and abs(m["X_train"][:, j].std() - 1) < 1e-3
    assert np.abs(m["X_test"]).max() <= 10.0 + 1e-6 and not np.isnan(m["X_test"]).any()
    m2 = KronoDroidLoader(cfg).load_splits_with_meta()
    assert np.array_equal(m["X_test"], m2["X_test"]) and np.array_equal(m["X_train"], m2["X_train"])
    nat = KronoDroidLoader(KronoDroidConfig(data_dir=str(tmp_path), max_train=600, n_val=200, match_ratio=False,
                                            min_support=50)).load_splits_with_meta()
    assert nat["meta"]["n_test"] > meta["n_test"] and nat["meta"]["n_test"] == int((_years_after(tmp_path) > 2014).sum())


def _years_after(dir_):
    """Anos validos (apos descarte do placeholder) dos dois ZIPs, para conferir a proporcao natural."""
    import zipfile, io, pandas as pd
    ys = []
    for name in ("real_malware_v1", "real_legitimate_v1"):
        with zipfile.ZipFile(dir_ / f"{name}.zip") as z, z.open(f"{name}.csv") as f:
            d = pd.read_csv(io.BytesIO(f.read()))
        y = pd.to_datetime(d["HighestModDate"], format="%m/%d/%Y", errors="coerce").dt.year
        ys.append(y[y.between(2008, 2020)].values)
    return np.concatenate(ys)


def test_A29_kronodroid_coluna_so_de_um_lado_e_vazamento_e_cai(tmp_path):
    """Uma coluna presente so' no arquivo de benignos (ou so' no de malware) identificaria o rotulo:
    o loader usa a INTERSECAO das colunas e registra a descartada."""
    from tune3.data.kronodroid import KronoDroidLoader, KronoDroidConfig
    _krono_fake(tmp_path, leak_col=True)
    m = KronoDroidLoader(KronoDroidConfig(data_dir=str(tmp_path), max_train=600, n_val=200)).load_splits_with_meta()
    assert m["meta"]["dropped_columns"] == ["ONLY_BENIGN"] and "ONLY_BENIGN" not in m["feature_names"]


def test_A30_e0_carrega_kronodroid_e_rotula_alvos(tmp_path):
    """O E0 aceita --datasets kronodroid, devolve a 7-tupla coerente (mascara = futuro distante) e o
    veredito do kronodroid usa o teste INTEIRO como alvo primario."""
    import importlib.util, pathlib, argparse
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "e0_generalization.py"
    spec = importlib.util.spec_from_file_location("e0g", p); e0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e0)
    _krono_fake(tmp_path)
    args = argparse.Namespace(kronodroid=str(tmp_path), krono_cut=2014, krono_far=2018, krono_max_train=600,
                              krono_n_val=200, krono_natural_ratio=False)
    Xtr, ytr, Xv, yv, Xte, yte, novel = e0.load("kronodroid", args, 0)
    assert Xtr.shape[0] == len(ytr) == 600 and Xv.shape[0] == 200 and Xte.shape[0] == len(yte) == len(novel)
    assert Xtr.shape[1] == Xv.shape[1] == Xte.shape[1] and 0 < novel.sum() < len(novel)
    assert e0._DATA_META["kronodroid"]["cut_year"] == 2014
    res = {"dataset": "kronodroid", "n_ok": 50, "n_total": 60, "mode": "train_loss", "stop_train_loss": 0.1, "control": "val",
           "results": {"cvar_test": {"partial_given_val": 0.3, "partial_ci95": [0.1, 0.5]},
                       "cvar_test_novel": {"partial_given_val": -0.3, "partial_ci95": [-0.5, -0.1]}}}
    v = e0.verdict(res)
    assert "SUSTENTA H_G" in v and "FUTURO" in v and "CONTRADIZ" not in v       # usou cvar_test, nao o 'novel'


# ---------------------------------------------------------------- Cenarios 1-3 (out/2026): medidas e intra-configuracao
def test_A31_lambda_max_e_segunda_estimativa_exatas_e_sem_tocar_no_rng():
    """lambda_max por iteracao de potencia e Tr(H^2) por Hutchinson (gerador proprio) batem com a Hessiana
    EXATA de uma forma quadratica; o RNG global do torch nao e' consumido (reprodutibilidade bit a bit da
    curvatura primaria quando as medidas extras sao ligadas); make_extra_measures devolve as chaves pedidas."""
    from tune3.curvature.measures import lambda_max, tr_h2_hutchinson, make_extra_measures
    A = torch.tensor([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])

    class Q(nn.Module):
        def __init__(self):
            super().__init__(); self.w = nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
        def forward(self, x=None):
            return 0.5 * self.w @ A @ self.w
    m = Q(); lf = lambda model: model()
    assert abs(lambda_max(m, lf, iters=300, tol=1e-12) - float(torch.linalg.eigvalsh(A).max())) < 1e-4
    assert abs(tr_h2_hutchinson(m, lf, probes=3000, seed=1) - float((A @ A).trace())) < 0.5
    s0 = torch.random.get_rng_state()
    _ = tr_h2_hutchinson(m, lf, probes=5, seed=3); _ = lambda_max(m, lf, iters=5)
    assert torch.equal(s0, torch.random.get_rng_state())
    net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2)); X = torch.randn(64, 6); y = (X[:, 0] > 0).long(); crit = nn.CrossEntropyLoss()
    extra = make_extra_measures(["sharp", "lmax", "curv2", "curvb"], sharp_rho=(0.1,), second_batch=(X[:32], y[:32]), probes=3)
    out = extra(net, X, y, crit)
    assert set(out) == {"sharp_0.1", "lmax", "curv2", "curvb"} and all(np.isfinite(v) for v in out.values()) and out["curv2"] > 0


def test_A32_analise_intra_configuracao_separa_geometria_de_arquitetura():
    """analyze() do e0_within_config: num cenario sintetico em que a curvatura varia ENTRE configuracoes junto com
    o alvo (confundimento por arquitetura) mas NAO dentro delas, a parcial intra e' nula e a entre e' positiva;
    com um efeito intra verdadeiro, o veredito passa a SUSTENTA; o ICC reflete a fracao entre configuracoes."""
    import importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "e0_within_config.py"
    spec = importlib.util.spec_from_file_location("e0w", p); w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    rng = np.random.default_rng(0); K, S = 12, 20

    def make(within_effect):
        rows = []
        for c in range(K):
            base_curv = 10 ** rng.uniform(0.5, 2.5); base_y = 2.0 + 0.6 * np.log10(base_curv)     # confundimento ENTRE
            for s in range(S):
                dc = rng.normal(0, 0.15); dv = rng.normal(0, 0.1)
                y = base_y + dv + within_effect * dc + rng.normal(0, 0.05)
                rows.append({"cfg": c, "seed": s, "reached": True, "curvature": base_curv * 10 ** dc, "curv2": base_curv * 10 ** (dc + rng.normal(0, 0.02)),
                             "cvar_val": 1.0 + dv + rng.normal(0, 0.03), "cvar_test": y, "hparams": {}})
        return rows
    r0 = w.analyze(make(0.0), "kronodroid", B=200, seed=0)
    m0 = r0["measures"]["curvature"]
    assert m0["ci95_within"][0] < 0 < m0["ci95_within"][1] and m0["partial_between"] > 0.5 and m0["icc_between"] > 0.9
    assert "nulo" in r0["verdict"]
    r1 = w.analyze(make(1.0), "kronodroid", B=200, seed=0)
    assert r1["measures"]["curvature"]["ci95_within"][0] > 0 and "SUSTENTA" in r1["verdict"]
    assert r1["reliability_curv2"]["spearman_within"] > 0.9


def test_A33_select_configs_estratifica_e_loader_aceita_emulador(tmp_path):
    """select_configs: K configuracoes elegiveis (atingiram T com folga), ordenadas e espalhadas pelos quantis da
    curvatura; loader do KronoDroid: sem os arquivos do real device, usa os do emulador (FirstModDate/LastModDate)."""
    import importlib.util, pathlib, json, zipfile
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "e0_within_config.py"
    spec = importlib.util.spec_from_file_location("e0w", p); w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    rows = [{"cfg": i, "reached": i % 3 != 0, "curvature": float(10 ** (i / 20)), "cvar_test": 1.0, "cvar_val": 1.0, "epochs_run": 100 + 10 * i,
             "hparams": {"learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.0, "hidden_dim": 64, "n_layers": 2}} for i in range(60)]
    J = {"meta": {"args": {"epochs": 1000, "seed": 0}}, "payload": {"datasets": [{"dataset": "kronodroid", "rows": rows}]}}
    f = tmp_path / "e0.json"; json.dump(J, open(f, "w"))
    sel, ds, _, n_pool = w.select_configs(str(f), 5)
    assert ds == "kronodroid" and n_pool == 40 and len(sel) == 5
    cs = [s["e0_curvature"] for s in sel]
    assert cs == sorted(cs) and all(s["cfg"] % 3 != 0 for s in sel) and all(s["e0_epochs_run"] <= 800 for s in sel)
    assert cs[0] < 10 ** 0.8 and cs[-1] > 10 ** 2.2                                  # espalhadas pelos quantis
    # emulador: mesma fixture do A28, renomeada e com as colunas de data do emulador
    import pandas as pd
    _krono_fake(tmp_path)
    for name, emu in (("real_malware_v1", "emu_malware_v1"), ("real_legitimate_v1", "emu_legitimate_v1")):
        with zipfile.ZipFile(tmp_path / f"{name}.zip") as z, z.open(f"{name}.csv") as fh:
            d = pd.read_csv(fh)
        d = d.rename(columns={"EarliestModDate": "FirstModDate", "HighestModDate": "LastModDate"})
        with zipfile.ZipFile(tmp_path / f"{emu}.zip", "w") as z:
            z.writestr(f"{emu}.csv", d.to_csv(index=False))
        (tmp_path / f"{name}.zip").unlink()
    from tune3.data.kronodroid import KronoDroidLoader, KronoDroidConfig
    m = KronoDroidLoader(KronoDroidConfig(data_dir=str(tmp_path), max_train=600, n_val=200)).load_splits_with_meta()
    assert m["meta"]["files"] == ["emu_malware_v1.zip", "emu_legitimate_v1.zip"] and m["meta"]["time_col"] == "LastModDate"
    assert "FirstModDate" not in m["feature_names"] and "LastModDate" not in m["feature_names"]


def _load_script(name):
    import importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_A34_check_reproduction_detecta_qualquer_diferenca():
    """check_reproduction.compare_dataset: linhas iguais -> IDENTICO em todos os campos; 1 ulp num CVaR -> falha com rtol 0
    e passa com rtol 1e-6; hiperparametro diferente ou linha faltando -> falha mesmo com tolerancia."""
    import copy
    c = _load_script("check_reproduction.py"); rng = np.random.default_rng(0)
    rows = [{"cfg": i, "hparams": {"learning_rate": 10 ** rng.uniform(-3, -1), "n_layers": int(rng.integers(1, 4))}, "curvature": float(10 ** rng.uniform(0, 3)),
             "cvar_val": float(rng.uniform(0.5, 2)), "cvar_test": float(rng.uniform(0.5, 3)), "best_epoch": int(rng.integers(1, 100)),
             "epochs_run": int(rng.integers(1, 100)), "cvar_test_novel": float(rng.uniform(0.5, 3)), "reached": True} for i in range(40)]
    A = {"rows": rows}; B = copy.deepcopy(A)
    s = c.compare_dataset(A, B, 0.0)
    assert s["ok"] and all(f["identical"] for f in s["fields"].values() if f["present"]) and s["hparams_diff"] == 0
    B2 = copy.deepcopy(A); B2["rows"][7]["cvar_test"] = np.nextafter(B2["rows"][7]["cvar_test"], 10.0)
    s0 = c.compare_dataset(A, B2, 0.0); s6 = c.compare_dataset(A, B2, 1e-6)
    assert not s0["ok"] and s0["fields"]["cvar_test"]["n_equal"] == 39 and not s0["fields"]["cvar_test"]["identical"]
    assert s6["ok"] and s6["fields"]["cvar_test"]["within_rtol"] and not s6["fields"]["cvar_test"]["identical"]
    B3 = copy.deepcopy(A); B3["rows"][3]["hparams"]["n_layers"] += 1
    assert not c.compare_dataset(A, B3, 1e-6)["ok"]
    B4 = copy.deepcopy(A); B4["rows"].pop()
    assert not c.compare_dataset(A, B4, 1e-6)["ok"]
    B5 = copy.deepcopy(A); B5["rows"][0]["curvature"] *= 1.5
    assert not c.compare_dataset(A, B5, 1e-6)["ok"]


def test_A35_measures_replication_regra_3_de_3_e_ordem_dos_folds(tmp_path):
    """measures_replication: veredito identico em todas as sementes -> replicado; qualquer mistura -> NAO replicado;
    uma semente -> 'nao replicado por desenho'; combinacao de Fisher com o sinal das parciais; semente/fold resolvidos
    do JSON de origem do E0 (ou da tag); tau de Kendall da ordem dos folds = 1 para ordens iguais."""
    import json
    m = _load_script("measures_replication.py")
    assert [m.code(v) for v in ("SUSTENTA (alem dos hiperparametros)", "proxy de hiperparametros (some dados os 5 hp)", "CONTRADIZ", "nao sustenta")] == ["S", "P", "C", "0"]
    assert m.rule(["S", "S", "S"]).startswith("S em 3/3") and "NAO replicado (S/S/0)" == m.rule(["S", "S", "0"]) and "por desenho" in m.rule(["C"])

    def meas(p, p5, verdict, ci2=True):
        return {"label": "x", "n_ok": 200, "rho_raw": p, "partial_val": p, "ci95_configs": [p - 0.1, p + 0.1],
                "ci95_two_level": [p - 0.15, p + 0.15] if ci2 else [float("nan")] * 2,
                "partial_val_lr_wd": p, "partial_val_5hp": p5, "ci95_5hp": [p5 - 0.1, p5 + 0.1], "verdict": verdict}
    S, P, Z = "SUSTENTA (alem dos hiperparametros)", "proxy de hiperparametros (some dados os 5 hp)", "nao sustenta"
    runs = []
    for seed, (p, p5, v) in enumerate([(0.3, 0.2, S), (0.25, 0.15, S), (0.35, 0.05, Z)]):
        runs.append({"scenario": "kronodroid", "dataset": "kronodroid", "fold": None, "seed": seed, "src": f"x-krono-m{seed}", "file": "f", "target": "cvar_test",
                     "measures": {"curvature": meas(0.05, 0.0, Z), "sharp_0.1": meas(p, p5, v)}})
    out = m.analyze(runs); e = out[0]["measures"]
    assert e["curvature"]["rule"].startswith("0 em 3/3") and e["sharp_0.1"]["rule"] == "NAO replicado (S/S/0)"
    assert e["sharp_0.1"]["n_val_above0"] == 3 and e["sharp_0.1"]["n_5hp_above0"] == 2 and 0.25 < e["sharp_0.1"]["fe_val"]["r"] < 0.35
    assert e["sharp_0.1"]["fe_val"]["ci95"][0] > 0 and m.analyze(runs[:1])[0]["measures"]["sharp_0.1"]["rule"].endswith("por desenho)")
    # mesma semente duas vezes no mesmo cenario -> erro (nao soma rodadas repetidas)
    with pytest.raises(SystemExit):
        m.analyze(runs + [dict(runs[0])])
    # semente/fold: do JSON de origem do E0 quando existe; da tag quando nao
    e0 = tmp_path / "e0"; e0.mkdir()
    json.dump({"meta": {"args": {"seed": 2, "cluster_fold": 3}}, "payload": {}}, open(e0 / "20261013_abc_dionatan-drebin-cluster-m2-f3.json", "w"))
    assert m.resolve_seed_fold({"src": "20261013_abc_dionatan-drebin-cluster-m2-f3", "name": "drebin_cluster fold 3"}, str(e0)) == (2, 3)
    assert m.resolve_seed_fold({"src": "20261013_abc_dionatan-krono-m1", "name": "kronodroid semente 1"}, str(e0)) == (1, None)
    assert m.resolve_seed_fold({"src": "20261013_abc_dionatan-drebin-cluster-m0-f4", "name": "drebin_cluster fold 4"}, str(e0)) == (0, 4)
    # ordem dos folds entre sementes
    dre = []
    for seed in (0, 1):
        for fold, p in enumerate([0.5, 0.1, -0.1, 0.2, 0.0]):
            dre.append({"scenario": f"drebin_cluster fold {fold}", "dataset": "drebin_cluster", "fold": fold, "seed": seed, "src": "", "file": "f",
                        "target": "cvar_test_novel", "measures": {"curvature": meas(p + 0.01 * seed, 0.0, Z)}})
    t = m.fold_ordering(dre)
    assert abs(t["curvature"]["tau"]["0-1"] - 1.0) < 1e-12 and t["curvature"]["order_by_seed"]["0"] == [0, 3, 1, 4, 2]


def test_A36_halves_baseline_grava_json_igual_ao_impresso(tmp_path, monkeypatch, capsys):
    """e0_halves_baseline --tag: o JSON gravado traz as mesmas medias/desvios impressos; sem --tag, nada e' gravado;
    o calculo (media sobre R divisoes da parcial(H2, curv | H1)) nao muda com a opcao."""
    import json, sys
    hb = _load_script("e0_halves_baseline.py"); rng = np.random.default_rng(0); n, nt, nv = 60, 400, 200
    curv = 10 ** rng.uniform(0, 3, n); L = rng.exponential(1.0, (n, nt)) * (1 + 0.3 * np.log10(curv))[:, None]; V = rng.exponential(1.0, (n, nv))
    rows = [{"cfg": i, "curvature": float(curv[i]), "cvar_val": 1.0, "cvar_test": 1.0, "reached": True} for i in range(n)]
    monkeypatch.chdir(tmp_path); (tmp_path / "results" / "raw").mkdir(parents=True); (tmp_path / "results" / "e0_generalization").mkdir()
    stem = "20261014-000000_abc_dionatan-x"
    json.dump({"meta": {"args": {"seed": 0}}, "payload": {"datasets": [{"dataset": "kronodroid", "rows": rows, "results": {"cvar_test": {"partial_given_val": 0.1}}}]}},
              open(tmp_path / "results" / "e0_generalization" / f"{stem}.json", "w"))
    np.savez(tmp_path / "results" / "raw" / f"{stem}_kronodroid.npz", val=V.astype(np.float16), test=L.astype(np.float16), test_novel_mask=np.zeros(nt, bool))
    monkeypatch.setattr(sys, "argv", ["x", f"results/e0_generalization/{stem}.json", "--R", "30"]); hb.main()
    printed = capsys.readouterr().out; assert "kronodroid semente 0" in printed and not (tmp_path / "results" / "e0_halves_baseline").exists()
    monkeypatch.setattr(sys, "argv", ["x", f"results/e0_generalization/{stem}.json", "--R", "30", "--tag", "t"]); hb.main()
    printed2 = capsys.readouterr().out
    files = list((tmp_path / "results" / "e0_halves_baseline").glob("*_t.json")); assert len(files) == 1
    row = json.load(open(files[0]))["payload"]["rows"][0]
    assert row["target"] == "cvar_test" and row["n"] == n and row["R"] == 30 and row["partial_val_e0"] == 0.1
    line = [l for l in printed2.splitlines() if l.strip().startswith("kronodroid semente 0")][0].split()
    assert f"{row['halves_mean']:+.3f}" == line[-4] and f"{row['halves_std']:.3f}" == line[-3] and f"{row['halves_q97_5']:+.2f}" == line[-1]
    line1 = [l for l in printed.splitlines() if l.strip().startswith("kronodroid semente 0")][0].split()
    assert line1[-4:] == line[-4:]                                                    # mesmo calculo com e sem --tag


def test_A37_e1_offline_criterio_de_escolha_detecta_informacao_alem_da_validacao():
    """e1_offline: uma medida sem informacao alem da validacao ATRAPALHA tanto quanto o placebo (dilui a validacao); uma
    medida que e' so' proxy da validacao e' nula; informativa alem da validacao, V+Tr ajuda (IC < 0); anti-informativa,
    atrapalha (IC > 0); w = 0 reproduz V exatamente (pareado); oraculo <= todos; regra 3/3 entre sementes."""
    m = _load_script("e1_offline.py"); rng = np.random.default_rng(0); n = 250

    def pool(effect, proxy=False):
        val = rng.normal(2.0, 0.3, n); meas = val + rng.normal(0, 0.01, n) if proxy else rng.normal(0, 1, n)
        y = val + effect * meas + rng.normal(0, 0.15, n)
        return {"name": "x", "dataset": "kronodroid", "seed": 0, "fold": None, "target": "cvar_test", "n": n,
                "val": val, "y": y, "measures": {"curvature": meas}}
    p0, p_help, p_hurt, p_proxy = pool(0.0), pool(0.4), pool(-0.4), pool(0.0, proxy=True)
    r0 = m.analyze_pool(p0, [30], [1.0], R=1500, B=60, Rb=200, seed=0)["runs"][0]["arms"]
    assert r0["V+Tr"]["verdict"] == "atrapalha" and r0["V+P"]["verdict"] == "atrapalha" and r0["V+P"]["delta_vs_V"] > 0
    assert abs(r0["V+Tr"]["delta_vs_V"] - r0["V+P"]["delta_vs_V"]) < 0.1            # ruido independente ~ placebo
    assert r0["oraculo"]["mean_target"] <= min(r0[a]["mean_target"] for a in r0) and r0["aleatorio"]["delta_vs_V"] > r0["V+Tr"]["delta_vs_V"]
    rp = m.analyze_pool(p_proxy, [30], [1.0], R=1500, B=60, Rb=200, seed=0)["runs"][0]["arms"]
    assert rp["V+Tr"]["verdict"] == "nulo" and abs(rp["V+Tr"]["delta_vs_V"]) < 0.01
    rh = m.analyze_pool(p_help, [30], [1.0], R=1500, B=60, Rb=200, seed=0)["runs"][0]["arms"]
    assert rh["V+Tr"]["verdict"] == "ajuda" and rh["V+Tr"]["delta_vs_V"] < 0 and rh["V+Tr"]["ci95_delta"][1] < 0
    rw = m.analyze_pool(p_hurt, [30], [1.0], R=1500, B=60, Rb=200, seed=0)["runs"][0]["arms"]
    assert rw["V+Tr"]["verdict"] == "atrapalha" and rw["V+Tr"]["ci95_delta"][0] > 0
    sim = m.simulate(p_help, 30, 0.0, 500, np.random.default_rng(1))                 # w = 0: mesma escolha de V, pareada
    assert np.array_equal(sim["V+Tr"], sim["V"]) and np.all(sim["oraculo"] <= sim["V"])
    # regra 3/3 entre sementes
    def res(seed, v):
        return {"dataset": "kronodroid", "fold": None, "seed": seed, "n_pool": n,
                "runs": [{"budget": 30, "w": 1.0, "arms": {"V+Tr": {"verdict": v}, "V+P": {"verdict": "atrapalha"}}}]}
    t = m.rule_across_seeds([res(0, "ajuda"), res(1, "ajuda"), res(2, "ajuda")], 30, 1.0)["kronodroid"]["arms"]
    assert t["V+Tr"]["rule"] == "ajuda em 3/3" and t["V+P"]["rule"] == "atrapalha em 3/3"
    t2 = m.rule_across_seeds([res(0, "ajuda"), res(1, "nulo"), res(2, "ajuda")], 30, 1.0)["kronodroid"]["arms"]
    assert t2["V+Tr"]["rule"].startswith("misto") and m.rule_across_seeds([res(0, "ajuda")], 30, 1.0)["kronodroid"]["arms"]["V+Tr"]["rule"].endswith("por desenho)")


def test_A38_paper_tables_fontes_formatos_e_ic_dois_niveis(tmp_path):
    """paper_tables: (i) a conferencia de fontes falha com commit errado ou arvore suja e passa so' com check=False;
    (ii) as tabelas Markdown nao levam '|' dentro das celulas e o LaTeX escapa os simbolos; (iii) two_level_ci reproduz
    a receita (e a ordem de sorteios) do e0_bootstrap2 num caso sintetico; (iv) channel() acha o hiperparametro cujo
    controle zera a parcial."""
    import json
    pt = _load_script("paper_tables.py"); b2 = _load_script("e0_bootstrap2.py")
    # (i) fontes
    d = tmp_path / "results" / "e1_offline"; d.mkdir(parents=True)
    json.dump({"meta": {"git": {"commit_short": "deadbee", "dirty": False}}, "payload": {}}, open(d / "20261021-000000_deadbee_dionatan.json", "w"))
    with pytest.raises(pt.SourceError):
        pt.resolve(tmp_path, "e1_offline", check=True)
    assert pt.resolve(tmp_path, "e1_offline", check=False)[0]["meta"]["git"]["commit_short"] == "deadbee"
    json.dump({"meta": {"git": {"commit_short": "fa36555", "dirty": True}}, "payload": {}}, open(d / "20261021-000000_deadbee_dionatan.json", "w"))
    with pytest.raises(pt.SourceError):
        pt.resolve(tmp_path, "e1_offline", check=True)
    json.dump({"meta": {"git": {"commit_short": "fa36555", "dirty": False}}, "payload": {}}, open(d / "20261021-000000_deadbee_dionatan.json", "w"))
    assert pt.resolve(tmp_path, "e1_offline", check=True)[1].name.endswith("_dionatan.json")
    # (ii) formatos
    md = pt.md_table(["a", "partial " + pt.GIVEN + " val"], [["x", pt.fmt(-0.123)]])
    assert md.count("|") == 3 * 3 and "−0.12" in md and pt.GIVEN in md
    tex = pt.tex_table(["a_b", "ρ 50%"], [["x", "−0.5"]], "cap & tion", "tab:x")
    assert r"a\_b" in tex and r"\%" in tex and r"$-$0.5" in tex and r"\&" in tex and r"\rho" in tex
    # (iii) IC em dois niveis == receita do e0_bootstrap2
    rng = np.random.default_rng(3); n, nv, nt = 40, 60, 80
    rows = [{"cfg": i, "curvature": float(10 ** rng.uniform(0, 2)), "cvar_val": 1.0, "cvar_test": 1.0, "reached": True,
             "hparams": {"learning_rate": float(10 ** rng.uniform(-4, -2)), "weight_decay": float(10 ** rng.uniform(-6, -2)), "dropout": float(rng.uniform(0, .5)), "hidden_dim": int(rng.integers(32, 257)), "n_layers": int(rng.integers(1, 5))}} for i in range(n)]
    Z = {"val": rng.exponential(1, (n, nv)).astype(np.float16), "test": rng.exponential(1, (n, nt)).astype(np.float16), "test_novel_mask": np.arange(nt) % 2 == 0}
    ci, ci5 = pt.two_level_ci(rows, Z, "cvar_test_novel", B=25, seed=7)
    # replica literal do laco do e0_bootstrap2 para uma rodada
    idx = np.array([r["cfg"] for r in rows]); V = Z["val"][idx].astype(np.float64); T = Z["test"][idx].astype(np.float64); TT = T[:, Z["test_novel_mask"]]
    h = lambda k: np.array([r["hparams"][k] for r in rows], float)
    H = [np.log10(h("learning_rate")), np.log10(h("weight_decay") + 1e-6), h("dropout"), h("hidden_dim"), h("n_layers")]
    lc = np.log10([r["curvature"] for r in rows]); rr = np.random.default_rng(7); a_, b_ = [], []; ntt = TT.shape[1]
    for _ in range(25):
        v = b2.cvar_rows(V[:, rr.integers(0, nv, nv)]); y = b2.cvar_rows(TT[:, rr.integers(0, ntt, ntt)]); ic = rr.integers(0, n, n)
        a_.append(pt.partial_spearman(y[ic], lc[ic], [v[ic]])); b_.append(pt.partial_spearman(y[ic], lc[ic], [v[ic]] + [x[ic] for x in H]))
    assert np.allclose(ci, np.nanpercentile(a_, [2.5, 97.5])) and np.allclose(ci5, np.nanpercentile(b_, [2.5, 97.5]))
    # (iv) canal
    depth = rng.integers(1, 5, 300); val = rng.normal(2, .3, 300); curv = 10 ** (0.4 * depth + rng.normal(0, .2, 300)); y = val + 0.3 * depth + rng.normal(0, .1, 300)
    rows2 = [{"cfg": i, "curvature": float(curv[i]), "cvar_val": float(val[i]), "cvar_test": float(y[i]), "reached": True,
              "hparams": {"learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": float(rng.uniform(0, .5)), "hidden_dim": int(rng.integers(32, 257)), "n_layers": int(depth[i])}} for i in range(300)]
    pv, with_hp, hp_own, ch = pt.channel(rows2, "cvar_test")
    assert pv > 0.5 and ch == "depth" and abs(with_hp["depth"]) < 0.15 and hp_own["depth"] > 0.5
