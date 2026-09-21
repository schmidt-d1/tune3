#!/usr/bin/env python
# scripts/validate_theory.py
"""
VALIDACAO TEORIA x IMPLEMENTACAO do Tune3 (Teste 0 do guia dos alunos).

Cada item confronta um componente do codigo com a propriedade matematica que a
teoria promete, por um CAMINHO INDEPENDENTE (a formula testada nunca gera o
valor esperado). O veredito usa a taxonomia do projeto:
    PROVADO | VERIFICADO(escopo) | REFUTADO | EM ABERTO

  V1  Hutchinson: HVP == H v (Hessiana exata) e E||Hv||^2 -> ||H||_F^2 = Tr(H^2)
  V2  Hutchinson: erro-padrao cai como 1/sqrt(M) (M = numero de sondas)
  V3  GSNR: formula == definicao (Liu et al. 2020) e GSNR cresce ~linearmente com B
  V4  CVaR: quantil == Rockafellar-Uryasev; media <= VaR <= CVaR
  V5  Cantelli: taxa de alarme falso <= 1-gamma sob cauda pesada (t de Student, 3 g.l.)
  V6  DDKF: corrige o log lr na direcao certa (regulador de desvios); engajamento R^2 (E2)
  V7  EoS: dispara sob progressive sharpening e NAO dispara em serie estacionaria
  V8  Protocolo estatistico: erro tipo I <= alpha sob H0 (simulacao) e poder em d_z=0.8, N=20
  V9  Ponta a ponta: Tune3 + 7 comparadores rodam com orcamento pequeno; Pareto
      nao-dominada; telemetria de engajamento do DDKF (E2) e do EoS presentes

Uso:
    python scripts/validate_theory.py                       # sintetico (CPU, ~2-5 min)
    python scripts/validate_theory.py --csv data/drebin215.csv --device cuda --tag aluno-nome
Saida: tabela de vereditos no terminal + results/validation/<ts>_<commit>_<tag>.json
"""
from __future__ import annotations

import argparse, math, time, warnings
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, torch.get_num_threads()))

REPORT: List[Dict] = []


def verdict(item: str, status: str, evidence: str, **numbers):
    REPORT.append({"item": item, "veredito": status, "evidencia": evidence, **numbers})
    print(f"  [{status:20s}] {item}\n      {evidence}")


def _synthetic(n, seed=0, d=215):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, d)) < 0.15).astype(np.float32)
    w = np.zeros(d); w[:20] = 1.0
    s = X @ w + rng.normal(0, 1.5, n)
    return X, (s > np.quantile(s, 0.63)).astype(int)


def load_data(csv, seed=0):
    if csv:
        from tune3.data.drebin import DrebinLoader, DrebinConfig
        Xtr, Xv, Xte, ytr, yv, yte = DrebinLoader(DrebinConfig(csv_path=csv, random_state=seed)).load_splits()
        return (Xtr, ytr, Xv, yv, Xte, yte), "DREBIN-215"
    Xtr, ytr = _synthetic(3000, seed); Xv, yv = _synthetic(1000, seed + 1); Xte, yte = _synthetic(1000, seed + 2)
    return (Xtr, ytr, Xv, yv, Xte, yte), "sintetico(3000/1000/1000, 215 feats)"


# ============================================================ V1 / V2
def v1_v2_hutchinson():
    from tune3.models.factory import build_model
    from tune3.curvature import HutchinsonEstimator, HutchinsonConfig
    torch.manual_seed(0)
    model = build_model(12, 2, hparams={"hidden_dim": 8, "n_layers": 2, "dropout": 0.3}).eval()
    X = torch.randn(64, 12); y = (X[:, 0] + 0.5 * X[:, 1] > 0).long(); crit = nn.CrossEntropyLoss()
    params = list(model.parameters())

    def flat_loss(th):
        idx = 0; out = {}
        for k, p in model.named_parameters():
            out[k] = th[idx:idx + p.numel()].view_as(p); idx += p.numel()
        return crit(torch.func.functional_call(model, out, (X,)), y)

    theta = torch.cat([p.detach().reshape(-1) for p in params]).clone().requires_grad_(True)
    H = torch.autograd.functional.hessian(flat_loss, theta).detach()       # lado B (exato)
    exact = float((H * H).sum())
    grads = torch.autograd.grad(crit(model(X), y), params, create_graph=True)
    g = torch.Generator().manual_seed(1); maxdiff = 0.0
    for _ in range(20):
        v = (torch.randint(0, 2, (theta.numel(),), generator=g) * 2 - 1).float()
        vs = []; idx = 0
        for p in params:
            vs.append(v[idx:idx + p.numel()].view_as(p)); idx += p.numel()
        gv = sum((gi * vi).sum() for gi, vi in zip(grads, vs))
        Hv = torch.cat([h.reshape(-1) for h in torch.autograd.grad(gv, params, retain_graph=True)])
        maxdiff = max(maxdiff, float((Hv - H @ v).abs().max()))
    # Monte Carlo com o estimador do repositorio (M grande)
    M = 4000
    est = HutchinsonEstimator(HutchinsonConfig(num_probes=M)).estimate(model, crit(model(X), y))
    per_probe = np.array([float(((H @ ((torch.randint(0, 2, (theta.numel(),)) * 2 - 1).float())) ** 2).sum())
                          for _ in range(3000)])
    se = per_probe.std(ddof=1) / math.sqrt(M); z = (est - exact) / se
    ok = maxdiff < 1e-5 and abs(z) < 4
    verdict("V1 Hutchinson == Tr(H^2) exato (n=%d params)" % theta.numel(),
            "PROVADO" if ok else "REFUTADO",
            f"max|HVP - Hv|={maxdiff:.1e}; estimador(M={M})={est:.4f} vs exato={exact:.4f}, z={z:+.2f}",
            maxdiff=maxdiff, est=est, exact=exact, z=z)
    # dropout ativo: mede outra coisa (sub-rede aleatoria)
    model.train()
    tr = np.mean([HutchinsonEstimator(HutchinsonConfig(num_probes=1)).estimate(model, crit(model(X), y)) for _ in range(300)])
    model.eval()
    verdict("V1b Hutchinson em model.train() (dropout=0.3) desvia do exato",
            "VERIFICADO(escopo: este MLP)" if abs(tr / exact - 1) > 0.05 else "EM ABERTO",
            f"razao train/exato = {tr / exact:.3f} -> por isso o trial usa model.eval()", ratio=tr / exact)
    # V2: erro-padrao ~ 1/sqrt(M)
    sds = {}
    for m in (1, 4, 16):
        vals = [HutchinsonEstimator(HutchinsonConfig(num_probes=m)).estimate(model, crit(model(X), y)) for _ in range(120)]
        sds[m] = float(np.std(vals, ddof=1))
    r1 = sds[1] / sds[4]; r2 = sds[4] / sds[16]     # teoria: 2 e 2
    ok2 = 1.4 < r1 < 2.8 and 1.4 < r2 < 2.8
    verdict("V2 desvio do estimador cai ~1/sqrt(M)", "VERIFICADO(escopo: M<=16, 120 rep.)" if ok2 else "REFUTADO",
            f"sd(M=1)/sd(M=4)={r1:.2f}, sd(M=4)/sd(M=16)={r2:.2f} (teoria: 2.00 e 2.00)", r1=r1, r2=r2)


# ============================================================ V3
def v3_gsnr(data):
    from tune3.curvature import GSNREstimator
    from tune3.models.factory import build_model
    Xtr, ytr = data[0], data[1]
    torch.manual_seed(0)
    model = build_model(Xtr.shape[1], 2, hparams={"hidden_dim": 64, "n_layers": 2, "dropout": 0.0})
    crit = nn.CrossEntropyLoss()
    X = torch.as_tensor(Xtr); y = torch.as_tensor(ytr, dtype=torch.long)
    est = GSNREstimator()

    def grads_for(B, K=16, seed=0):
        gen = torch.Generator().manual_seed(seed); out = []
        for _ in range(K):
            idx = torch.randperm(X.shape[0], generator=gen)[:B]
            model.zero_grad(); crit(model(X[idx]), y[idx]).backward()
            out.append(torch.cat([p.grad.reshape(-1) for p in model.parameters()]).numpy().copy())
        return out
    grads = grads_for(256)
    G = np.stack(grads); direct = (G.mean(0) @ G.mean(0)) / G.var(0, ddof=1).sum()   # definicao, sem o modulo
    g_est = est.from_gradients(grads)
    ok = abs(g_est - direct) / direct < 1e-6
    verdict("V3a GSNR(codigo) == definicao Liu et al. 2020", "PROVADO" if ok else "REFUTADO",
            f"codigo={g_est:.4e} definicao={direct:.4e}", gsnr=g_est)
    ratios = []
    for s in range(3):
        a = est.from_gradients(grads_for(128, seed=s)); b = est.from_gradients(grads_for(1024, seed=s))
        ratios.append(b / a)
    r = float(np.median(ratios))
    # Predicao ingenua (populacao infinita / amostragem COM reposicao): GSNR ∝ B => razao 8.
    # Aqui os minibatches vem de torch.randperm (SEM reposicao) de uma populacao finita N,
    # entao Var da media amostral = (sigma^2/B)*(N-B)/(N-1) e a razao prevista e' maior:
    N = int(X.shape[0])
    pred_naive = 1024 / 128
    pred_fpc = (1024 * (N - 1) / (N - 1024)) / (128 * (N - 1) / (N - 128))
    ok = 3.0 < r < 20.0
    verdict("V3b GSNR ~ proporcional a B (dGSNR/dB = GSNR/B)", "VERIFICADO(escopo: B in {128,1024})" if ok else "REFUTADO",
            f"GSNR(B=1024)/GSNR(B=128) mediana={r:.2f}; previsto={pred_naive:.2f} (populacao infinita) "
            f"e {pred_fpc:.2f} (correcao de populacao finita, N={N}, sem reposicao); tolerancia [3,20]",
            ratio=r, pred_naive=pred_naive, pred_fpc=pred_fpc, n_pop=N)


# ============================================================ V4
def v4_cvar():
    from tune3.core.objectives import cvar_quantile, cvar_rockafellar, value_at_risk
    rng = np.random.default_rng(0)
    L = rng.lognormal(0.0, 1.0, 20000)
    worst = 0.0
    for g in (0.8, 0.9, 0.95, 0.99):
        a, b = cvar_quantile(L, g), cvar_rockafellar(L, g); worst = max(worst, abs(a - b) / b)
    order = L.mean() <= value_at_risk(L, 0.95) <= cvar_quantile(L, 0.95)
    # valor teorico do CVaR lognormal(0,1) em 0.95: E[X | X>q] = exp(1/2) * Phi(1 - z_0.95)/(1-0.95)
    from scipy.stats import norm
    z = norm.ppf(0.95); theo = math.exp(0.5) * norm.cdf(1 - z) / 0.05
    emp = cvar_quantile(L, 0.95); rel = abs(emp - theo) / theo
    ok = worst < 0.02 and order and rel < 0.05
    verdict("V4 CVaR: quantil == Rockafellar-Uryasev; ordem media<=VaR<=CVaR; valor teorico lognormal",
            "PROVADO" if ok else "REFUTADO",
            f"max desvio quantil/RU={worst:.2%}; ordem={order}; CVaR_0.95 emp={emp:.3f} teo={theo:.3f} ({rel:.1%})",
            worst=worst, emp=emp, theo=theo)


# ============================================================ V5
def v5_cantelli():
    """A cota de Cantelli e' distribution-free para media e variancia CONHECIDAS.
    O guarda usa mediana/MAD ESTIMADOS numa janela deslizante, entao a cota vira
    heuristica: precisa ser verificada empiricamente NA JANELA QUE O TRIAL USA.
    V5 = janela 30 (referencia); V5b = janela do TrialConfig real."""
    from tune3.safety import CantelliGuard, CantelliConfig
    from tune3.integration.trial import TrialConfig

    def false_alarm_rate(window, reps=200, seed=0):
        rng = np.random.default_rng(seed)
        alarms = 0; tests = 0
        for _ in range(reps):
            g = CantelliGuard(CantelliConfig(gamma=0.95, window_size=window))
            series = 1.0 + 0.05 * rng.standard_t(df=3, size=130)   # cauda pesada, estacionaria
            for L in series:
                n_before = len(g.loss_buffer)
                fired = g.should_abort(float(L))
                if n_before >= window:
                    tests += 1; alarms += int(fired)
        return alarms / tests, tests

    rate, tests = false_alarm_rate(30)
    verdict("V5 Cantelli: alarme falso <= 1-gamma sob t(3)", "VERIFICADO(escopo: t3, janela 30)" if rate <= 0.05 else "REFUTADO",
            f"taxa empirica={rate:.4f} (cota de Cantelli: 0.05; {tests} testes)", rate=rate)

    w_trial = int(TrialConfig().cantelli.window_size)
    rate_b, tests_b = false_alarm_rate(w_trial)
    verdict(f"V5b Cantelli na janela REAL do trial (window={w_trial}, mediana/MAD estimados)",
            f"VERIFICADO(escopo: t3, janela {w_trial})" if rate_b <= 0.05 else "REFUTADO",
            f"taxa empirica={rate_b:.4f} (cota nominal 0.05; {tests_b} testes). "
            f"Sem garantia formal com localizacao/escala estimadas -- e' verificacao empirica.",
            rate=rate_b, window=w_trial)


# ============================================================ V6
def v6_ddkf():
    """O DDKF e' um REGULADOR de desvios (nao um otimizador do lr): ele identifica
    Cov(log lr, z) durante o warm-up (com excitacao) e, depois, corrige o log lr
    na direcao que traz a observacao de volta a' media recente. Testamos
    exatamente isso; e medimos separadamente o ENGAJAMENTO (R^2) ao longo do
    tempo, que e' a pergunta do E2."""
    from tune3.micro import DDKFController, DDKFConfig
    rng = np.random.default_rng(0)
    # (a) sinal da correcao: Cov(lr, loss) > 0 e choque de loss ACIMA da media => u > 0 (reduz lr)
    d = DDKFController(0.0, DDKFConfig(min_samples=20, window_size=60, exploration_std=0.0, max_correction=100.0))
    for _ in range(40):
        lr = rng.normal(0, 0.3); d.x_hat = lr
        d.update([2.0 * lr + rng.normal(0, 0.05), 2.0 * lr, -lr])
    d.x_hat = 0.0; d.update([10.0, 10.0, 0.0]); u_pos = d.last_u
    d2 = DDKFController(0.0, DDKFConfig(min_samples=20, window_size=60, exploration_std=0.0, max_correction=100.0))
    for _ in range(40):
        lr = rng.normal(0, 0.3); d2.x_hat = lr
        d2.update([-2.0 * lr + rng.normal(0, 0.05), -2.0 * lr, lr])
    d2.x_hat = 0.0; d2.update([10.0, 10.0, 0.0]); u_neg = d2.last_u
    ok = u_pos > 0 and u_neg < 0
    verdict("V6a DDKF corrige na direcao certa (regulador de desvios)", "VERIFICADO(escopo: sistema linear sintetico)" if ok else "REFUTADO",
            f"Cov>0 & choque => u={u_pos:+.3f} (reduz lr); Cov<0 & choque => u={u_neg:+.3f} (aumenta lr)", u_pos=u_pos, u_neg=u_neg)
    # (b) engajamento: com exploration_decay=0 (padrao) a excitacao para no fim do warm-up e R^2 decai
    d3 = DDKFController(np.log(0.1), DDKFConfig(min_samples=20, window_size=60, exploration_std=0.05),
                        rng=np.random.default_rng(1))
    r2_hist = []
    for t in range(80):
        x = d3.x_hat
        loss = 1.0 + 3.0 * (x - np.log(0.01)) + rng.normal(0, 0.05)
        d3.update([loss, loss, -x]); r2_hist.append(d3.information_ratio())
    r2_early = float(np.mean(r2_hist[21:31])); r2_late = float(np.mean(r2_hist[70:]))
    verdict("V6b engajamento do DDKF ao longo do tempo (pergunta do E2)", "EM ABERTO",
            f"R^2 medio t=21..30: {r2_early:.4f}; t=70..79: {r2_late:.4f}; limiar do regime B = 0.10. "
            f"Com exploration_decay=0 a excitacao cessa apos o warm-up e o filtro tende ao regime A (predict-only). "
            f"Isto NAO e' um bug do teste: e' o comportamento a medir no E2 (telemetria n_ddkf_active_epochs).",
            r2_early=r2_early, r2_late=r2_late)


# ============================================================ V7
def v7_eos():
    from tune3.safety import EoSDetector, EoSConfig
    rng = np.random.default_rng(0)
    a = EoSDetector(EoSConfig()); [a.update(1.0 + rng.normal(0, 0.01)) for _ in range(40)]
    b = EoSDetector(EoSConfig()); [b.update(1.2 ** k) for k in range(40)]
    ok = a.n_triggers == 0 and b.n_triggers >= 1
    verdict("V7 EoS: 0 disparos em serie estacionaria; >=1 em sharpening geometrico", "PROVADO" if ok else "REFUTADO",
            f"estacionaria={a.n_triggers}, sharpening(1.2^k)={b.n_triggers}", stat=a.n_triggers, sharp=b.n_triggers)


# ============================================================ V8
def v8_stats():
    from tune3.experiments.stats import compare_paired
    rng = np.random.default_rng(0)
    n_sim = 300; N = 20
    false_wins = 0
    for _ in range(n_sim):
        t = rng.normal(1.0, 0.1, N)
        base = {f"b{i}": list(t + rng.normal(0, 0.1, N)) for i in range(3)}   # H0: iguais
        r = compare_paired(list(t), base, n_boot=50)
        false_wins += int(any(c["prereg_win"] for c in r["comparisons"].values()))
    fwer = false_wins / n_sim
    power_hits = 0
    for _ in range(n_sim):
        t = rng.normal(1.0, 0.1, N)
        base = {"b": list(t + 0.08 + rng.normal(0, 0.1, N))}                  # d_z = 0.8
        r = compare_paired(list(t), base, n_boot=50)
        power_hits += int(r["comparisons"]["b"]["prereg_win"])
    power = power_hits / n_sim
    ok = fwer <= 0.08 and power >= 0.7
    verdict("V8 protocolo: FWER sob H0 e poder em d_z=0.8 (N=20)", "VERIFICADO(escopo: normal, 300 sim.)" if ok else "REFUTADO",
            f"FWER empirico={fwer:.3f} (alvo <=0.05); poder={power:.2f}", fwer=fwer, power=power)


# ============================================================ V9
def v9_end_to_end(data, device, epochs, n_init, n_iter):
    from tune3.experiments.protocol import run_seed, ProtocolConfig, ALL_METHODS
    from botorch.utils.multi_objective.pareto import is_non_dominated
    Xtr, ytr, Xv, yv, Xte, yte = data
    cfg = ProtocolConfig(seeds=[0], epochs=epochs, n_init=n_init, n_iter=n_iter, asha_configs=8,
                         device=device, methods=list(ALL_METHODS))
    t0 = time.time()
    res = run_seed(0, lambda s: data, cfg)
    dt = time.time() - t0
    tel = res["_telemetry"]["tune3"]
    Y = torch.tensor([[t["cvar"], t["curvature"]] for t in tel if not t["aborted"]], dtype=torch.double)
    pf = Y[is_non_dominated(-Y)]
    n_eos = sum(t["n_eos_triggers"] for t in tel); n_active = sum(t["n_ddkf_active_epochs"] for t in tel)
    n_ep = sum(t["n_epochs_run"] for t in tel)
    regA = np.mean([t["regime_fractions"]["A"] for t in tel])
    cvars = {m: round(res[m]["cvar_test"], 4) for m in cfg.methods}
    ok = all(np.isfinite(v) and v < 1e3 for v in cvars.values()) and len(pf) >= 1
    verdict("V9 ponta a ponta: 8 metodos, Pareto, telemetria", "VERIFICADO(escopo: 1 seed, orcamento minimo)" if ok else "REFUTADO",
            f"{dt/60:.1f} min; CVaR_teste={cvars}; Pareto={len(pf)} pts de {len(Y)} trials; "
            f"DDKF ativo em {n_active}/{n_ep} epocas (regime A medio={regA:.2f}); EoS disparos={n_eos}",
            minutes=dt / 60, cvar_test=cvars, n_pareto=len(pf), ddkf_active=n_active, epochs=n_ep,
            regime_A=regA, eos_triggers=n_eos)
    if n_active == 0:
        print("      NOTA: o DDKF nao saiu do regime A (predict-only) neste orcamento -- e' exatamente a "
              "pergunta do E2 (engajamento). Nao e' erro do teste; e' um dado a reportar.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-init", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.time()
    data, dname = load_data(args.csv)
    print(f"\n=== VALIDACAO TEORIA x IMPLEMENTACAO -- dados: {dname} -- device: {args.device}\n")
    v1_v2_hutchinson(); v3_gsnr(data); v4_cvar(); v5_cantelli(); v6_ddkf(); v7_eos(); v8_stats()
    if not args.skip_e2e:
        v9_end_to_end(data, args.device, args.epochs, args.n_init, args.n_iter)

    n_ref = sum(r["veredito"] == "REFUTADO" for r in REPORT)
    print(f"\n=== RESUMO: {len(REPORT)} itens, {n_ref} REFUTADO(s), {(time.time()-t0)/60:.1f} min ===")
    from tune3.experiments.results_io import save_result
    save_result({"report": REPORT, "dataset": dname}, experiment="validation", tag=args.tag,
                args=vars(args), started_at=t0)
    raise SystemExit(1 if n_ref else 0)


if __name__ == "__main__":
    main()
