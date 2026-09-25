#!/usr/bin/env python
# scripts/e0_generalization.py
"""
E0 NA FORMA QUE A TESE EXIGE: a curvatura preve GENERALIZACAO alem do que a validacao ja mostra?

Por que este teste (23/09/2026). O compare_landscape mostrou que no NSL-KDD a curvatura se
correlaciona com o CVaR de VALIDACAO (rho = +0,56, n = 30) e no DREBIN nao (rho = +0,04). Isso
parece bom para o Tune3, mas e' ambiguo: se Tr(H^2) acompanha o CVaR de validacao, minimiza-la
duplica em parte o que o BO mono-objetivo ja' faz -- o 2o objetivo seria REDUNDANTE, nao
complementar (e a frente de Pareto colapsa, como visto no V9). A tese geometrica (H_G: minimos
planos generalizam melhor) so' se sustenta se a curvatura carregar informacao sobre o TESTE que
a validacao NAO carrega. Isso e' uma correlacao PARCIAL:

    rho( log Tr(H^2) , CVaR_teste  |  CVaR_validacao )

Predicao de H_G: parcial > 0 (mais curvatura -> pior no teste, a validacao fixada), com IC
excluindo zero -- e MAIS forte nos ataques NUNCA vistos (split oficial do NSL-KDD), que e' onde
"generalizar" significa alguma coisa. Predicao de H_M (curvatura redundante): parcial ~ 0.

Cada configuracao: treino com early stopping na VALIDACAO, modelo da melhor epoca restaurado,
curvatura (Hutchinson, M=20, eval) nesse modelo, e UMA avaliacao no teste. Nada do teste
entra em escolha nenhuma. Correlacoes em POSTOS (Spearman), robustas a cauda pesada; IC por
bootstrap percentilico pareado sobre as configuracoes.

Criterios de qualidade da amostra (acrescentados em 23/09/2026, ANTES da rodada k=400, apos a
rodada k=40 com --epochs 30 mostrar 38/40 configuracoes do NSL-KDD com a melhor epoca = ultima):
  - ESTAGNADA: melhor epoca < --min-best-epoch (padrao 2). Com lr ~1e-5 e batch 4096 a melhora
    por epoca fica abaixo de min_delta=1e-4, a paciencia esgota e o "melhor modelo" e' o da
    epoca 0 -- praticamente a inicializacao. Essas configuracoes sao EXCLUIDAS da analise.
  - TRUNCADA: melhor epoca = ultima epoca permitida (o early stopping nao disparou). Com SGD,
    batch 4096 e lr ate' 1e-5, isso e' a REGRA, nao a excecao: num teste com 200 epocas, 11 de 12
    configuracoes ainda melhoravam na epoca 199. Por isso o orcamento padrao aqui e' o MESMO do
    protocolo (ProtocolConfig: 100 epocas, paciencia 10) -- a pergunta pre-registrada e' se H_G
    vale no orcamento em que o Tune3 de fato opera. Quando mais da metade da amostra e' truncada,
    o veredito diz isso explicitamente: vale para ESTE orcamento, nao para minimos convergidos.
    A correlacao curvatura x CVaR de validacao DEPENDE do orcamento (NSL-KDD: +0,56 com 10
    epocas, -0,03 com 30, -0,77 com 200 num teste de 12 configuracoes), porque o posto do CVaR
    entre configuracoes muda com o treino (rho = +0,39 entre 10 e 30 epocas) e o da curvatura
    quase nao muda (rho = +0,87). Rode tambem --epochs 30 e 200 como analise de sensibilidade.

Dois modos acrescentados em 24/09/2026 (fase exploratoria, apos o E0 k=400):
  --val-split   divide a validacao em duas metades estratificadas: SEL escolhe a melhor epoca,
                HOLD so' e' medida no modelo final. Testa a hipotese de VIES DE SELECAO: no k=400
                o lr previa o teste alem da validacao (parcial +0,32 a +0,85); se a causa for o
                otimismo do CVaR medido na mesma validacao que escolheu a epoca, o lr tambem deve
                prever HOLD alem de SEL -- e SEL e HOLD sao i.i.d., entao isso so' pode ser selecao.
                O controle das parciais passa a ser o CVaR de HOLD.
  --crossfit    (24/09) --val-split com os papeis trocados numa segunda passada. Motivo: no NSL-KDD a
                HOLD saiu 0,13 MELHOR que a SEL em media -- sinal que selecao nao produz; e' diferenca de
                composicao entre as metades (validacao de 5 000 -> cauda de 125 amostras por metade).
                Media dos dois gaps = otimismo por selecao; metade da diferenca = composicao.
  --stop-train-loss T
                parada por perda de treino (Jiang et al., ICLR 2020): cada configuracao treina ate'
                a perda de treino <= T (teto --epochs); a validacao nao escolhe nada. Remove o
                confundimento de orcamento. Configuracoes que nao atingem T sao EXCLUIDAS e a faixa
                de lr retida e' reportada (as de lr baixo tendem a nao atingir).

CHECAGEM DE ARTEFATO (25/09/2026), impressa sempre. Motivo: na rodada com parada por perda de treino
(k=600) a parcial saiu NEGATIVA nos dois datasets -- inclusive no DREBIN, onde validacao e teste sao
i.i.d. e, portanto, nao ha' "generalizacao alem da validacao" a medir, so' ruido e vies de estimacao.
Um valor sistematico ali aponta para artefato do estimador. Duas checagens: (i) CVaR com n pareado
(validacao e alvo em subamostras do mesmo tamanho) e (ii) parcial entre duas metades i.i.d. do
proprio alvo, cujo valor esperado e' zero.

Uso:
    python scripts/e0_generalization.py --device cuda --tag dionatan                 # NSL-KDD + DREBIN
    python scripts/e0_generalization.py --k 400 --val-split --device cuda --tag dionatan-k400-split
    python scripts/e0_generalization.py --k 400 --crossfit --device cuda --tag dionatan-k400-crossfit
    python scripts/e0_generalization.py --k 400 --stop-train-loss 0.05 --epochs 1000 --device cuda --tag dionatan-k400-stop
    python scripts/e0_generalization.py --k 400 --device cuda --tag dionatan-k400          # primario
    python scripts/e0_generalization.py --k 400 --epochs 30 --device cuda --tag dionatan-k400-e30
    python scripts/e0_generalization.py --k 400 --epochs 200 --patience 15 --device cuda --tag dionatan-k400-e200
"""
from __future__ import annotations

import argparse, time, warnings
import numpy as np
from scipy.stats import rankdata, pearsonr

from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
from tune3.core.objectives import cvar
from tune3.experiments.results_io import save_result
from tune3.experiments.stats import partial_spearman

SPACE = {"log_lr": (-5.0, -2.0), "log_wd": (-6.0, -2.0), "dropout": (0.0, 0.5),
         "hidden_dim": (32.0, 256.0), "n_layers": (1.0, 4.0)}


def sample_configs(k, seed):
    rng = np.random.default_rng(seed); out = []
    for _ in range(k):
        r = {n: rng.uniform(lo, hi) for n, (lo, hi) in SPACE.items()}
        out.append({"learning_rate": 10 ** r["log_lr"], "weight_decay": 10 ** r["log_wd"],
                    "dropout": r["dropout"], "hidden_dim": int(round(r["hidden_dim"])),
                    "n_layers": int(round(r["n_layers"]))})
    return out


def boot_ci(fn, n, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def split_half(X, y, seed):
    """Divide (X, y) em duas metades estratificadas por classe (SEL, HOLD)."""
    rng = np.random.default_rng(seed); a, b = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c)); h = len(idx) // 2
        a.append(idx[:h]); b.append(idx[h:])
    a = np.sort(np.concatenate(a)); b = np.sort(np.concatenate(b))
    return X[a], y[a], X[b], y[b]


def load(name, args, seed):
    if name == "drebin_cluster":
        # DREBIN com deslocamento: o malware de TESTE vem inteiro de um cluster (k-means) que nao
        # aparece no treino nem na validacao; benignos divididos nas mesmas proporcoes. E' o analogo
        # dos "ataques novos" do NSL-KDD dentro do dominio malware Android.
        from tune3.data.registry import load_dataset
        from tune3.data.shift import cluster_holdout_split
        Xa, Xb, Xc, ya, yb, yc = load_dataset("drebin", args.drebin, 0)
        X = np.concatenate([Xa, Xb, Xc]); y = np.concatenate([ya, yb, yc])
        Xtr, ytr, Xv, yv, Xte, yte = cluster_holdout_split(X, y, args.n_clusters, args.cluster_fold,
                                                           seed=seed)
        return Xtr, ytr, Xv, yv, Xte, yte, (yte == 1)
    if name == "nslkdd":
        from tune3.data.nslkdd import NSLKDDLoader, NSLKDDConfig
        m = NSLKDDLoader(NSLKDDConfig(data_dir=args.nslkdd, random_state=seed, split="official",
                                      max_train=args.nsl_max_train)).load_splits_with_meta()
        return m["X_train"], m["y_train"], m["X_val"], m["y_val"], m["X_test"], m["y_test"], m["test_novel_mask"]
    from tune3.data.registry import load_dataset
    Xtr, Xv, Xte, ytr, yv, yte = load_dataset("drebin", args.drebin, seed)
    return Xtr, ytr, Xv, yv, Xte, yte, None


def run_dataset(name, configs, args):
    Xtr, ytr, Xv, yv, Xte, yte, novel = load(name, args, args.seed)
    hold = None
    if args.crossfit:
        args.val_split = True
    if args.val_split:
        Xv, yv, Xh, yh = split_half(Xv, yv, args.seed + 1000)
        hold = (Xh, yh)
    stop = args.stop_train_loss
    # --- checagens de artefato (25/09): indices sorteados UMA vez por dataset (numeros aleatorios
    # comuns entre configuracoes), para CVaR com n pareado e para metades i.i.d. do teste ---
    arng = np.random.default_rng(args.seed + 2000)
    subsets = {"cvar_test": np.arange(len(yte))}
    if novel is not None and novel.any():
        subsets["cvar_test_novel"] = np.flatnonzero(novel); subsets["cvar_test_known"] = np.flatnonzero(~novel)
    nv = len(yv) // 2 if args.val_split else len(yv)
    match = {}
    for key, idx in subsets.items():          # m = min(n_val, n_alvo); R subamostras de tamanho m
        m_ = min(nv, len(idx))
        match[key] = (m_, [arng.choice(nv, m_, replace=False) for _ in range(args.match_reps)],
                      [arng.choice(idx, m_, replace=False) for _ in range(args.match_reps)])
    halves = {}
    for key, idx in subsets.items():          # metades i.i.d. de mesmo tamanho do alvo
        perm = arng.permutation(idx); h = len(perm) // 2
        halves[key] = (perm[:h], perm[h:2 * h])
    tc = PlainTrialConfig(max_epochs=args.epochs, patience=args.patience, device=args.device,
                          gamma=0.95, seed=args.seed, stop_train_loss=stop)
    rows = []; losses_val, losses_test = [], []
    for i, hp in enumerate(configs):
        r = plain_trial(hp, (Xtr, ytr, Xv, yv), tc, test_data=(Xte, yte),
                        return_test_losses=True, holdout_data=hold, return_val_losses=True)
        tl = r.get("test_losses")
        row = {"cfg": i, "hparams": hp, "curvature": float(r["curvature"]),
               "cvar_val": float(r["cvar"]), "cvar_test": float(r["cvar_test"]),
               "best_epoch": int(r.get("best_epoch", -1)), "epochs_run": int(r.get("epochs_run", -1)),
               "train_loss_final": float(r.get("train_loss_final", float("nan")))}
        if hold is not None:
            row["cvar_val_hold"] = float(r["cvar_holdout"])
        if args.crossfit:                      # papeis trocados: HOLD escolhe a epoca, SEL so' e' medida
            rb = plain_trial(hp, (Xtr, ytr, hold[0], hold[1]), tc, holdout_data=(Xv, yv))
            row["cvar_sel_B"] = float(rb["cvar"]); row["cvar_hold_B"] = float(rb["cvar_holdout"])
            row["best_epoch_B"] = int(rb.get("best_epoch", -1))
        if stop is not None:
            row["reached"] = bool(r["reached_train_loss"])
        if novel is not None and tl is not None and novel.any():
            row["cvar_test_novel"] = float(cvar(tl[novel], 0.95))
            row["cvar_test_known"] = float(cvar(tl[~novel], 0.95))
        vl_ = r.get("val_losses")
        if args.save_losses and tl is not None and vl_ is not None:
            losses_val.append(np.asarray(vl_, np.float16)); losses_test.append(np.asarray(tl, np.float16))
        if tl is not None and vl_ is not None and np.all(np.isfinite(tl)) and np.all(np.isfinite(vl_)):
            for key in subsets:
                m_, iv, it = match[key]
                row[key + "_m"] = float(np.mean([cvar(tl[j], 0.95) for j in it]))
                row["cvar_val_m_" + key] = float(np.mean([cvar(vl_[j], 0.95) for j in iv]))
                a_, b_ = halves[key]
                row[key + "_h1"] = float(cvar(tl[a_], 0.95)); row[key + "_h2"] = float(cvar(tl[b_], 0.95))
        rows.append(row)

    # 1e3 e' o SENTINELA de falha do plain_trial (valor nao finito); uma curvatura real acima de
    # 1000 e' valida (25/09: no fold 1 do DREBIN por cluster uma configuracao com Tr(H^2) = 1115 era
    # descartada pelo filtro antigo "< 1e3").
    base_ok = lambda r: (np.isfinite(r["curvature"]) and r["curvature"] > 0 and r["curvature"] != 1e3
                         and r["cvar_test"] != 1e3 and r.get("cvar_val_hold", 0.0) != 1e3
                         and np.isfinite(r["cvar_test"]))
    if stop is not None:                       # parada por perda de treino: exclui quem nao atingiu
        excluded = [r for r in rows if not r["reached"]]
        ok = [r for r in rows if r["reached"] and base_ok(r)]
        excl_label = f"nao atingiram perda de treino <= {stop}"
    else:                                      # early stopping: exclui estagnadas
        excluded = [r for r in rows if r["best_epoch"] < args.min_best_epoch]
        ok = [r for r in rows if r["best_epoch"] >= args.min_best_epoch and base_ok(r)]
        excl_label = "estagnadas"
    n = len(ok)
    if stop is not None:                       # para calibrar T: distribuicao da perda de treino final
        tlf = np.array([r["train_loss_final"] for r in rows])
        print(f"  perda de treino final (todas): min {np.nanmin(tlf):.3f}  mediana {np.nanmedian(tlf):.3f}  "
              f"max {np.nanmax(tlf):.3f};  atingiram T={stop}: {len(rows) - len(excluded)}/{len(rows)}")
    n_trunc = sum(r["best_epoch"] >= args.epochs - 1 for r in ok) if stop is None else 0
    frac_trunc = n_trunc / n if (n and stop is None) else 0.0

    arr = lambda k: np.array([r[k] for r in ok], float)
    hpa = lambda k: np.array([r["hparams"][k] for r in ok], float)
    lc = np.log10(arr("curvature")) if n else np.array([])
    cv_sel = arr("cvar_val") if n else np.array([])
    ctrl = arr("cvar_val_hold") if (n and hold is not None) else cv_sel
    ctrl_name = "HOLD" if hold is not None else "val"
    lr = np.log10(hpa("learning_rate")) if n else np.array([])
    lwd = np.log10(hpa("weight_decay")) if n else np.array([])
    do, hd, nl = (hpa("dropout"), hpa("hidden_dim"), hpa("n_layers")) if n else ([], [], [])
    lr_all = np.log10([r["hparams"]["learning_rate"] for r in rows])

    targets = [("cvar_test", "teste (todo)")]
    if ok and "cvar_test_novel" in ok[0]:
        if name == "drebin_cluster":
            targets += [("cvar_test_novel", "teste: malware do cluster NOVO"), ("cvar_test_known", "teste: benignos")]
        else:
            targets += [("cvar_test_novel", "teste: ataques NOVOS"), ("cvar_test_known", "teste: ataques conhecidos")]

    out = {"dataset": name, "epochs": args.epochs, "mode": "train_loss" if stop is not None else "val_early_stopping",
           "stop_train_loss": stop, "val_split": bool(hold is not None), "control": ctrl_name,
           "n_ok": n, "n_total": len(rows), "n_excluded": len(excluded), "excluded_reason": excl_label,
           "n_stalled": len(excluded) if stop is None else None,
           "n_truncated": n_trunc, "frac_truncated": frac_trunc,
           "log10_lr_range_all": [float(lr_all.min()), float(lr_all.max())],
           "log10_lr_range_ok": [float(lr.min()), float(lr.max())] if n else None,
           "rows": rows, "results": {}}
    if args.save_losses and losses_val:
        out["_losses"] = {"val": np.stack(losses_val), "test": np.stack(losses_test),
                          "y_val": np.asarray(yv), "y_test": np.asarray(yte),
                          "test_novel_mask": (np.asarray(novel) if novel is not None else np.zeros(len(yte), bool))}
    extra = (f"{n_trunc} truncadas = {frac_trunc:.0%} das validas" if stop is None else
             f"log10 lr retido em [{lr.min():.2f}, {lr.max():.2f}] de [{lr_all.min():.2f}, {lr_all.max():.2f}]" if n else "")
    n_invalid = len(rows) - len(excluded) - n
    inv = f"; {n_invalid} com valor invalido (falha numerica)" if n_invalid else ""
    out["n_invalid"] = n_invalid
    print(f"\n[{name}] {n}/{len(rows)} configuracoes validas ({len(excluded)} {excl_label} excluidas{inv}; {extra})")
    if n and frac_trunc > 0.5:
        print(f"  NOTA: em {frac_trunc:.0%} das configuracoes a melhor epoca foi a ULTIMA (early stopping "
              f"nao disparou). O resultado vale para o orcamento de {args.epochs} epocas, nao para "
              f"minimos convergidos.")
    if n < 10:
        out["verdict_note"] = "amostra pequena"
        return out
    print(f"  controle = CVaR de {ctrl_name}")
    print(f"  {'alvo':28s} {'rho(curv,alvo)':>15} {'parcial|ctrl':>13} {'IC95':>16} {'|ctrl,lr,wd':>12} {'|ctrl,5hp':>10}")
    for key, label in targets:
        y = arr(key)
        raw = float(pearsonr(rankdata(lc), rankdata(y))[0])
        p1 = partial_spearman(y, lc, [ctrl])
        ci = boot_ci(lambda idx: partial_spearman(y[idx], lc[idx], [ctrl[idx]]), n)
        p2 = partial_spearman(y, lc, [ctrl, lr, lwd])
        p3 = partial_spearman(y, lc, [ctrl, lr, lwd, do, hd, nl])
        plr = partial_spearman(y, lr, [ctrl])
        out["results"][key] = {"rho_raw": raw, "partial_given_val": p1, "partial_ci95": ci,
                               "partial_given_val_lr_wd": p2, "partial_given_val_5hp": p3,
                               "partial_lr_given_val": plr}
        print(f"  {label:28s} {raw:>+15.3f} {p1:>+13.3f}  [{ci[0]:+.2f}, {ci[1]:+.2f}] {p2:>+12.3f} {p3:>+10.3f}")
    k0 = targets[-2][0] if len(targets) > 1 else "cvar_test"
    print(f"  (referencia) rho(curvatura, CVaR de {ctrl_name}) = {float(pearsonr(rankdata(lc), rankdata(ctrl))[0]):+.3f}"
          f";  parcial(alvo, lr | {ctrl_name}) = {out['results'][k0]['partial_lr_given_val']:+.3f}")

    if hold is not None:                       # teste direto do vies de selecao
        gap = ctrl - cv_sel
        b = partial_spearman(ctrl, lr, [cv_sel])
        bci = boot_ci(lambda idx: partial_spearman(ctrl[idx], lr[idx], [cv_sel[idx]]), n)
        rg = float(pearsonr(rankdata(lr), rankdata(gap))[0])
        plr_sel = partial_spearman(arr(k0), lr, [cv_sel])
        out["selection_bias"] = {"partial_hold_lr_given_sel": b, "ci95": bci, "gap_mean": float(gap.mean()),
                                 "rho_lr_gap": rg, "partial_target_lr_given_sel": plr_sel}
        print(f"  VIES DE SELECAO: parcial(CVaR_HOLD, lr | CVaR_SEL) = {b:+.3f} [{bci[0]:+.2f}, {bci[1]:+.2f}]; "
              f"gap medio HOLD-SEL = {gap.mean():+.4f}; rho(lr, gap) = {rg:+.2f}")
        print(f"                   parcial(alvo, lr | SEL) = {plr_sel:+.3f}  vs  parcial(alvo, lr | HOLD) = "
              f"{out['results'][k0]['partial_lr_given_val']:+.3f}")

    # --- CHECAGEM DE ARTEFATO -------------------------------------------------------------------
    # (i) n pareado: CVaR de validacao e do alvo recalculados em subamostras do MESMO tamanho m,
    #     mediadas; remove o vies de amostra finita do CVaR empirico, que depende de n e do formato
    #     da cauda (e este depende dos hiperparametros).
    # (ii) metades i.i.d.: parcial(CVaR_H2, curv | CVaR_H1) com H1, H2 metades do proprio alvo.
    #     H1 e H2 sao trocaveis e de mesmo tamanho: o valor esperado e' ZERO se nao houver
    #     artefato de estimacao. O que a curvatura "preve" aqui e' ruido/vies, nao generalizacao.
    okm = [r for r in ok if all((k + "_m") in r for k in subsets)]
    if len(okm) >= 10:
        out["artifact_checks"] = {}
        lcm = np.log10([r["curvature"] for r in okm]); nm = len(okm)
        print(f"  CHECAGEM DE ARTEFATO (n={nm}; n pareado com {args.match_reps} subamostras):")
        for key, label in targets:
            y_m = np.array([r[key + "_m"] for r in okm]); v_m = np.array([r["cvar_val_m_" + key] for r in okm])
            h1 = np.array([r[key + "_h1"] for r in okm]); h2 = np.array([r[key + "_h2"] for r in okm])
            pm = partial_spearman(y_m, lcm, [v_m])
            pm_ci = boot_ci(lambda i: partial_spearman(y_m[i], lcm[i], [v_m[i]]), nm)
            ph = partial_spearman(h2, lcm, [h1])
            ph_ci = boot_ci(lambda i: partial_spearman(h2[i], lcm[i], [h1[i]]), nm)
            out["artifact_checks"][key] = {"m": match[key][0], "partial_matched_n": pm, "ci95_matched_n": pm_ci,
                                           "partial_iid_halves": ph, "ci95_iid_halves": ph_ci}
            print(f"    {label:28s} n pareado (m={match[key][0]}): parcial {pm:+.3f} [{pm_ci[0]:+.2f}, {pm_ci[1]:+.2f}]"
                  f"   |  metades i.i.d.: parcial(H2, curv | H1) {ph:+.3f} [{ph_ci[0]:+.2f}, {ph_ci[1]:+.2f}]")

    if args.crossfit and n >= 10:              # composicao das metades se cancela na media dos dois papeis
        okb = [r for r in ok if r.get("cvar_hold_B", 1e3) < 1e3 and r.get("cvar_sel_B", 1e3) < 1e3]
        m = np.array([r in okb for r in ok])
        optA = ctrl - cv_sel                   # HOLD - SEL com a metade A escolhendo
        optB = np.array([r["cvar_hold_B"] - r["cvar_sel_B"] if r in okb else np.nan for r in ok])
        opt = ((optA + optB) / 2)[m]; lrm, lcm = lr[m], lc[m]; nm = int(m.sum())
        rng = np.random.default_rng(0)
        bm = [opt[rng.integers(0, nm, nm)].mean() for _ in range(4000)]
        rl = float(pearsonr(rankdata(lrm), rankdata(opt))[0])
        rl_ci = boot_ci(lambda i: float(pearsonr(rankdata(lrm[i]), rankdata(opt[i]))[0]), nm)
        rc = float(pearsonr(rankdata(lcm), rankdata(opt))[0])
        out["crossfit"] = {"n": nm, "optimism_mean": float(opt.mean()),
                           "optimism_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                           "composition_A_minus_B": float(np.nanmean((optA - optB) / 2)),
                           "rho_lr_optimism": rl, "rho_lr_optimism_ci95": rl_ci, "rho_curv_optimism": rc}
        # mesma semente => mesma trajetoria; o otimismo so' existe quando os dois papeis escolhem
        # epocas diferentes (com a melhor epoca = ultima nos dois, os modelos sao identicos e ele e' 0)
        dif = np.array([r["best_epoch"] != r.get("best_epoch_B", r["best_epoch"]) for r in ok])[m]
        out["crossfit"]["frac_epoch_differs"] = float(dif.mean())
        cf = out["crossfit"]
        print(f"  CROSS-FIT (n={nm}): otimismo medio do CVaR de validacao = {cf['optimism_mean']:+.4f} "
              f"[{cf['optimism_ci95'][0]:+.4f}, {cf['optimism_ci95'][1]:+.4f}]; efeito de composicao A-B = "
              f"{cf['composition_A_minus_B']:+.4f}")
        print(f"                     epoca escolhida difere entre os papeis em {dif.mean():.0%} das configuracoes"
              f"{' (sem variacao: correlacoes indefinidas)' if np.ptp(opt) == 0 else ''}")
        print(f"                     rho(lr, otimismo) = {rl:+.3f} [{rl_ci[0]:+.2f}, {rl_ci[1]:+.2f}];  "
              f"rho(curvatura, otimismo) = {rc:+.3f}")
    return out


def verdict(res):
    if not res["results"]:
        return f"EM ABERTO: so' {res['n_ok']} configuracoes validas"
    key = "cvar_test_novel" if "cvar_test_novel" in res["results"] else "cvar_test"
    r = res["results"][key]; lo, hi = r["partial_ci95"]
    alvo = ("malware do cluster novo" if res["dataset"] == "drebin_cluster" else "ataques novos") \
        if key == "cvar_test_novel" else "teste"
    if res["n_ok"] < 10 or not np.isfinite(lo):
        return f"EM ABERTO: so' {res['n_ok']} configuracoes validas (use --k >= 30)"
    if res.get("mode") == "train_loss":
        pre = f"[parada por perda de treino <= {res['stop_train_loss']}; {res['n_ok']}/{res['n_total']} atingiram"
    else:
        pre = f"[orcamento {res.get('epochs', '?')} epocas"
        if res.get("frac_truncated", 0) > 0.5:
            pre += f"; {res['frac_truncated']:.0%} truncadas: nao vale para minimos convergidos"
    pre += f"; controle = CVaR de {res.get('control', 'val')}] "
    return pre + _verdict_core(r, lo, hi, alvo)


def _verdict_core(r, lo, hi, alvo):
    p = r["partial_given_val"]; forca = "fraco" if abs(p) < 0.2 else "moderado" if abs(p) < 0.4 else "forte"
    if lo > 0:
        return (f"SUSTENTA H_G (marginal): com a validacao fixada, mais curvatura -> pior CVaR em {alvo} "
                f"(parcial {p:+.2f}, IC [{lo:+.2f},{hi:+.2f}], efeito {forca}). Antes de usar isto para "
                f"justificar o 2o objetivo, olhe a coluna |ctrl,5hp: se ela some, o sinal vem dos hiperparametros.")
    if hi < 0:
        return (f"CONTRADIZ H_G: com a validacao fixada, MAIS curvatura -> MELHOR em {alvo} "
                f"(parcial {p:+.2f}, IC [{lo:+.2f},{hi:+.2f}], efeito {forca}).")
    return (f"NAO SUSTENTA H_G: IC da parcial contem zero em {alvo} "
            f"(parcial {p:+.2f}, IC [{lo:+.2f},{hi:+.2f}]). Neste protocolo, a curvatura nao acrescenta "
            f"informacao sobre o teste alem da validacao.")


def main():
    ap = argparse.ArgumentParser(description="E0: a curvatura preve generalizacao alem da validacao?")
    ap.add_argument("--datasets", nargs="+", default=["nslkdd", "drebin"],
                    choices=["nslkdd", "drebin", "drebin_cluster"])
    ap.add_argument("--n-clusters", type=int, default=5, help="drebin_cluster: clusters de malware (k-means)")
    ap.add_argument("--cluster-fold", type=int, default=0,
                    help="drebin_cluster: cluster retido como malware NOVO (0 = o maior)")
    ap.add_argument("--save-losses", action="store_true",
                    help="grava as perdas por amostra (val e teste) de cada configuracao em results/raw/*.npz "
                         "(fora do git) -- permite reanalisar sem treinar de novo")
    ap.add_argument("--drebin", default="data/drebin215.csv")
    ap.add_argument("--nslkdd", default="data/nsl_kdd")
    ap.add_argument("--nsl-max-train", type=int, default=25000)
    ap.add_argument("--k", type=int, default=40, help="configuracoes sorteadas (as MESMAS em todos os datasets)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="teto de epocas (padrao = ProtocolConfig.epochs: o orcamento do Tune3)")
    ap.add_argument("--patience", type=int, default=10, help="padrao = ProtocolConfig.patience")
    ap.add_argument("--min-best-epoch", type=int, default=2,
                    help="configuracoes com melhor epoca abaixo disto sao ESTAGNADAS e excluidas")
    ap.add_argument("--val-split", action="store_true",
                    help="metade da validacao escolhe a epoca (SEL), a outra so' e' medida (HOLD)")
    ap.add_argument("--crossfit", action="store_true",
                    help="implica --val-split e treina cada configuracao 2x com os papeis SEL/HOLD trocados: "
                         "a media dos dois gaps cancela a diferenca de composicao entre as metades e isola o "
                         "vies de selecao (dobra o custo)")
    ap.add_argument("--stop-train-loss", type=float, default=None,
                    help="parada por perda de treino <= T (Jiang et al. 2020); a validacao nao escolhe nada")
    ap.add_argument("--match-reps", type=int, default=30,
                    help="subamostras para o CVaR com n pareado (checagem de artefato)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time()
    warnings.filterwarnings("ignore")

    configs = sample_configs(args.k, args.seed)
    modo = (f"parada por perda de treino <= {args.stop_train_loss} (teto {args.epochs} epocas)"
            if args.stop_train_loss is not None else
            f"ate' {args.epochs} epocas, early stopping na validacao (paciencia {args.patience})")
    print(f"[E0-generalizacao] {args.k} configuracoes, {modo}"
          f"{', validacao dividida SEL/HOLD' if args.val_split else ''}, device={args.device}")
    results = [run_dataset(d, configs, args) for d in args.datasets]
    print("\nVEREDITO (H_G: a curvatura preve o teste alem do que a validacao ja' mostra):")
    for res in results:
        res["verdict"] = verdict(res)
        print(f"  [{res['dataset']}] {res['verdict']}")
    losses = {res["dataset"]: res.pop("_losses", None) for res in results}   # fora do JSON
    path = save_result({"datasets": results, "configs": configs}, experiment="e0_generalization",
                       tag=args.tag, args=vars(args), started_at=t0)
    if args.save_losses:
        import os
        os.makedirs("results/raw", exist_ok=True)
        stem = os.path.splitext(os.path.basename(str(path)))[0] if path else f"e0_{int(t0)}"
        for res in results:
            L = losses.get(res["dataset"])
            if L:
                f = f"results/raw/{stem}_{res['dataset']}.npz"
                np.savez_compressed(f, **L)
                print(f"[perdas por amostra] {f}")


if __name__ == "__main__":
    main()
