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

Uso:
    python scripts/e0_generalization.py --device cuda --tag dionatan                 # NSL-KDD + DREBIN
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


def load(name, args, seed):
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
    tc = PlainTrialConfig(max_epochs=args.epochs, patience=args.patience, device=args.device,
                          gamma=0.95, seed=args.seed)
    rows = []
    for i, hp in enumerate(configs):
        r = plain_trial(hp, (Xtr, ytr, Xv, yv), tc, test_data=(Xte, yte), return_test_losses=True)
        tl = r.get("test_losses")
        row = {"cfg": i, "hparams": hp, "curvature": float(r["curvature"]),
               "cvar_val": float(r["cvar"]), "cvar_test": float(r["cvar_test"]),
               "best_epoch": int(r.get("best_epoch", -1))}
        if novel is not None and tl is not None and novel.any():
            row["cvar_test_novel"] = float(cvar(tl[novel], 0.95))
            row["cvar_test_known"] = float(cvar(tl[~novel], 0.95))
        rows.append(row)
    stalled = [r for r in rows if r["best_epoch"] < args.min_best_epoch]
    ok = [r for r in rows if np.isfinite(r["curvature"]) and 0 < r["curvature"] < 1e3
          and r["cvar_test"] < 1e3 and r["best_epoch"] >= args.min_best_epoch]
    n = len(ok)
    n_trunc = sum(r["best_epoch"] >= args.epochs - 1 for r in ok)
    frac_trunc = n_trunc / n if n else float("nan")
    lc = np.log10([r["curvature"] for r in ok]); cv = np.array([r["cvar_val"] for r in ok])
    lr = np.log10([r["hparams"]["learning_rate"] for r in ok])
    lwd = np.log10([r["hparams"]["weight_decay"] for r in ok])

    targets = [("cvar_test", "teste (todo)")]
    if ok and "cvar_test_novel" in ok[0]:
        targets += [("cvar_test_novel", "teste: ataques NOVOS"), ("cvar_test_known", "teste: ataques conhecidos")]

    out = {"dataset": name, "epochs": args.epochs, "n_ok": n, "n_total": len(rows), "n_stalled": len(stalled),
           "n_truncated": n_trunc, "frac_truncated": frac_trunc, "rows": rows, "results": {}}
    print(f"\n[{name}] {n}/{len(rows)} configuracoes validas "
          f"({len(stalled)} estagnadas excluidas; {n_trunc} truncadas = {frac_trunc:.0%} das validas)")
    if n and frac_trunc > 0.5:
        print(f"  NOTA: em {frac_trunc:.0%} das configuracoes a melhor epoca foi a ULTIMA (early stopping "
              f"nao disparou). O resultado vale para o orcamento de {args.epochs} epocas, nao para "
              f"minimos convergidos.")
    print(f"  {'alvo':28s} {'rho(curv,alvo)':>15} {'parcial|val':>12} {'IC95':>18} {'parcial|val,lr,wd':>18}")
    for key, label in targets:
        y = np.array([r[key] for r in ok])
        raw = float(pearsonr(rankdata(lc), rankdata(y))[0]) if n >= 4 else float("nan")
        p1 = partial_spearman(y, lc, [cv]) if n >= 6 else float("nan")
        ci = boot_ci(lambda idx: partial_spearman(y[idx], lc[idx], [cv[idx]]), n) if n >= 10 else (float("nan"),) * 2
        p2 = partial_spearman(y, lc, [cv, lr, lwd]) if n >= 8 else float("nan")
        out["results"][key] = {"rho_raw": raw, "partial_given_val": p1, "partial_ci95": ci,
                               "partial_given_val_lr_wd": p2}
        print(f"  {label:28s} {raw:>+15.3f} {p1:>+12.3f}  [{ci[0]:+.2f}, {ci[1]:+.2f}] {p2:>+18.3f}")
    rv = float(pearsonr(rankdata(lc), rankdata(cv))[0]) if n >= 4 else float("nan")
    out["rho_curv_val"] = rv
    print(f"  (referencia) rho(curvatura, CVaR de validacao) = {rv:+.3f}")
    return out


def verdict(res):
    key = "cvar_test_novel" if "cvar_test_novel" in res["results"] else "cvar_test"
    r = res["results"][key]; lo, hi = r["partial_ci95"]
    alvo = "ataques novos" if key == "cvar_test_novel" else "teste"
    if res["n_ok"] < 10 or not np.isfinite(lo):
        return f"EM ABERTO: so' {res['n_ok']} configuracoes validas (use --k >= 30)"
    pre = f"[orcamento {res.get('epochs', '?')} epocas"
    if res.get("frac_truncated", 0) > 0.5:
        pre += f"; {res['frac_truncated']:.0%} truncadas: nao vale para minimos convergidos"
    return pre + "] " + _verdict_core(r, lo, hi, alvo)


def _verdict_core(r, lo, hi, alvo):
    if lo > 0:
        return (f"SUSTENTA H_G: com a validacao fixada, mais curvatura -> pior CVaR em {alvo} "
                f"(parcial {r['partial_given_val']:+.2f}, IC [{lo:+.2f},{hi:+.2f}]). A curvatura carrega "
                f"informacao de generalizacao que a validacao nao carrega: o 2o objetivo e' justificado.")
    if hi < 0:
        return (f"CONTRADIZ H_G: com a validacao fixada, MAIS curvatura -> MELHOR em {alvo} "
                f"(parcial {r['partial_given_val']:+.2f}, IC [{lo:+.2f},{hi:+.2f}]).")
    return (f"NAO SUSTENTA H_G: IC da parcial contem zero em {alvo} "
            f"(parcial {r['partial_given_val']:+.2f}, IC [{lo:+.2f},{hi:+.2f}]). Alem do que a validacao "
            f"ja' mostra, a curvatura nao acrescenta informacao -- o 2o objetivo e' redundante com o 1o.")


def main():
    ap = argparse.ArgumentParser(description="E0: a curvatura preve generalizacao alem da validacao?")
    ap.add_argument("--datasets", nargs="+", default=["nslkdd", "drebin"], choices=["nslkdd", "drebin"])
    ap.add_argument("--drebin", default="data/drebin215.csv")
    ap.add_argument("--nslkdd", default="data/nsl_kdd")
    ap.add_argument("--nsl-max-train", type=int, default=25000)
    ap.add_argument("--k", type=int, default=40, help="configuracoes sorteadas (as MESMAS em todos os datasets)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="teto de epocas (padrao = ProtocolConfig.epochs: o orcamento do Tune3)")
    ap.add_argument("--patience", type=int, default=10, help="padrao = ProtocolConfig.patience")
    ap.add_argument("--min-best-epoch", type=int, default=2,
                    help="configuracoes com melhor epoca abaixo disto sao ESTAGNADAS e excluidas")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time()
    warnings.filterwarnings("ignore")

    configs = sample_configs(args.k, args.seed)
    print(f"[E0-generalizacao] {args.k} configuracoes, ate' {args.epochs} epocas "
          f"(early stopping na validacao, paciencia {args.patience}), device={args.device}")
    results = [run_dataset(d, configs, args) for d in args.datasets]
    print("\nVEREDITO (H_G: a curvatura preve o teste alem do que a validacao ja' mostra):")
    for res in results:
        res["verdict"] = verdict(res)
        print(f"  [{res['dataset']}] {res['verdict']}")
    save_result({"datasets": results, "configs": configs}, experiment="e0_generalization",
                tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
