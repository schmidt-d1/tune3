#!/usr/bin/env python
# scripts/e0_within_config.py
"""
Cenario 3 (out/2026): MESMA CONFIGURACAO, SEMENTES DIFERENTES -- o teste que separa geometria de arquitetura.

Por que. No E0, configuracoes diferentes tem hiperparametros diferentes, e a curvatura acompanha a
arquitetura (R^2 ~ 0,8 sobre os 5 hiperparametros); a parcial "| val, 5 hp" e' um ajuste estatistico. Aqui
os hiperparametros sao fixados POR CONSTRUCAO: K configuracoes (escolhidas de uma rodada do E0, estratificadas
pela curvatura) sao treinadas cada uma com S sementes de treino, na MESMA divisao de dados do E0. Dentro de uma
configuracao, a unica fonte de variacao da geometria e' o sorteio do treino (inicializacao, ordem dos lotes,
dropout). Pergunta: dentro da configuracao, modelos mais planos vao melhor no alvo deslocado, dada a validacao?

Regra (fixada antes de rodar; "Cenario 3" em claude/TUNE3_PREREGISTRO_FASE_DIRECAO.md):
  parcial INTRA-configuracao rho_w(log medida, CVaR_alvo | CVaR_val) -- valores centrados na media da sua
  configuracao, agrupados; IC95 por bootstrap de sementes DENTRO de cada configuracao (B reamostragens).
  IC inteiramente > 0 -> SUSTENTA H_G (intra); inteiramente < 0 -> CONTRADIZ; senao -> nulo.
  Primaria: Tr(H^2). Secundarias: sharpness adaptativa 0,1 (invariante a reescala) e 0,5, lambda_max.
  Tambem: combinacao por configuracao (Fisher z, n = S cada; I^2), parcial ENTRE configuracoes (medias por
  configuracao, n = K) para comparar, ICC (fracao da variancia de cada medida que e' entre configuracoes) e
  confiabilidade do estimador (curv2).

Retomavel: cada treino concluido e' gravado em results/raw/within_<tag>/ (fora do git); rodar de novo continua.

Uso:
    python scripts/e0_within_config.py --from "results/e0_generalization/*krono-s0.json" --datasets kronodroid \\
        --n-configs 15 --n-seeds 20 --stop-train-loss 0.10 --epochs 1000 --measures sharp lmax curv2 --device cuda --tag dionatan-within-krono
    python scripts/e0_within_config.py --analyze results/e0_within_config/<arquivo>.json      (so' reanalisa)
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os, pathlib, time, warnings
import numpy as np
from scipy.stats import spearmanr

from tune3.baselines.plain_trial import PlainTrialConfig, plain_trial
from tune3.core.objectives import cvar
from tune3.experiments.results_io import save_result
from tune3.experiments.stats import fisher_combine, partial_spearman

MEASURES = [("curvature", "log Tr(H2)"), ("sharp_0.1", "log sharp 0.1"), ("sharp_0.5", "log sharp 0.5"),
            ("lmax", "log lambda_max"), ("curv2", "log Tr(H2) 2a est.")]


def _e0():
    p = pathlib.Path(__file__).resolve().parent / "e0_generalization.py"
    spec = importlib.util.spec_from_file_location("e0g", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def select_configs(e0_json, n_configs, max_epoch_frac=0.8):
    """Configuracoes da rodada do E0 que atingiram T com folga (epocas <= frac do teto), validas,
    estratificadas pelos quantis de log Tr(H^2)."""
    J = json.load(open(e0_json)); d = J["payload"]["datasets"][0]; cap = J["meta"]["args"].get("epochs", 1000)
    ok = [r for r in d["rows"] if r.get("reached") and np.isfinite(r["curvature"]) and 0 < r["curvature"] != 1e3
          and r["cvar_test"] != 1e3 and r.get("epochs_run", cap) <= max_epoch_frac * cap]
    ok.sort(key=lambda r: r["curvature"])
    pos = [int(round((i + 0.5) / n_configs * (len(ok) - 1))) for i in range(n_configs)]
    sel = [ok[i] for i in sorted(set(pos))]
    return [{"cfg": r["cfg"], "hparams": r["hparams"], "e0_curvature": r["curvature"], "e0_cvar_val": r["cvar_val"],
             "e0_cvar_test": r["cvar_test"], "e0_epochs_run": r.get("epochs_run")} for r in sel], d["dataset"], J["meta"]["args"], len(ok)


def _target_key(dataset):
    return "cvar_test" if dataset == "kronodroid" else "cvar_test_novel"


def analyze(rows, dataset, B=1000, seed=0):
    rng = np.random.default_rng(seed); tkey = _target_key(dataset)
    ok = [r for r in rows if r.get("reached") and np.isfinite(r["curvature"]) and 0 < r["curvature"] != 1e3 and np.isfinite(r["cvar_test"]) and r["cvar_test"] != 1e3]
    if tkey not in ok[0]:
        tkey = "cvar_test"
    cfgs = sorted({r["cfg"] for r in ok}); g = np.array([cfgs.index(r["cfg"]) for r in ok])
    y = np.array([r[tkey] for r in ok]); v = np.array([r["cvar_val"] for r in ok]); n = len(ok); K = len(cfgs)
    sizes = np.bincount(g, minlength=K)
    out = {"target": tkey, "n_runs": n, "n_configs": K, "seeds_per_config": {"min": int(sizes.min()), "max": int(sizes.max())},
           "n_not_reached": int(sum(1 for r in rows if not r.get("reached"))), "measures": {}}

    def demean(a, gg):
        m = np.zeros(gg.max() + 1); cnt = np.bincount(gg, minlength=len(m)); np.add.at(m, gg, a)
        return a - (m / np.maximum(cnt, 1))[gg]

    def within_partial(yy, xx, vv, gg):
        return partial_spearman(demean(yy, gg), demean(xx, gg), [demean(vv, gg)])

    def icc1(x, gg):
        """ICC(1), ANOVA de um fator: fracao da variancia que e' ENTRE configuracoes."""
        k = len(np.unique(gg)); nn = len(x); means = np.array([x[gg == i].mean() for i in range(k)]); ns = np.array([(gg == i).sum() for i in range(k)])
        msb = (ns * (means - x.mean()) ** 2).sum() / max(k - 1, 1)
        msw = sum(((x[gg == i] - means[i]) ** 2).sum() for i in range(k)) / max(nn - k, 1)
        k0 = (nn - (ns ** 2).sum() / nn) / max(k - 1, 1)
        return float((msb - msw) / (msb + (k0 - 1) * msw)) if (msb + (k0 - 1) * msw) > 0 else float("nan")

    out["rho_val_target_within"] = float(spearmanr(demean(v, g), demean(y, g))[0])
    out["sd_within"] = {"target": float(np.mean([y[g == i].std() for i in range(K)])), "val": float(np.mean([v[g == i].std() for i in range(K)]))}
    print(f"\n  alvo = {tkey}; {n} treinos validos em {K} configuracoes ({sizes.min()}-{sizes.max()} sementes cada; "
          f"{out['n_not_reached']} nao atingiram T); rho intra(val, alvo) = {out['rho_val_target_within']:+.3f}; "
          f"desvio intra-config do alvo {out['sd_within']['target']:.3f}, da validacao {out['sd_within']['val']:.3f}")
    print(f"  {'medida':20s} {'ICC':>5} {'sd intra(log)':>13} {'rho_w(med,alvo)':>15} {'parcial INTRA|val':>17} {'IC95 (boot sementes)':>21} {'Fisher RE [IC]':>22} {'I2':>4} {'ENTRE (n=K)':>11}  leitura")
    for key, label in MEASURES:
        if key not in ok[0]:
            continue
        m = np.array([r.get(key, np.nan) for r in ok], float); okm = np.isfinite(m) & (m > 0)
        if okm.sum() < 20:
            continue
        x = np.log10(m[okm]); yy, vv, gg_ = y[okm], v[okm], g[okm]
        _, gg = np.unique(gg_, return_inverse=True)
        pw = within_partial(yy, x, vv, gg); rw = float(spearmanr(demean(x, gg), demean(yy, gg))[0])
        # bootstrap de sementes dentro de cada configuracao
        groups = [np.flatnonzero(gg == i) for i in range(gg.max() + 1)]; bs = []
        for _ in range(B):
            idx = np.concatenate([grp[rng.integers(0, len(grp), len(grp))] for grp in groups])
            val_ = within_partial(yy[idx], x[idx], vv[idx], gg[idx])
            if np.isfinite(val_):
                bs.append(val_)
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) >= B // 2 else [float("nan")] * 2
        # por configuracao (Fisher) e entre configuracoes
        rs, ns = [], []
        for grp in groups:
            if len(grp) >= 6:
                rc = partial_spearman(yy[grp], x[grp], [vv[grp]])
                if np.isfinite(rc):
                    rs.append(rc); ns.append(len(grp))
        fc = fisher_combine(rs, ns, 1) if len(rs) >= 2 else {"r_re": float("nan"), "ci95_re": [float("nan")] * 2, "I2": float("nan")}
        means = lambda a: np.array([a[grp].mean() for grp in groups])
        between = partial_spearman(means(yy), means(x), [means(vv)]) if len(groups) >= 6 else float("nan")
        sdw = float(np.mean([x[grp].std() for grp in groups]))
        lead = "SUSTENTA H_G (intra-config)" if ci[0] > 0 else "CONTRADIZ (intra-config)" if ci[1] < 0 else "nulo (intra-config)"
        out["measures"][key] = {"label": label, "n": int(okm.sum()), "icc_between": icc1(x, gg), "sd_within_log": sdw, "rho_within": rw,
                                "partial_within": pw, "ci95_within": ci, "fisher_re": fc["r_re"], "fisher_ci95_re": fc["ci95_re"], "I2": fc["I2"],
                                "n_configs_fisher": len(rs), "partial_between": between, "verdict": lead}
        print(f"  {label:20s} {icc1(x, gg):>5.2f} {sdw:>13.3f} {rw:>+15.3f} {pw:>+17.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]        "
              f"{fc['r_re']:>+6.3f} [{fc['ci95_re'][0]:+.2f}, {fc['ci95_re'][1]:+.2f}] {fc['I2']:>4.0%} {between:>+11.3f}  {lead}")
    if "curv2" in ok[0]:
        c1 = np.log10([r["curvature"] for r in ok]); c2 = np.array([r.get("curv2", np.nan) for r in ok], float); okc = np.isfinite(c2) & (c2 > 0)
        out["reliability_curv2"] = {"spearman_total": float(spearmanr(c1[okc], np.log10(c2[okc]))[0]),
                                    "spearman_within": float(spearmanr(demean(c1[okc], g[okc]), demean(np.log10(c2[okc]), g[okc]))[0])}
        print(f"  confiabilidade do estimador (Tr(H2) vs 2a estimativa): Spearman total {out['reliability_curv2']['spearman_total']:+.3f}; "
              f"intra-configuracao {out['reliability_curv2']['spearman_within']:+.3f}")
    prim = out["measures"].get("curvature", {}); ci_p = prim.get("ci95_within", [float("nan")] * 2)
    out["verdict"] = (f"Tr(H2) intra-configuracao: {prim.get('verdict', 'EM ABERTO')} "
                      f"(parcial {prim.get('partial_within', float('nan')):+.3f}, IC95 [{ci_p[0]:+.2f}, {ci_p[1]:+.2f}])")
    print("\n  VEREDITO (regra pre-registrada, Cenario 3):", out["verdict"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Mesma configuracao, sementes diferentes: a geometria preve o alvo deslocado dentro da configuracao?")
    ap.add_argument("--from", dest="src", default=None, help="JSON do E0 de onde vem a divisao de dados e as configuracoes")
    ap.add_argument("--analyze", default=None, help="so' reanalisa um JSON deste script")
    ap.add_argument("--datasets", nargs="+", default=["kronodroid"], choices=["nslkdd", "drebin", "drebin_cluster", "kronodroid"])
    ap.add_argument("--n-configs", type=int, default=15); ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=1000, help="semente de treino = seed-base + s (distinta das do E0)")
    ap.add_argument("--epochs", type=int, default=1000); ap.add_argument("--stop-train-loss", type=float, default=0.10)
    ap.add_argument("--measures", nargs="*", default=["sharp", "lmax", "curv2"], choices=["sharp", "lmax", "curv2", "curvb"])
    ap.add_argument("--sharp-rho", type=float, nargs="+", default=[0.1, 0.5]); ap.add_argument("--sharp-steps", type=int, default=10)
    ap.add_argument("--lmax-iters", type=int, default=20)
    ap.add_argument("--B", type=int, default=1000)
    # argumentos de dados (iguais ao E0)
    ap.add_argument("--n-clusters", type=int, default=5); ap.add_argument("--cluster-fold", type=int, default=0); ap.add_argument("--cluster-seed", type=int, default=0)
    ap.add_argument("--drebin", default="data/drebin215.csv"); ap.add_argument("--nslkdd", default="data/nsl_kdd"); ap.add_argument("--nsl-max-train", type=int, default=25000)
    ap.add_argument("--kronodroid", default="data/kronodroid/real_device"); ap.add_argument("--krono-cut", type=int, default=2014); ap.add_argument("--krono-far", type=int, default=2018)
    ap.add_argument("--krono-max-train", type=int, default=25000); ap.add_argument("--krono-n-val", type=int, default=5000); ap.add_argument("--krono-natural-ratio", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"]); ap.add_argument("--seed", type=int, default=0, help="semente da DIVISAO de dados (a do E0 de origem)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(); t0 = time.time(); warnings.filterwarnings("ignore")

    if args.analyze:
        J = json.load(open(args.analyze)); P = J["payload"]
        print(f"[reanalise] {args.analyze}")
        res = analyze(P["rows"], P["dataset"], B=args.B)
        save_result({"dataset": P["dataset"], "selected": P["selected"], "source": P.get("source"), "rows": P["rows"], "analysis": res},
                    experiment="e0_within_config", tag=(args.tag or "reanalise"), args=vars(args), started_at=t0)
        return

    src = sorted(glob.glob(args.src))[0]
    selected, dataset, src_args, n_pool = select_configs(src, args.n_configs)
    if dataset != args.datasets[0]:
        raise SystemExit(f"o JSON de origem e' de {dataset}, mas --datasets diz {args.datasets[0]}")
    e0 = _e0()
    Xtr, ytr, Xv, yv, Xte, yte, novel = e0.load(dataset, args, args.seed)
    print(f"[within-config] {dataset}: {len(selected)} configuracoes (de {n_pool} elegiveis em {os.path.basename(src)}) x {args.n_seeds} sementes; "
          f"treino {Xtr.shape}, val {Xv.shape}, teste {Xte.shape}; parada em perda <= {args.stop_train_loss} (teto {args.epochs}); device={args.device}")
    import torch
    from tune3.curvature.measures import make_extra_measures
    second = None
    if "curvb" in args.measures:
        rng2 = np.random.default_rng(args.seed + 3000); j = rng2.choice(len(ytr), min(4096, len(ytr)), replace=False)
        second = (torch.as_tensor(Xtr[j], dtype=torch.float32, device=args.device), torch.as_tensor(ytr[j], dtype=torch.long, device=args.device))
    extra_keys = ([f"sharp_{r_:g}" for r_ in args.sharp_rho] if "sharp" in args.measures else []) + [k for k in ("lmax", "curv2", "curvb") if k in args.measures]
    prog = os.path.join("results", "raw", f"within_{args.tag or 'sem_tag'}"); os.makedirs(prog, exist_ok=True)
    rows, L_val, L_test = [], [], []
    for c in selected:
        for s in range(args.n_seeds):
            fj = os.path.join(prog, f"cfg_{c['cfg']:03d}_seed_{s:02d}.json"); fz = fj.replace(".json", ".npz")
            if os.path.exists(fj) and os.path.exists(fz):
                row = json.load(open(fj)); Z = np.load(fz)
            else:
                tseed = args.seed_base + s
                extra = make_extra_measures(args.measures, sharp_rho=args.sharp_rho, sharp_steps=args.sharp_steps, lmax_iters=args.lmax_iters,
                                            probes=5, second_batch=second, seed=tseed)
                tc = PlainTrialConfig(max_epochs=args.epochs, device=args.device, gamma=0.95, seed=tseed, stop_train_loss=args.stop_train_loss)
                t1 = time.time()
                r = plain_trial(c["hparams"], (Xtr, ytr, Xv, yv), tc, test_data=(Xte, yte), return_test_losses=True, return_val_losses=True, extra_measures=extra)
                tl = np.asarray(r["test_losses"], float); vl = np.asarray(r["val_losses"], float)
                row = {"cfg": c["cfg"], "hparams": c["hparams"], "seed": s, "torch_seed": tseed, "reached": bool(r["reached_train_loss"]),
                       "epochs_run": int(r["epochs_run"]), "train_loss_final": float(r["train_loss_final"]), "seconds": round(time.time() - t1, 1),
                       "curvature": float(r["curvature"]), "cvar_val": float(r["cvar"]), "cvar_test": float(r["cvar_test"])}
                for k in extra_keys:
                    row[k] = float(r[k]) if np.isfinite(r.get(k, float("nan"))) else float("nan")
                if novel is not None and novel.any() and np.all(np.isfinite(tl)):
                    row["cvar_test_novel"] = float(cvar(tl[novel], 0.95))
                    row["cvar_test_known"] = float(cvar(tl[~novel], 0.95)) if (~novel).any() else float("nan")
                json.dump(row, open(fj, "w")); np.savez_compressed(fz, val=vl.astype(np.float16), test=tl.astype(np.float16)); Z = np.load(fz)
                print(f"  cfg {c['cfg']:3d} semente {s:2d}: {'ok ' if row['reached'] else 'NAO'} {row['epochs_run']:4d} ep; curv {row['curvature']:.3g}; "
                      f"CVaR val {row['cvar_val']:.3f} alvo {row.get(_target_key(dataset), row['cvar_test']):.3f}; {row['seconds']}s", flush=True)
            rows.append(row); L_val.append(Z["val"]); L_test.append(Z["test"])
    res = analyze(rows, dataset, B=args.B)
    path = save_result({"dataset": dataset, "selected": selected, "source": os.path.basename(src), "source_args": src_args, "rows": rows, "analysis": res},
                       experiment="e0_within_config", tag=args.tag, args=vars(args), started_at=t0)
    stem = os.path.splitext(os.path.basename(str(path)))[0] if path else f"within_{int(t0)}"
    np.savez_compressed(f"results/raw/{stem}_{dataset}.npz", val=np.stack(L_val), test=np.stack(L_test), y_val=np.asarray(yv), y_test=np.asarray(yte),
                        test_novel_mask=(np.asarray(novel) if novel is not None else np.zeros(len(yte), bool)),
                        cfg=np.array([r["cfg"] for r in rows]), seed=np.array([r["seed"] for r in rows]))
    print(f"[perdas por amostra] results/raw/{stem}_{dataset}.npz")


if __name__ == "__main__":
    main()
