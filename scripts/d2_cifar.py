#!/usr/bin/env python
# scripts/d2_cifar.py
"""
ETAPA D2 -- controle positivo no CIFAR-10 (pre-registrada em claude/TUNE3_PREREGISTRO_FASE_DIRECAO.md).

Pergunta: a nossa medida de curvatura funciona onde a literatura diz que a relacao existe?
Grade 3^5 = 243 CNNs pequenas (sem batch-norm), cada uma treinada ate' a perda de treino <= 0,01
(criterio de Jiang et al., 2020), sem aumento de dados. Em cada modelo final:
  - Tr(H^2) (Hutchinson, M=20) e sharpness ADAPTATIVA de pior caso (rho = 0,1 primaria; 0,5
    secundaria), no mesmo lote de treino;
  - erro e CVaR na validacao, no teste do CIFAR-10 (i.i.d.) e no CIFAR-10.1 (deslocado).

Analise (A) -- replicacao de Jiang et al.: tau de Kendall entre a medida e o gap de generalizacao
  (erro de teste CIFAR-10 - erro de treino), e o Kendall granulado Psi (um fator por vez).
  POSITIVO se o IC95 do tau ficar acima de zero, Psi > 0 E pelo menos 3 dos 5 fatores com psi_i
  significativamente positivo (teste do sinal sobre os grupos, p < 0,05). Uma medida que so'
  acompanha a profundidade passaria nas duas primeiras condicoes.
Analise (B) -- a pergunta do Tune3: parcial rho(medida, CVaR no CIFAR-10.1 | CVaR_val, profundidade,
  largura). POSITIVO se o IC95 ficar acima de zero. Tambem: so' | val, | val + 5 hiperparametros, e a
  versao com erro de classificacao.

Retomavel: cada configuracao concluida e' gravada em results/raw/d2_<tag>/ (fora do git); rodar de
novo com a mesma --tag pula as prontas. O JSON final vai para results/d2_cifar/.

Uso:
    python scripts/download_cifar.py                                   # uma vez
    python scripts/d2_cifar.py --configs calib --device cuda --tag dionatan-calib
    python scripts/d2_cifar.py --configs grid  --device cuda --tag dionatan-grid
"""
from __future__ import annotations

import argparse, itertools, json, os, time, warnings
import numpy as np

from tune3.baselines.plain_trial import plain_trial, PlainTrialConfig
from tune3.core.objectives import cvar
from tune3.curvature import adaptive_sharpness
from tune3.data.cifar import CIFARLoader, CIFARConfig
from tune3.experiments.generalization_measures import kendall_tau, granulated_kendall, replication_positive
from tune3.experiments.results_io import save_result
from tune3.experiments.stats import partial_spearman

LEVELS = {"n_layers": [2, 3, 4], "hidden_dim": [32, 64, 128],
          "learning_rate": [0.003, 0.01, 0.03], "weight_decay": [0.0, 1e-4, 5e-4],
          "dropout": [0.0, 0.25, 0.5]}
FACTORS = list(LEVELS)
# calibracao: cantos e centro da grade -- inclui a combinacao mais lenta (rede rasa e estreita,
# lr baixo, wd e dropout altos) e a mais propensa a divergir (profunda, larga, lr alto, sem wd)
CALIB = [(2, 32, 0.003, 5e-4, 0.5), (2, 32, 0.03, 0.0, 0.0), (4, 128, 0.03, 0.0, 0.0),
         (4, 128, 0.003, 5e-4, 0.5), (3, 64, 0.01, 1e-4, 0.25), (4, 32, 0.03, 5e-4, 0.0),
         (2, 128, 0.003, 0.0, 0.5), (3, 128, 0.01, 5e-4, 0.25), (4, 64, 0.003, 1e-4, 0.0),
         (2, 64, 0.01, 0.0, 0.25)]


def grid_configs():
    return [dict(zip(FACTORS, v)) for v in itertools.product(*[LEVELS[f] for f in FACTORS])]


def calib_configs():
    return [dict(zip(FACTORS, v)) for v in CALIB]


def boot_ci(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        v = fn(rng.integers(0, n, n))
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < B // 2:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run_config(i, hp, data, test, novel, args):
    tc = PlainTrialConfig(max_epochs=args.epochs, batch_size=args.batch_size, arch="cnn",
                          device=args.device, seed=args.seed, gamma=0.95,
                          stop_train_loss=args.stop_train_loss, curvature_probes=args.probes,
                          curvature_batch=args.curv_batch, use_class_weight=False)

    def extra(model, xb, yb, crit):
        lf = lambda m: crit(m(xb), yb)
        return {f"sharp_{r:g}": adaptive_sharpness(model, lf, rho=r, steps=args.sharp_steps)
                for r in args.rho}

    t0 = time.time()
    r = plain_trial(hp, data, tc, test_data=test, return_test_losses=True, return_val_losses=True,
                    return_errors=True, extra_measures=extra)
    tl, ok = r["test_losses"], r["test_correct"]
    row = {"cfg": i, "hparams": hp, "reached": bool(r["reached_train_loss"]),
           "epochs_run": int(r["epochs_run"]), "seconds": round(time.time() - t0, 1),
           "train_loss_final": float(r["train_loss_final"]), "curvature": float(r["curvature"]),
           "train_error": float(r["train_error"]), "val_error": float(r["val_error"]),
           "cvar_val": float(r["cvar"]),
           "test_error_c10": float(1 - ok[~novel].mean()), "test_error_c101": float(1 - ok[novel].mean()),
           "cvar_c10": float(cvar(tl[~novel], 0.95)), "cvar_c101": float(cvar(tl[novel], 0.95)),
           "loss_c10": float(tl[~novel].mean()), "loss_c101": float(tl[novel].mean())}
    for rr in args.rho:
        row[f"sharp_{rr:g}"] = float(r[f"sharp_{rr:g}"])
    row["gap_error"] = row["test_error_c10"] - row["train_error"]
    row["gap_loss"] = row["loss_c10"] - row["train_loss_final"]
    losses = {"val": np.asarray(r["val_losses"], np.float16), "test": np.asarray(tl, np.float16),
              "test_correct": np.asarray(ok, bool)}
    return row, losses


def analyze(rows, args):
    ok = [r for r in rows if r["reached"] and np.isfinite(r["curvature"]) and r["curvature"] > 0
          and r["curvature"] != 1e3 and all(np.isfinite(r[f"sharp_{x:g}"]) and r[f"sharp_{x:g}"] > 0
                                            for x in args.rho)]
    for r in ok:
        r["log_curv"] = float(np.log10(r["curvature"]))
        for x in args.rho:
            r[f"log_sharp_{x:g}"] = float(np.log10(r[f"sharp_{x:g}"]))
    n = len(ok); out = {"n_ok": n, "n_total": len(rows), "measures": {}}
    print(f"\n[D2] {n}/{len(rows)} modelos atingiram perda de treino <= {args.stop_train_loss} "
          f"(e tem medidas validas)")
    if n < 10:
        print("  amostra pequena demais para analisar"); return out
    measures = ["log_curv"] + [f"log_sharp_{x:g}" for x in args.rho]
    A = lambda k: np.array([r[k] for r in ok], float)
    H = lambda k: np.array([r["hparams"][k] for r in ok], float)
    depth, width = H("n_layers"), H("hidden_dim")
    lr, wd, do = np.log10(H("learning_rate")), np.log10(H("weight_decay") + 1e-6), H("dropout")
    gap, y101, v, e101, ev = A("gap_error"), A("cvar_c101"), A("cvar_val"), A("test_error_c101"), A("val_error")
    full = len({tuple(r["hparams"].values()) for r in ok}) == len(grid_configs())
    print(f"  {'medida':16s} | (A) tau(gap)        IC95        Psi    | (B) parcial|val,arq      IC95       |val   |val,5hp  erro|val,arq")
    for m in measures:
        x = A(m)
        tau = kendall_tau(x, gap); tci = boot_ci(lambda i: kendall_tau(x[i], gap[i]), n)
        gk = granulated_kendall(ok, m, "gap_error", FACTORS)
        pb = partial_spearman(y101, x, [v, depth, width])
        pci = boot_ci(lambda i: partial_spearman(y101[i], x[i], [v[i], depth[i], width[i]]), n)
        p_val = partial_spearman(y101, x, [v]); p5 = partial_spearman(y101, x, [v, lr, wd, do, depth, width])
        pe = partial_spearman(e101, x, [ev, depth, width])
        A_pos = replication_positive(tci, gk); B_pos = bool(pci[0] > 0)
        out["measures"][m] = {"tau_gap": tau, "tau_ci95": tci, "Psi": gk["Psi"], "psi": gk["per_factor"],
                              "partial_B": pb, "partial_B_ci95": pci, "partial_val": p_val,
                              "partial_5hp": p5, "partial_error_arch": pe,
                              "A_positive": A_pos, "B_positive": B_pos}
        print(f"  {m:16s} |   {tau:+.3f}  [{tci[0]:+.2f}, {tci[1]:+.2f}]  {gk['Psi']:+.3f} |      {pb:+.3f}   "
              f"[{pci[0]:+.2f}, {pci[1]:+.2f}]  {p_val:+.3f}   {p5:+.3f}    {pe:+.3f}")
        print("  " + " " * 16 + "   psi por fator: " + "  ".join(
            f"{f}={gk['per_factor'][f]['psi']:+.2f}{'*' if gk['per_factor'][f]['p_sign'] < 0.05 and gk['per_factor'][f]['psi'] > 0 else ''}"
            for f in FACTORS)
              + f"   -> (A) {'POSITIVO' if A_pos else 'nao'} | (B) {'POSITIVO' if B_pos else 'nao'}")
    if not full:
        print("  NOTA: grade incompleta (modelos que nao atingiram o limiar ou configuracoes fora da grade);"
              " o Psi usa so' os grupos com >= 2 modelos validos.")
    prim = out["measures"]["log_curv"]; inv = out["measures"][f"log_sharp_{args.rho[0]:g}"]
    print("\nLEITURA PELA TABELA PRE-REGISTRADA (Tr(H^2) / sharpness adaptativa rho=%g):" % args.rho[0])
    if prim["A_positive"] and prim["B_positive"]:
        v_ = "(A)+ (B)+ com Tr(H^2): a curvatura funciona aqui -> artigo de contraste (quando ajuda)"
    elif prim["A_positive"]:
        v_ = "(A)+ (B) nulo com Tr(H^2): acompanha o gap mas nao acrescenta nada alem da validacao"
    elif inv["A_positive"]:
        v_ = "(A) nulo com Tr(H^2), positivo com a adaptativa: o problema e' a metrica"
    else:
        v_ = "(A) nulo com as duas: instrumento ou hipotese falham ate' no cenario favoravel"
    print("  " + v_)
    out["reading"] = v_
    return out


def main():
    ap = argparse.ArgumentParser(description="D2: controle positivo no CIFAR-10.")
    ap.add_argument("--configs", choices=["grid", "calib"], default="calib")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--stop-train-loss", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=400, help="teto de epocas")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--curv-batch", type=int, default=512, help="lote de treino das medidas de curvatura")
    ap.add_argument("--probes", type=int, default=20)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.1, 0.5], help="o primeiro e' o primario")
    ap.add_argument("--sharp-steps", type=int, default=10)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None, help="(teste) so' as primeiras N configuracoes")
    args = ap.parse_args(); t_start = time.time()
    warnings.filterwarnings("ignore")

    m = CIFARLoader(CIFARConfig(n_train=args.n_train, n_val=args.n_val, random_state=args.seed)).load_splits_with_meta()
    data = (m["X_train"], m["y_train"], m["X_val"], m["y_val"])
    test = (m["X_test"], m["y_test"]); novel = m["test_novel_mask"]
    configs = grid_configs() if args.configs == "grid" else calib_configs()
    if args.limit:
        configs = configs[: args.limit]
    prog = os.path.join("results", "raw", f"d2_{args.tag or 'sem_tag'}_{args.configs}")
    os.makedirs(prog, exist_ok=True)
    print(f"[D2] {len(configs)} configuracoes ({args.configs}); treino {len(data[1])}, val {len(data[3])}, "
          f"teste CIFAR-10 {int((~novel).sum())} + CIFAR-10.1 {int(novel.sum())}; parada em perda de treino "
          f"<= {args.stop_train_loss} (teto {args.epochs} epocas); device={args.device}; progresso em {prog}")

    rows = []
    for i, hp in enumerate(configs):
        fj, fz = os.path.join(prog, f"cfg_{i:03d}.json"), os.path.join(prog, f"cfg_{i:03d}.npz")
        if os.path.exists(fj) and os.path.exists(fz):
            rows.append(json.load(open(fj))); continue
        row, losses = run_config(i, hp, data, test, novel, args)
        np.savez_compressed(fz, **losses)
        json.dump(row, open(fj, "w"))
        rows.append(row)
        print(f"  [{i + 1:3d}/{len(configs)}] prof={hp['n_layers']} larg={hp['hidden_dim']:3d} lr={hp['learning_rate']:<6g} "
              f"wd={hp['weight_decay']:<6g} drop={hp['dropout']:<4g} -> {'atingiu' if row['reached'] else 'NAO atingiu'} "
              f"em {row['epochs_run']:3d} ep ({row['seconds']:6.1f} s); erro teste {row['test_error_c10']:.3f} / "
              f"10.1 {row['test_error_c101']:.3f}; Tr(H2) {row['curvature']:.3g}; sharp {row[f'sharp_{args.rho[0]:g}']:.3g}",
              flush=True)

    secs = [r["seconds"] for r in rows]
    print(f"\n[tempo] mediana {np.median(secs):.0f} s por modelo; total {sum(secs) / 3600:.2f} h; "
          f"atingiram {sum(r['reached'] for r in rows)}/{len(rows)}")
    res = analyze(rows, args)
    path = save_result({"rows": rows, "analysis": res, "levels": LEVELS,
                        "cifar": {"n_train": args.n_train, "n_val": args.n_val,
                                  "channel_mean": m["channel_mean"], "channel_std": m["channel_std"]}},
                       experiment="d2_cifar", tag=args.tag, args=vars(args), started_at=t_start)


if __name__ == "__main__":
    main()
