#!/usr/bin/env python
# scripts/paper_tables.py
"""
Tabelas e figuras do artigo, geradas DIRETO dos JSONs oficiais (out/2026). Nenhum numero e' digitado a mao.

- SOURCES lista cada JSON com o commit esperado; o script FALHA se o commit do JSON nao for o listado ou se a arvore
  estava suja quando ele foi gravado (--no-check desliga, so' para desenvolvimento).
- Saidas em results/paper/: tab_*.md e tab_*.tex (booktabs), fig_*.pdf e fig_*.png, numbers.json (todos os numeros
  citados no texto, com chave estavel, + as tabelas em JSON), MANIFEST.md (saida -> fontes).
- Vereditos: importados dos scripts que os definiram (e0_measures, measures_replication, e1_offline, replication_summary),
  nao reimplementados. IC em dois niveis: mesma receita do e0_bootstrap2.py (le results/raw/*.npz quando existe; senao,
  usa o IC em dois niveis gravado no e0_measures da mesma rodada -m*, e marca a origem no MANIFEST).

Uso:
    python scripts/paper_tables.py --out results/paper --B2 1000
    python scripts/paper_tables.py --no-check --B2 200     (desenvolvimento)
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os, pathlib, sys, time
import numpy as np
from scipy.stats import spearmanr, rankdata, kendalltau

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tune3.experiments.stats import partial_spearman, fisher_combine  # noqa: E402


def _load_script(name):
    p = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- fontes oficiais
SOURCES = {
    "e0_nsl_s0": ("results/e0_generalization/*_dionatan-k600-stop-s0.json", "03e25ff"),
    "e0_nsl_s1": ("results/e0_generalization/*_dionatan-k600-stop-s1.json", "03e25ff"),
    "e0_nsl_s2": ("results/e0_generalization/*_dionatan-k600-stop-s2.json", "03e25ff"),
    "e0_krono_s0": ("results/e0_generalization/*_dionatan-krono-s0.json", "d163b0d"),
    "e0_krono_s1": ("results/e0_generalization/*_dionatan-krono-s1.json", "8f2ac06"),
    "e0_krono_s2": ("results/e0_generalization/*_dionatan-krono-s2.json", "aacfa3e"),
    "e0_krono_emu": ("results/e0_generalization/*_dionatan-krono-emu-s0.json", "e600f41"),
    "e0_krono_natural": ("results/e0_generalization/*_dionatan-krono-natural-s0.json", "e600f41"),
    "e0_nsl_m0": ("results/e0_generalization/*_dionatan-k600-stop-m0.json", "0755024"),
    "e0_krono_m0": ("results/e0_generalization/*_dionatan-krono-m0.json", "0755024"),
    "e0_krono_m1": ("results/e0_generalization/*_dionatan-krono-m1.json", "04ef921"),
    "e0_krono_m2": ("results/e0_generalization/*_dionatan-krono-m2.json", "04ef921"),
    "d2_s0": ("results/d2_cifar/*_dionatan-grid.json", "995071b"),
    "d2_s1": ("results/d2_cifar/*_dionatan-grid-s1.json", "0a7a9bd"),
    "d2_s2": ("results/d2_cifar/*_dionatan-grid-s2.json", "4225b1b"),
    "d2_followup": ("results/d2_followup/*_dionatan-grid.json", "8ce2dd4"),
    "replication_4exp": ("results/replication_summary/*_dionatan-4exp.json", "1225f5d"),
    "measures_nsl_m0": ("results/e0_measures/*_dionatan-nsl-m0.json", "bbcc376"),
    "measures_krono_m0": ("results/e0_measures/*_dionatan-krono-m0.json", "bbcc376"),
    "measures_drebin_m0": ("results/e0_measures/*_dionatan-drebin-m0.json", "bbcc376"),
    "measures_krono_m1": ("results/e0_measures/*_dionatan-krono-m1.json", "615ca2b"),
    "measures_krono_m2": ("results/e0_measures/*_dionatan-krono-m2.json", "615ca2b"),
    "measures_drebin_m1": ("results/e0_measures/*_dionatan-drebin-m1.json", "b67f44a"),
    "measures_drebin_m2": ("results/e0_measures/*_dionatan-drebin-m2.json", "b67f44a"),
    "measures_replication": ("results/measures_replication/*_dionatan.json", "b67f44a"),
    "within_krono": ("results/e0_within_config/*_dionatan-within-krono-p200.json", "a8428e4"),
    "within_drebin_f0": ("results/e0_within_config/*_dionatan-within-drebin-f0-p200.json", "a8428e4"),
    "within_drebin_f3": ("results/e0_within_config/*_dionatan-within-drebin-f3-p200.json", "11a3b1d"),
    "within_krono_m5": ("results/e0_within_config/*_dionatan-within-krono.json", "ff54e3e"),
    "within_drebin_f0_m5": ("results/e0_within_config/*_dionatan-within-drebin-f0.json", "7c5c5b7"),
    "curvature_vs_hparams": ("results/curvature_vs_hparams/*_dionatan.json", "afaceea"),
    "halves": ("results/e0_halves_baseline/*_dionatan.json", "afaceea"),
    "e1_offline": ("results/e1_offline/*_dionatan.json", "fa36555"),
    "f1_d2_drebin": ("results/f1_val_size/*_dionatan.json", "cdc4048"),
    "f1_nsl": ("results/f1_val_size/*_dionatan-nsl.json", "de05cff"),
}
for k in range(5):
    SOURCES[f"e0_drebin_s0_f{k}"] = (f"results/e0_generalization/*_dionatan-drebin-cluster-f{k}.json", "908c435")
    SOURCES[f"e0_drebin_s1_f{k}"] = (f"results/e0_generalization/*_dionatan-drebin-cluster-s1-f{k}.json", "3c01d30")
    SOURCES[f"e0_drebin_s2_f{k}"] = (f"results/e0_generalization/*_dionatan-drebin-cluster-s2-f{k}.json", "d53f976")
    SOURCES[f"e0_drebin_m0_f{k}"] = (f"results/e0_generalization/*_dionatan-drebin-cluster-m0-f{k}.json", "0755024")
    SOURCES[f"e0_drebin_m1_f{k}"] = (f"results/e0_generalization/*_dionatan-drebin-cluster-m1-f{k}.json", "0b25308")
    SOURCES[f"e0_drebin_m2_f{k}"] = (f"results/e0_generalization/*_dionatan-drebin-cluster-m2-f{k}.json", "0b25308")

# fatos dos datasets que nao estao nos JSONs (nao sao resultados): tamanhos de conjuntos publicos
DATASET_FACTS = {"cifar10_test": 10000, "cifar101_v6": 2000, "drebin215_rows": 15036, "kddtest_plus": 22544, "nsl_raw_features": 41}
GIVEN = "∣"   # barra de condicional nas celulas (o "|" quebraria a tabela Markdown)

# cenarios do E0 na ordem do artigo: (chave de cenario, rotulo, [(semente, chave da fonte)], alvo)
SCENARIOS = [
    ("nslkdd", "NSL-KDD (novel attacks)", [(0, "e0_nsl_s0"), (1, "e0_nsl_s1"), (2, "e0_nsl_s2")], "cvar_test_novel"),
    ("kronodroid", "KronoDroid (temporal)", [(0, "e0_krono_s0"), (1, "e0_krono_s1"), (2, "e0_krono_s2")], "cvar_test"),
] + [(f"drebin_f{k}", f"DREBIN cluster, fold {k}", [(s, f"e0_drebin_s{s}_f{k}") for s in range(3)], "cvar_test_novel") for k in range(5)]
HP = [("log_lr", "learning_rate"), ("log_wd", "weight_decay"), ("dropout", "dropout"), ("width", "hidden_dim"), ("depth", "n_layers")]
MEASURE_LABELS = {"curvature": "Tr(H²) M=5", "lmax": "λmax", "curv2": "Tr(H²) M=200", "curvb": "Tr(H²) M=200, 2nd batch",
                  "sharp_0.1": "sharpness ρ=0.1", "sharp_0.5": "sharpness ρ=0.5"}


def en(rule: str) -> str:
    """Traduz os rotulos de regra impressos pelos scripts (pt) para as tabelas do artigo (en)."""
    for a, b in (("NAO replicado", "not replicated"), ("nao replicado por desenho", "by design"), ("semente", "seed"), ("atrapalha", "hurts"), ("ajuda", "helps"),
                 ("misto", "mixed"), ("nulo", "null"), ("SUSTENTA H_G", "sustains"), ("SUSTENTA", "sustains"), ("CONTRADIZ", "contradicts"), (" em ", " in "), ("sem rodadas", "no runs")):
        rule = rule.replace(a, b)
    return rule


class SourceError(RuntimeError):
    pass


def resolve(root: pathlib.Path, key: str, check: bool = True):
    pat, commit = SOURCES[key]
    hits = sorted(glob.glob(str(root / pat)))
    if len(hits) != 1:
        raise SourceError(f"{key}: '{pat}' casa com {len(hits)} arquivo(s): {hits}")
    J = json.load(open(hits[0])); g = J["meta"].get("git", {})
    if check and (g.get("commit_short") != commit or g.get("dirty", True)):
        raise SourceError(f"{key}: commit {g.get('commit_short')} (esperado {commit}), dirty={g.get('dirty')} em {hits[0]}")
    return J, pathlib.Path(hits[0])


def check_sources(root: pathlib.Path, check: bool = True):
    loaded, manifest = {}, []
    for key in SOURCES:
        J, p = resolve(root, key, check); loaded[key] = (J, p)
        manifest.append((key, str(p.relative_to(root)) if str(p).startswith(str(root)) else str(p), J["meta"]["git"].get("commit_short"), J["meta"]["git"].get("dirty")))
    return loaded, manifest


# ----------------------------------------------------------------------------------------------- utilidades
def rows_ok(d):
    return [r for r in d["rows"] if r.get("reached", r.get("best_epoch", 2) >= 2) and np.isfinite(r["curvature"])
            and 0 < r["curvature"] != 1e3 and r.get("cvar_test", 0) != 1e3]


def hp_arrays(rows):
    h = lambda k: np.array([r["hparams"][k] for r in rows], float)
    return {"log_lr": np.log10(h("learning_rate")), "log_wd": np.log10(h("weight_decay") + 1e-6), "dropout": h("dropout"),
            "width": h("hidden_dim"), "depth": h("n_layers")}


def cvar_rows(L, gamma=0.95):
    """Identico ao e0_bootstrap2.py (quantil + media da cauda >= VaR)."""
    v = np.quantile(L, gamma, axis=1, keepdims=True); m = L >= v
    return (L * m).sum(1) / m.sum(1)


def two_level_ci(rows, Z, target, B=1000, seed=0):
    """IC95 em dois niveis (instancias de val e do alvo + configuracoes), mesma receita e mesma ordem de sorteios do
    e0_bootstrap2.py para uma rodada: reproduz os numeros dele com o mesmo B e a mesma semente."""
    idx = np.array([r["cfg"] for r in rows]); V = Z["val"][idx].astype(np.float64); T = Z["test"][idx].astype(np.float64)
    mask = {"cvar_test": np.ones(T.shape[1], bool), "cvar_test_novel": Z["test_novel_mask"], "cvar_test_known": ~Z["test_novel_mask"]}[target]
    TT = T[:, mask]; lc = np.log10([r["curvature"] for r in rows]); H = list(hp_arrays(rows).values())
    rng = np.random.default_rng(seed); n = len(rows); nv, nt = V.shape[1], TT.shape[1]; a_, b_ = [], []
    for _ in range(B):
        v = cvar_rows(V[:, rng.integers(0, nv, nv)]); y = cvar_rows(TT[:, rng.integers(0, nt, nt)]); ic = rng.integers(0, n, n)
        a_.append(partial_spearman(y[ic], lc[ic], [v[ic]])); b_.append(partial_spearman(y[ic], lc[ic], [v[ic]] + [x[ic] for x in H]))
    q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return q(a_), q(b_)


def channel(rows, target):
    """Canal do sinal marginal: o hiperparametro que, sozinho como controle extra, mais reduz |parcial(curv, alvo | val)|.
    Devolve (parcial | val, {hp: parcial | val, hp}, {hp: parcial(hp, alvo | val)}, canal)."""
    y = np.array([r[target] for r in rows]); v = np.array([r["cvar_val"] for r in rows]); lc = np.log10([r["curvature"] for r in rows])
    H = hp_arrays(rows); p = partial_spearman(y, lc, [v])
    with_hp = {k: partial_spearman(y, lc, [v, c]) for k, c in H.items()}
    hp_own = {k: partial_spearman(y, c, [v]) for k, c in H.items()}
    ch = min(with_hp, key=lambda k: abs(with_hp[k]))
    return float(p), with_hp, hp_own, ch


def fmt(x, nd=2, sign=True, minus="−"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", minus)


def fci(ci, nd=2, minus="−"):
    return "—" if ci is None or not np.isfinite(ci[0]) else f"[{fmt(ci[0], nd, minus=minus)}; {fmt(ci[1], nd, minus=minus)}]"


# ----------------------------------------------------------------------------------------------- escrita de tabelas
def md_table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = "|" + "|".join({"l": ":--", "r": "--:", "c": ":-:"}[a] for a in align) + "|"
    out = ["| " + " | ".join(header) + " |", sep] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("−", "$-$"), ("ρ", r"$\rho$"), ("λ", r"$\lambda$"), ("²", "$^2$"), ("≥", r"$\geq$"), ("→", r"$\rightarrow$"), ("∣", r"$\mid$"), ("Δ", r"$\Delta$"), ("τ", r"$\tau$"), ("Ψ", r"$\Psi$"), ("ψ", r"$\psi$"), ("∋", r"$\ni$"), ("×", r"$\times$"), ("≤", r"$\leq$"), ("√", r"$\sqrt{}$"), ("²", "$^2$"), ("⁵", "$^5$"), ("ρ", r"$\rho$"), ("λ", r"$\lambda$")):
        s = s.replace(a, b)
    return s


def tex_table(header, rows, caption, label, align=None):
    align = align or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering\small", f"\\caption{{{tex_escape(caption)}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


class Out:
    def __init__(self, out_dir: pathlib.Path):
        self.dir = out_dir; self.dir.mkdir(parents=True, exist_ok=True); self.manifest = []; self.numbers = {}; self.tables = {}

    def table(self, name, header, rows, caption, sources, align=None, tex_align=None):
        (self.dir / f"tab_{name}.md").write_text(f"<!-- fontes: {', '.join(sources)} -->\n\n**{caption}**\n\n" + md_table(header, rows, align), encoding="utf-8")
        (self.dir / f"tab_{name}.tex").write_text(f"% fontes: {', '.join(sources)}\n" + tex_table(header, rows, caption, f"tab:{name}", tex_align), encoding="utf-8")
        self.tables[name] = {"header": header, "rows": [[str(c) for c in r] for r in rows]}
        self.manifest.append((f"tab_{name}.md / .tex", caption, sources))

    def figure(self, name, fig, caption, sources):
        fig.savefig(self.dir / f"fig_{name}.pdf", bbox_inches="tight"); fig.savefig(self.dir / f"fig_{name}.png", dpi=200, bbox_inches="tight")
        self.manifest.append((f"fig_{name}.pdf / .png", caption, sources))

    def num(self, key, value):
        self.numbers[key] = value if not isinstance(value, (np.floating, np.integer)) else value.item()


# ----------------------------------------------------------------------------------------------- dados derivados
def load_npz(root, src_path, dataset):
    p = root / "results" / "raw" / f"{src_path.stem}_{dataset}.npz"
    return np.load(p) if p.exists() else None


def e0_runs(L, root, args, measures_by_src):
    """Uma entrada por rodada do E0 (21 oficiais + emulador + natural): parciais, ICs (configs; dois niveis), 5 hp, canal."""
    rs = _load_script("replication_summary.py"); em = _load_script("e0_measures.py"); mr = _load_script("measures_replication.py")
    runs = []
    extra = [("kronodroid_emu", "KronoDroid emulator", [(0, "e0_krono_emu")], "cvar_test"), ("kronodroid_natural", "KronoDroid natural ratio", [(0, "e0_krono_natural")], "cvar_test")]
    for scen, label, seeds, target in SCENARIOS + extra:
        for seed, key in seeds:
            J, p = L[key]; d = J["payload"]["datasets"][0]; rows = rows_ok(d); r = d["results"][target]
            p5, ci5 = rs._partial_5hp_ci(rows, target)                       # mesma funcao e semente do replication_summary
            Z = load_npz(root, p, d["dataset"]); prov = "npz"
            if Z is not None:
                ci2, ci2_5 = two_level_ci(rows, Z, target, B=args.B2, seed=0)
            else:
                m = measures_by_src.get(p.stem) or measures_by_src.get(_mstem(p.stem, measures_by_src)); ci2_5 = None
                ci2 = m["measures"]["curvature"]["ci95_two_level"] if m else [float("nan")] * 2; prov = "e0_measures(-m*)" if m else "sem npz"
            pv, with_hp, hp_own, ch = channel(rows, target)
            code = mr.code(em.verdict(pv, ci2 if np.isfinite(ci2[0]) else r["partial_ci95"], ci5))
            runs.append({"scenario": scen, "label": label, "seed": seed, "key": key, "src": p.name, "commit": J["meta"]["git"]["commit_short"], "n": len(rows),
                         "n_total": d["n_total"], "target": target, "rho_raw": r["rho_raw"], "partial_val": r["partial_given_val"], "ci_configs": r["partial_ci95"],
                         "ci_two_level": ci2, "ci_two_level_prov": prov, "partial_5hp": p5, "ci_5hp": ci5, "ci_5hp_two_level": ci2_5, "code": code,
                         "channel": ch, "partial_given_channel": with_hp[ch], "channel_own_partial": hp_own[ch], "with_hp": with_hp, "hp_own": hp_own,
                         "rho_val_target": float(spearmanr([x["cvar_val"] for x in rows], [x[target] for x in rows])[0]),
                         "n_val": int(Z["val"].shape[1]) if Z is not None else None, "n_target": int(Z["test_novel_mask"].sum() if target == "cvar_test_novel" else Z["test"].shape[1]) if Z is not None else None, "n_test_total": int(Z["test"].shape[1]) if Z is not None else None,
                         "data_meta": d.get("data_meta", {}), "args": J["meta"].get("args", {})})
    return runs


def _mstem(stem, measures_by_src):
    """A rodada -m* que reproduz bit a bit a rodada original (mesma semente/fold): casa pelo sufixo da tag."""
    tag = stem.split("_", 2)[-1]
    cands = {"dionatan-k600-stop-s0": "dionatan-k600-stop-m0", "dionatan-krono-s0": "dionatan-krono-m0", "dionatan-krono-s1": "dionatan-krono-m1", "dionatan-krono-s2": "dionatan-krono-m2"}
    for k in range(5):
        cands[f"dionatan-drebin-cluster-f{k}"] = f"dionatan-drebin-cluster-m0-f{k}"
        for s in (1, 2):
            cands[f"dionatan-drebin-cluster-s{s}-f{k}"] = f"dionatan-drebin-cluster-m{s}-f{k}"
    want = cands.get(tag)
    for s in measures_by_src:
        if want and s.endswith(want):
            return s
    return None


def measures_index(L):
    idx = {}
    for key in SOURCES:
        if key.startswith("measures_") and key != "measures_replication":
            for run in L[key][0]["payload"]["runs"]:
                idx[run["src"]] = run
    return idx


# ----------------------------------------------------------------------------------------------- tabelas
def tab_scenarios(L, runs, out):
    header = ["Scenario", "Shift", "n train", "n val", "n target", "p(mal) train", "features", "k", "reached T (s0/s1/s2)"]
    rows = []
    by = {}
    for r in runs:
        by.setdefault(r["scenario"], []).append(r)
    f1 = L["f1_d2_drebin"][0]["payload"]; f1n = L["f1_nsl"][0]["payload"]
    def nk(r):
        return "/".join(str(x["n"]) for x in sorted(r, key=lambda x: x["seed"]))
    r = by["nslkdd"]; a = r[0]["args"]
    rows.append(["NSL-KDD", "novel attack categories (KDDTest+)", a.get("nsl_max_train", "—"), r[0]["n_val"] or f1n["e0"]["nslkdd"]["runs"][0]["scan"]["n_val"], f"{r[0]['n_target']} of {DATASET_FACTS['kddtest_plus']}" if r[0]["n_target"] else "—", "—", f"{DATASET_FACTS['nsl_raw_features']} (one-hot)", a.get("k"), nk(r)])
    for k in range(5):
        r = by[f"drebin_f{k}"]; a = r[0]["args"]; nv = r[0]["n_val"] or f1["e0"]["drebin_cluster"]["runs"][0]["scan"]["n_val"]
        ntr = DATASET_FACTS["drebin215_rows"] - nv - r[0]["n_test_total"] if r[0].get("n_test_total") else "—"
        rows.append([f"DREBIN cluster, fold {k}", "held-out malware family (k-means cluster)", ntr, nv, f"{r[0]['n_target']} (of {r[0]['n_test_total']} test rows)" if r[0]["n_target"] else "—", "—", "215", a.get("k"), nk(r)])
    r = by["kronodroid"]; dm = r[0]["data_meta"]; a = r[0]["args"]
    rows.append(["KronoDroid (real device)", "temporal: train ≤ 2014, test 2015–2020", dm.get("n_train"), dm.get("n_val"), dm.get("n_test"), fmt(dm.get("frac_mal_train"), 3, sign=False), dm.get("n_features"), a.get("k"), nk(r)])
    for scen, lab in (("kronodroid_emu", "KronoDroid (emulator)"), ("kronodroid_natural", "KronoDroid (natural class ratio)")):
        r = by[scen]; dm = r[0]["data_meta"]
        rows.append([lab, "temporal, same cut", dm.get("n_train"), dm.get("n_val"), dm.get("n_test"), fmt(dm.get("frac_mal_train"), 3, sign=False), dm.get("n_features"), r[0]["args"].get("k"), nk(r)])
    d2 = [L[f"d2_s{s}"][0] for s in range(3)]; c = d2[0]["payload"]["cifar"]
    rows.append(["CIFAR-10 → CIFAR-10.1", "dataset shift (Recht et al.)", c["n_train"], c["n_val"], f"{DATASET_FACTS['cifar10_test']} + {DATASET_FACTS['cifar101_v6']}", "—", "32×32×3", d2[0]["payload"]["analysis"]["n_total"],
                 "/".join(str(x["payload"]["analysis"]["n_ok"]) for x in d2)])
    out.table("scenarios", header, rows, "Table 1. Scenarios, splits and the number of configurations that reached the training-loss threshold.", ["E0 JSONs", "D2 JSONs", "f1_val_size", "results/raw/*.npz"])
    for r in runs:
        out.num(f"n_ok.{r['scenario']}.s{r['seed']}", r["n"])


def tab_main(L, runs, out):
    header = ["Scenario", "seed", "n", "ρ raw", "partial " + GIVEN + " val", "CI95 configs", "CI95 two-level", "partial " + GIVEN + " val, 5 hp", "CI95", "verdict"]
    rows = []
    for r in runs:
        if r["scenario"] in ("kronodroid_emu", "kronodroid_natural"):
            continue
        rows.append([r["label"], r["seed"], r["n"], fmt(r["rho_raw"]), fmt(r["partial_val"]), fci(r["ci_configs"]), fci(r["ci_two_level"]) + ("" if r["ci_two_level_prov"] == "npz" else "†"), fmt(r["partial_5hp"]), fci(r["ci_5hp"]), r["code"]])
        out.num(f"e0.{r['scenario']}.s{r['seed']}.partial_val", round(r["partial_val"], 3)); out.num(f"e0.{r['scenario']}.s{r['seed']}.ci2", [round(x, 3) for x in r["ci_two_level"]])
        out.num(f"e0.{r['scenario']}.s{r['seed']}.partial_5hp", round(r["partial_5hp"], 3)); out.num(f"e0.{r['scenario']}.s{r['seed']}.ci_5hp", [round(x, 3) for x in r["ci_5hp"]]); out.num(f"e0.{r['scenario']}.s{r['seed']}.code", r["code"])
    out.table("main_e0", header, rows, "Table 2. Partial Spearman correlation of log Tr(H²) with the shifted-target CVaR given validation CVaR, and given validation plus the five hyperparameters, per scenario and seed. Verdict by the pre-registered per-measure rule: S sustains (both CIs above zero), P hyperparameter proxy (only the first), C contradicts (first CI below zero), 0 null; † = two-level CI taken from the bit-identical -m run.", ["E0 JSONs", "replication_summary 4exp (5-hp CI recipe)", "results/raw/*.npz"])
    # regra pre-registrada por experimento (12 pares) -- direto do replication_summary
    P = L["replication_4exp"][0]["payload"]; rows2 = []
    for r in P["nsl"]:
        rows2.append(["NSL-KDD", r["seed"], "partial " + GIVEN + " val not > 0 and " + GIVEN + " val, 5hp ∋ 0", f"{fmt(r['partial_val'])} {fci(r['ci_val'])}", f"{fmt(r['partial_5hp'])} {fci(r['ci_5hp'])}", "yes" if r["replica"] else "NO"])
    for r in P["drebin_cluster"]:
        rows2.append(["DREBIN cluster (5 folds, RE)", r["seed"][0], "RE " + GIVEN + " val ∋ 0 and RE " + GIVEN + " val, 5hp ∋ 0", f"{fmt(r['re'])} {fci(r['ci_re'])}, I²={r['I2']:.0%}", f"{fmt(r['re_5hp'])} {fci(r['ci_re_5hp'])}", "yes" if r["replica"] else "NO"])
    for r in P["cifar"]:
        rows2.append(["CIFAR-10 → 10.1", r["seed"], "(A) τ(gap) > 0; (B) partial " + GIVEN + " val, arch ∋ 0", f"(A) τ={fmt(r['tau_gap'])} {fci(r['tau_ci'])}, Ψ={fmt(r['Psi'])}", f"(B) {fmt(r['partial_B'])} {fci(r['ci_B'])}", "yes" if r["replica"] else "NO"])
    for r in P["kronodroid"]:
        rows2.append(["KronoDroid", r["seed"], "partial " + GIVEN + " val not > 0 and " + GIVEN + " val, 5hp ∋ 0", f"{fmt(r['partial_val'])} {fci(r['ci_val'])}", f"{fmt(r['partial_5hp'])} {fci(r['ci_5hp'])}", "yes" if r["replica"] else "NO"])
    out.table("main_rule", ["Experiment", "seed", "pre-registered pattern", "statistic 1", "statistic 2", "replicated?"], rows2,
              "Table 2b. Pre-registered replication rule per experiment and seed (replication_summary, four experiments).", ["replication_summary 4exp"], align=["l", "r", "l", "l", "l", "c"], tex_align="lrlllc")
    out.num("rule.pairs_total", sum(len(P[k]) for k in P)); out.num("rule.pairs_replicated", sum(int(r["replica"]) for k in P for r in P[k]))
    null5 = sum(1 for k in ("nsl", "kronodroid") for r in P[k] if r["ci_5hp"][0] <= 0 <= r["ci_5hp"][1]) + sum(1 for r in P["drebin_cluster"] if r["ci_re_5hp"][0] <= 0 <= r["ci_re_5hp"][1])
    null5 += sum(1 for s in range(3) if (lambda ci: ci[0] <= 0 <= ci[1])(cifar_5hp(L, s)[1]))
    out.num("rule.null_5hp_pairs", null5)


def tab_channels(L, runs, out):
    header = ["Scenario", "seed", "partial " + GIVEN + " val", "channel", "partial " + GIVEN + " val, channel", "channel's own partial " + GIVEN + " val", "ρ(val, target)"]
    rows = []
    for r in runs:
        rows.append([r["label"], r["seed"], fmt(r["partial_val"]), r["channel"], fmt(r["partial_given_channel"]), fmt(r["channel_own_partial"]), fmt(r["rho_val_target"])])
        out.num(f"channel.{r['scenario']}.s{r['seed']}", {"hp": r["channel"], "partial_given_channel": round(r["partial_given_channel"], 3), "own": round(r["channel_own_partial"], 3)})
    # CIFAR (alvo CVaR no 10.1)
    for s in range(3):
        J = L[f"d2_s{s}"][0]; rows_ = [x for x in J["payload"]["rows"] if x.get("reached") and np.isfinite(x["curvature"]) and x["curvature"] > 0]
        pv, with_hp, hp_own, ch = channel(rows_, "cvar_c101")
        rows.append(["CIFAR-10 → 10.1", s, fmt(pv), ch, fmt(with_hp[ch]), fmt(hp_own[ch]), fmt(float(spearmanr([x["cvar_val"] for x in rows_], [x["cvar_c101"] for x in rows_])[0]))])
        out.num(f"channel.cifar.s{s}", {"hp": ch, "partial_val": round(pv, 3), "partial_given_channel": round(with_hp[ch], 3), "own": round(hp_own[ch], 3)})
    out.table("channels", header, rows, "Table 3. The hyperparameter channel of the marginal signal: the single hyperparameter whose addition as a control brings the partial closest to zero, with its own partial correlation with the target given validation.", ["E0 JSONs", "D2 JSONs"], align=["l", "r", "r", "l", "r", "r", "r"], tex_align="lrrlrrr")


def tab_measures(L, runs, out):
    R = L["measures_replication"][0]["payload"]; order = ["nslkdd", "kronodroid"] + [f"drebin_cluster fold {k}" for k in range(5)]
    keys = ["curvature", "lmax", "curv2", "curvb", "sharp_0.1", "sharp_0.5"]
    header = ["Scenario"] + [MEASURE_LABELS[k] for k in keys]; rows = []; counts = {"S": 0, "P": 0, "C": 0, "0": 0, "mixed": 0, "cells": 0}
    by = {e["scenario"]: e for e in R["scenarios"]}
    for sc in order:
        e = by[sc]; cells = []
        for k in keys:
            m = e["measures"].get(k)
            if m is None:
                cells.append("—"); continue
            codes = "".join(m["codes"]); fe = m["fe_val"]; f5 = m["fe_5hp"]
            cells.append(f"{' '.join(m['codes'])} → {en(m['rule'].split(' (')[0])}; FE {fmt(fe['r'])} / {fmt(f5['r'])}")
            if m["k"] == 3:
                counts["cells"] += 1
                counts[codes[0] if len(set(codes)) == 1 else "mixed"] += 1
        rows.append([sc.replace("drebin_cluster", "DREBIN cluster,").replace("nslkdd", "NSL-KDD").replace("kronodroid", "KronoDroid")] + cells)
    out.table("measures", header, rows, "Table 4a. Verdict per measure and seed (S sustains, P hyperparameter proxy, C contradicts, 0 null; seeds 0/1/2), the 3/3 replication rule, and the fixed-effect combination of the three seeds for partial " + GIVEN + " val / partial " + GIVEN + " val, 5 hp.", ["measures_replication"], align=["l"] + ["l"] * len(keys), tex_align="l" + "l" * len(keys))
    for k, v in counts.items():
        out.num(f"measures.cells_{k}", v)
    out.num("measures.S_per_seed", {str(s): sum(1 for e in R["scenarios"] for m in e["measures"].values() for sd, c in zip(e["seeds"], m["codes"]) if c == "S" and sd == s) for s in range(3)})
    # 4b: C3-C
    header2 = ["Run", "n", "partial " + GIVEN + " val", "CI95 configs", "CI95 two-level", "partial " + GIVEN + " val, 5 hp", "CI95", "channel", "partial | val, channel", "verdict"]
    rows2 = []
    for r in runs:
        if r["scenario"] in ("kronodroid", "kronodroid_emu", "kronodroid_natural"):
            rows2.append([r["label"] + f" s{r['seed']}", r["n"], fmt(r["partial_val"]), fci(r["ci_configs"]), fci(r["ci_two_level"]) + ("" if r["ci_two_level_prov"] == "npz" else "†"), fmt(r["partial_5hp"]), fci(r["ci_5hp"]), r["channel"], fmt(r["partial_given_channel"]), r["code"]])
            out.num(f"e0.{r['scenario']}.s{r['seed']}.partial_val", round(r["partial_val"], 3)); out.num(f"e0.{r['scenario']}.s{r['seed']}.ci2", [round(x, 3) for x in r["ci_two_level"]]); out.num(f"e0.{r['scenario']}.s{r['seed']}.partial_5hp", round(r["partial_5hp"], 3))
    out.table("kronodroid_variants", header2, rows2, "Table 4b. KronoDroid: real device (three seeds), emulator and natural class ratio (seed 0), read by the same rule.", ["E0 JSONs (KronoDroid)", "results/raw/*.npz"])
    # confiabilidade do estimador (e0_measures)
    rows3 = []
    for key in ["measures_nsl_m0", "measures_krono_m0", "measures_krono_m1", "measures_krono_m2", "measures_drebin_m0", "measures_drebin_m1", "measures_drebin_m2"]:
        for run in L[key][0]["payload"]["runs"]:
            rel = run["reliability"]; rows3.append([run["name"], run["n"], fmt(rel["curv2"]["spearman"], sign=False), fmt(rel["curv2"]["attenuation_sqrt_r"], sign=False), fmt(rel["curvb"]["spearman"], sign=False), fmt(rel["curvb"]["attenuation_sqrt_r"], sign=False)])
    out.table("reliability", ["Run", "n", "r (same batch, M=5 vs M=200)", "√r", "r (second batch)", "√r"], rows3,
              "Table 6b. Test–retest reliability of the Hutchinson estimator between configurations and the implied maximum attenuation of the partials.", ["e0_measures (-m0/-m1/-m2)"])
    rels = [run["reliability"]["curv2"]["spearman"] for key in ["measures_nsl_m0", "measures_krono_m0", "measures_krono_m1", "measures_krono_m2", "measures_drebin_m0", "measures_drebin_m1", "measures_drebin_m2"] for run in L[key][0]["payload"]["runs"]]
    out.num("reliability.min", round(min(rels), 3)); out.num("reliability.max", round(max(rels), 3)); out.num("reliability.attenuation_max", round(1 - np.sqrt(min(rels)), 3))


def tab_within(L, out):
    keys = ["curvature", "curv2", "lmax", "sharp_0.1", "sharp_0.5"]
    header = ["Scenario", "measure", "ICC", "partial within " + GIVEN + " val", "CI95 (seed bootstrap)", "Fisher RE", "CI95", "I²", "between (n=15)", "verdict"]
    rows = []
    for key, lab in (("within_krono", "KronoDroid"), ("within_drebin_f0", "DREBIN cluster, fold 0"), ("within_drebin_f3", "DREBIN cluster, fold 3")):
        A = L[key][0]["payload"]["analysis"]
        for k in keys:
            m = A["measures"].get(k)
            if not m:
                continue
            rows.append([lab, {"curvature": "Tr(H²) M=200 (primary)", "curv2": "Tr(H²) M=200 (2nd estimate)"}.get(k, MEASURE_LABELS.get(k, k)), fmt(m["icc_between"], sign=False), fmt(m["partial_within"]), fci(m["ci95_within"]), fmt(m["fisher_re"]), fci(m["fisher_ci95_re"]), f"{m['I2']:.0%}", fmt(m["partial_between"]), en(m["verdict"].replace(" (intra-config)", ""))])
            out.num(f"within.{key}.{k}.partial", round(m["partial_within"], 3)); out.num(f"within.{key}.{k}.ci", [round(x, 3) for x in m["ci95_within"]]); out.num(f"within.{key}.{k}.between", round(m["partial_between"], 3)); out.num(f"within.{key}.{k}.icc", round(m["icc_between"], 3))
        rows.append([lab, "ρ within(val, target); sd within of target / val", "", fmt(A["rho_val_target_within"]), "", "", "", "", f"{A['sd_within']['target']:.3f} / {A['sd_within']['val']:.3f}", ""])
        out.num(f"within.{key}.rho_val_target", round(A["rho_val_target_within"], 3)); out.num(f"within.{key}.sd_target", round(A["sd_within"]["target"], 3)); out.num(f"within.{key}.sd_val", round(A["sd_within"]["val"], 3))
        out.num(f"within.{key}.reliability_within", round(A["reliability_curv2"]["spearman_within"], 3)); out.num(f"within.{key}.n_runs", A["n_runs"]); out.num(f"within.{key}.n_not_reached", A["n_not_reached"])
    out.table("within", header, rows, "Table 5. Same configuration, different training seeds (15 configurations × 20 seeds, Tr(H²) with 200 probes): within-configuration partial correlations, between-configuration contrast and the seed-to-seed spread of the shifted target versus validation.", ["e0_within_config (-p200)"])
    for key, lab in (("within_krono_m5", "KronoDroid"), ("within_drebin_f0_m5", "DREBIN f0")):
        A = L[key][0]["payload"]["analysis"]; out.num(f"within_m5.{key}.reliability_within", round(A["reliability_curv2"]["spearman_within"], 3)); out.num(f"within_m5.{key}.reliability_total", round(A["reliability_curv2"]["spearman_total"], 3))


def tab_halves(L, out):
    rows_ = L["halves"][0]["payload"]["rows"]; header = ["Run", "target", "n", "ρ raw", "partial " + GIVEN + " val (E0)", "halves mean", "sd", "q2.5", "q97.5", "outside band?"]
    rows = []; outside = []
    for r in rows_:
        prim = "cvar_test" if r["dataset"] == "kronodroid" else "cvar_test_novel"
        o = r["partial_val_e0"] < r["halves_q2_5"] or r["partial_val_e0"] > r["halves_q97_5"]
        rows.append([f"{r['run']}" + (f" s{r['seed']}" if r["dataset"] != "drebin_cluster" else f" s{r['seed']}"), r["target"], r["n"], fmt(r["rho_raw"]), fmt(r["partial_val_e0"]), fmt(r["halves_mean"]), fmt(r["halves_std"], sign=False), fmt(r["halves_q2_5"]), fmt(r["halves_q97_5"]), ("yes" if o else "no") + ("*" if r["target"] == prim else "")])
        if r["target"] == prim and o:
            outside.append(f"{r['run']} s{r['seed']}")
    out.table("halves", header, rows, "Table 6a. Baseline of the 'i.i.d. halves' check as the mean over 300 random splits of the target (the expected value of the partial given a noisy control is not zero); * marks the primary target of each run.", ["e0_halves_baseline"])
    out.num("halves.mean_min", round(min(r["halves_mean"] for r in rows_), 3)); out.num("halves.mean_max", round(max(r["halves_mean"] for r in rows_), 3))
    out.num("halves.sd_min", round(min(r["halves_std"] for r in rows_), 3)); out.num("halves.sd_max", round(max(r["halves_std"] for r in rows_), 3)); out.num("halves.outside_primary", outside)


def tab_e1(L, out):
    P = L["e1_offline"][0]["payload"]; prim = P["primary"]; order = ["nslkdd", "kronodroid"] + [f"drebin_cluster fold {k}" for k in range(5)]
    by = {}
    for r in P["results"]:
        key = f"{r['dataset']} fold {r['fold']}" if r["dataset"] == "drebin_cluster" else r["dataset"]; by.setdefault(key, []).append(r)
    header = ["Scenario", "ρ(val, target) s0/s1/s2", "regret of V", "V+Tr(H²): Δ s0/s1/s2 → rule", "V+λmax", "V+sharpness", "V+placebo"]; rows = []; helps = 0; cells = 0
    for sc in order:
        rs = sorted(by[sc], key=lambda r: r["seed"]); cells_ = []
        for arm in ("V+Tr", "V+lmax", "V+S", "V+P"):
            ds = []
            for r in rs:
                run = [x for x in r["runs"] if x["budget"] == min(prim["budget"], r["n_pool"]) and x["w"] == prim["w"]][0]
                if arm in run["arms"]:
                    ds.append(run["arms"][arm]["delta_vs_V"])
            rule = P["rule"][sc]["arms"].get(arm, {}).get("rule", "—")
            cells_.append("/".join(fmt(d) for d in ds) + f" → {en(rule.split(' (')[0])}")
            if len(ds) == 3:
                cells += 1; helps += int(rule.startswith("ajuda em"))
            out.num(f"e1.{sc}.{arm}.delta", [round(d, 3) for d in ds]); out.num(f"e1.{sc}.{arm}.rule", rule)
        rho = "/".join(fmt(r["rho_val_target"]) for r in rs); reg = "/".join(fmt([x for x in r["runs"] if x["budget"] == min(prim["budget"], r["n_pool"]) and x["w"] == prim["w"]][0]["arms"]["V"]["regret"], sign=False) for r in rs)
        rows.append([sc.replace("drebin_cluster", "DREBIN cluster,").replace("nslkdd", "NSL-KDD").replace("kronodroid", "KronoDroid"), rho, reg] + cells_)
    out.table("e1_offline", header, rows, f"Table 8. Offline tuner criterion (budget b={prim['budget']}, w={prim['w']:g}): change in the shifted-target CVaR of the chosen configuration relative to the validation-only criterion, per seed, and the 3/3 rule; regret = validation-only choice minus the oracle.", ["e1_offline"], align=["l", "l", "l", "l", "l", "l", "l"], tex_align="lllllll")
    out.num("e1.cells_3seeds", cells); out.num("e1.helps_3of3", helps)
    # robustez a b e w: vereditos 3/3 "atrapalha"/"ajuda" nas outras combinacoes
    rob = {}
    for b in (10, 30, 100):
        for w in (1.0, 0.5):
            lst = []
            for sc in order:
                rs = sorted(by[sc], key=lambda r: r["seed"])
                if len(rs) < 3:
                    continue
                for arm in ("V+Tr", "V+lmax", "V+S", "V+P"):
                    vs = []
                    for r in rs:
                        run = [x for x in r["runs"] if x["budget"] == min(b, r["n_pool"]) and x["w"] == w][0]
                        if arm in run["arms"]:
                            vs.append(run["arms"][arm]["verdict"])
                    if len(vs) == 3 and len(set(vs)) == 1 and vs[0] != "nulo":
                        lst.append(f"{sc}:{arm}:{vs[0]}")
            rob[f"b{b}_w{w:g}"] = lst
    out.num("e1.robustness_3of3", rob)


def en_f1(v):
    v = v.replace("NAO SUSTENTA: IC95 contem zero em algum n_v <= 200", "does not sustain (CI contains zero at some n_v ≤ 200)")
    v = v.replace("ESCASSEZ SUSTENTA H_G: IC95 acima de zero em todos os n_v <= 200 e P nao cresce com n_v", "scarcity sustains (CI above zero at all n_v ≤ 200, P non-increasing)")
    v = v.replace("CONTRADIZ H_G: IC95 abaixo de zero em todos os n_v <= 200", "contradicts (CI below zero at all n_v ≤ 200)")
    return v.split(" (tendencia")[0] if "tendencia" in v else v


def tab_f1(L, out):
    header = ["Scenario", "statistic", "n_v = 50", "n_v = 100", "n_v = 200", "full validation", "verdict"]; rows = []
    def line(label, res, key="P", ci="P_ci95"):
        per = {p["n_val"]: p for p in res["per_size"]}; full = res["per_size"][-1]
        return [label, key, *[f"{fmt(per[n][key])} {fci(per[n][ci])}" if n in per else "—" for n in (50, 100, 200)], f"{fmt(full[key])} {fci(full[ci])} (n_v={full['n_val']})", en_f1(res.get("verdict", ""))]
    d2 = L["f1_d2_drebin"][0]["payload"]["d2"]["results"]["cvar :: cvar_c101 (deslocado) :: log Tr(H2)"]
    rows.append(line("CIFAR-10.1 (CVaR), log Tr(H²)", d2)); rows.append(line("CIFAR-10.1, P_arch (| val, depth, width)", d2, "P_arch", "P_arch_ci95"))
    dc = L["f1_d2_drebin"][0]["payload"]["e0"]["drebin_cluster"]["combined"]["cvar"]
    rows.append(line("DREBIN cluster (RE over 5 folds)", dc)); rows.append(line("DREBIN cluster, P_arch", dc, "P_arch", "P_arch_ci95"))
    ns = L["f1_nsl"][0]["payload"]["e0"]["nslkdd"]["combined"]["cvar"]
    rows.append(line("NSL-KDD (RE over 3 seeds)", ns)); rows.append(line("NSL-KDD, P_arch", ns, "P_arch", "P_arch_ci95"))
    out.table("f1", header, rows, "Table 9. Validation scarcity: partial correlation of log Tr(H²) with the shifted target given a validation CVaR computed on n_v samples (mean over 200 subsamples; two-level bootstrap CI), and the same given depth and width.", ["f1_val_size"], align=["l", "l", "l", "l", "l", "l", "l"], tex_align="lllllll")
    for name, res in (("cifar", d2), ("drebin", dc), ("nsl", ns)):
        for p in res["per_size"]:
            out.num(f"f1.{name}.nv{p['n_val']}.P", round(p["P"], 3)); out.num(f"f1.{name}.nv{p['n_val']}.P_ci", [round(x, 3) for x in p["P_ci95"]]); out.num(f"f1.{name}.nv{p['n_val']}.P_arch", round(p["P_arch"], 3))
        out.num(f"f1.{name}.verdict", res.get("verdict", ""))


def cifar_5hp(L, s):
    """IC por configuracoes da parcial | val, 5 hp no CIFAR-10.1, com a mesma receita e semente do replication_summary."""
    rs = _load_script("replication_summary.py"); J = L[f"d2_s{s}"][0]
    rows_ = [x for x in J["payload"]["rows"] if x.get("reached") and np.isfinite(x["curvature"]) and x["curvature"] > 0]
    return rs._partial_5hp_ci(rows_, "cvar_c101")


def tab_cifar(L, out):
    header = ["seed", "n", "τ(gap) [CI95]", "Ψ", "factors with ψ > 0 (p < 0.05)", "(A)", "(B) partial " + GIVEN + " val, arch [CI95]", "partial " + GIVEN + " val", "partial " + GIVEN + " val, 5 hp [CI95]", "ref: τ of val error", "ref: τ of dropout"]
    rows = []; fu = L["d2_followup"][0]["payload"]["reference_tau"]
    for s in range(3):
        J = L[f"d2_s{s}"][0]; A = J["payload"]["analysis"]; m = A["measures"]["log_curv"]; p5, ci5 = cifar_5hp(L, s)
        sig = [k for k, v in m["psi"].items() if v["psi"] > 0 and v["p_sign"] < 0.05]
        rows.append([s, A["n_ok"], f"{fmt(m['tau_gap'])} {fci(m['tau_ci95'])}", fmt(m["Psi"]), ", ".join(sig), "positive" if m["A_positive"] else "no", f"{fmt(m['partial_B'])} {fci(m['partial_B_ci95'])}", fmt(m["partial_val"]), f"{fmt(p5)} {fci(ci5)}",
                     fmt(fu["erro de validacao"]) if s == 0 else "", fmt(fu["-dropout (so' o dropout)"]) if s == 0 else ""])
        out.num(f"cifar.s{s}.ci_5hp", [round(x, 3) for x in ci5])
        out.num(f"cifar.s{s}.tau_gap", round(m["tau_gap"], 3)); out.num(f"cifar.s{s}.tau_ci", [round(x, 3) for x in m["tau_ci95"]]); out.num(f"cifar.s{s}.partial_B", round(m["partial_B"], 3)); out.num(f"cifar.s{s}.partial_B_ci", [round(x, 3) for x in m["partial_B_ci95"]]); out.num(f"cifar.s{s}.partial_5hp", round(m["partial_5hp"], 3))
        for mk in ("log_sharp_0.1", "log_sharp_0.5"):
            mm = A["measures"][mk]; out.num(f"cifar.s{s}.{mk}.tau_gap", round(mm["tau_gap"], 3)); out.num(f"cifar.s{s}.{mk}.partial_B", round(mm["partial_B"], 3))
    out.table("cifar", header, rows, "Table 10. CIFAR-10 grid (3⁵ = 243 configurations, 3 seeds): (A) replication of the generalization-gap correlation (Kendall τ, granulated Ψ) and (B) the Tune3 question on CIFAR-10.1.", ["D2 JSONs", "d2_followup"], align=["r"] + ["l"] * 10, tex_align="r" + "l" * 10)


def tab_r2(L, out):
    rows_ = L["curvature_vs_hparams"][0]["payload"]["rows"]; by = {}
    for r in rows_:
        by.setdefault((r["run"].split(" ")[0], r["measure"]), []).append(r)
    header = ["dataset", "measure", "runs", "R² linear (min–max)", "R² quadratic (min–max)", "dominant single factor (R² range)"]; rows = []
    name = {"cifar": "CIFAR-10", "drebin_cluster": "DREBIN cluster", "nslkdd": "NSL-KDD", "kronodroid": "KronoDroid"}
    for (ds, m), v in sorted(by.items()):
        lin = [x["r2_linear"] for x in v]; quad = [x["r2_quadratic"] for x in v]
        dom = {}
        for x in v:
            k = max(x["r2_single"], key=x["r2_single"].get); dom.setdefault(k, []).append(x["r2_single"][k])
        hp_en = {"prof.": "depth", "largura": "width", "log_lr": "log lr", "log_wd": "log wd", "dropout": "dropout"}
        dom_s = "; ".join(f"{hp_en.get(k, k)} ({min(vv):.2f}–{max(vv):.2f}, {len(vv)} run{'s' if len(vv) > 1 else ''})" for k, vv in sorted(dom.items(), key=lambda kv: -len(kv[1])))
        rows.append([name.get(ds, ds), MEASURE_LABELS.get(m, m), len(v), f"{min(lin):.2f}–{max(lin):.2f}", f"{min(quad):.2f}–{max(quad):.2f}", dom_s])
        out.num(f"r2.{ds}.{m}.linear", [round(min(lin), 3), round(max(lin), 3)]); out.num(f"r2.{ds}.{m}.quadratic", [round(min(quad), 3), round(max(quad), 3)])
    out.table("r2", header, rows, "Table 7. How much of each geometric measure is a function of the five hyperparameters: R² of the log measure regressed on (log lr, log wd, dropout, width, depth), linear and with quadratic/cross terms, per dataset (range over runs).", ["curvature_vs_hparams"], align=["l", "l", "r", "l", "l", "l"], tex_align="llrlll")


# ----------------------------------------------------------------------------------------------- figuras
BLUE, ORANGE, AQUA, VIOLET, RED, GRAY, INK, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#8d8c88", "#0b0b0b", "#e6e5e1", "#ffffff"
SEED_COLORS = [BLUE, ORANGE, AQUA]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
                         "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42})
    return plt


def _errbar(ax, x, y, ci, color, label=None, marker="o", ms=5):
    ci = np.asarray(ci, float)
    ok = np.isfinite(ci).all(axis=1) if ci.ndim == 2 else np.isfinite(ci)
    xerr = np.vstack([np.where(ok, x - ci[:, 0], 0), np.where(ok, ci[:, 1] - x, 0)]) if ci.ndim == 2 else None
    ax.errorbar(x, y, xerr=xerr, fmt=marker, color=color, ecolor=color, elinewidth=1.4, capsize=0, ms=ms, mec=SURF, mew=1.2, label=label, zorder=3)


def fig_forest(runs, L, out):
    plt = _mpl()
    main = [r for r in runs if r["scenario"] not in ("kronodroid_emu", "kronodroid_natural")]
    labels = [f"{r['label'].replace('DREBIN cluster, fold ', 'DREBIN f').replace(' (novel attacks)', '').replace(' (temporal)', '')} s{r['seed']}" for r in main]
    cif = [(s, L[f"d2_s{s}"][0]["payload"]["analysis"]["measures"]["log_curv"]) for s in range(3)]
    labels += [f"CIFAR-10.1 (B) s{s}" for s, _ in cif]
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.0), sharey=True)
    pv = np.array([r["partial_val"] for r in main] + [m["partial_B"] for _, m in cif])
    ci = np.array([(r["ci_two_level"] if np.isfinite(r["ci_two_level"][0]) else r["ci_configs"]) for r in main] + [m["partial_B_ci95"] for _, m in cif])
    _errbar(axes[0], pv, y, ci, BLUE)
    c5 = [cifar_5hp(L, s) for s in range(3)]
    p5 = np.array([r["partial_5hp"] for r in main] + [p for p, _ in c5]); ci5 = np.array([r["ci_5hp"] for r in main] + [ci for _, ci in c5])
    _errbar(axes[1], p5, y, ci5, BLUE)
    for ax, title in zip(axes, ["a. given validation CVaR", "b. given validation CVaR and the 5 hyperparameters"]):
        ax.axvline(0, color=INK, lw=0.8, zorder=2); ax.set_title(title, fontsize=8.5, loc="left"); ax.set_xlim(-0.75, 0.75); ax.set_axisbelow(True)
    for yy in (y[2], y[5], y[20]):
        for ax in axes:
            ax.axhline(yy - 0.5, color=GRID, lw=0.8)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels)
    fig.supxlabel("partial Spearman correlation of log Tr(H²) with the shifted CVaR (positive = direction predicted by H_G)", fontsize=8.5)
    fig.tight_layout(); out.figure("forest_e0", fig, "Figure 2. Main result: partial correlations per scenario and seed with 95% CIs (two-level where per-sample losses exist; CIFAR (B) is the partial given validation, depth and width, as pre-registered).", ["E0 JSONs", "D2 JSONs", "results/raw/*.npz"]); plt.close(fig)


def fig_drebin(runs, out):
    plt = _mpl()
    dre = [r for r in runs if r["scenario"].startswith("drebin_f")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]; ax.set_axisbelow(True); axes[1].set_axisbelow(True)
    for s in range(3):
        rs = [r for r in dre if r["seed"] == s]; x = np.array([int(r["scenario"][-1]) for r in rs]) + (s - 1) * 0.15
        ax.scatter(x, [r["partial_val"] for r in rs], color=SEED_COLORS[s], s=28, edgecolor=SURF, linewidth=1, zorder=3, label=f"seed {s}")
    fe_vals = []
    for k in range(5):
        rs = [r for r in dre if r["scenario"] == f"drebin_f{k}"]; fc = fisher_combine([r["partial_val"] for r in rs], [r["n"] for r in rs], 1); fe_vals.append(fc)
        ax.plot([k - 0.3, k + 0.3], [fc["r"], fc["r"]], color=INK, lw=1.4, zorder=2)
        out.num(f"drebin.f{k}.fe_partial_val", round(fc["r"], 3)); out.num(f"drebin.f{k}.fe_ci", [round(x, 3) for x in fc["ci95"]]); out.num(f"drebin.f{k}.I2", round(fc["I2"], 3))
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels([f"fold {k}" for k in range(5)]); ax.set_ylabel("partial | val"); ax.set_title("a. partial of log Tr(H²) with the held-out family\n(bar = fixed effect of the 3 seeds)", fontsize=8, loc="left"); ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    for hp, col, lab in (("depth", BLUE, "depth"), ("dropout", ORANGE, "dropout")):
        for s in range(3):
            rs = [r for r in dre if r["seed"] == s]; x = np.array([int(r["scenario"][-1]) for r in rs]) + (-0.12 if hp == "depth" else 0.12) + (s - 1) * 0.05
            ax.scatter(x, [r["hp_own"][hp] for r in rs], color=col, s=22, edgecolor=SURF, linewidth=1, zorder=3, label=lab if s == 0 else None)
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels([f"fold {k}" for k in range(5)]); ax.set_ylabel("partial | val"); ax.set_title("b. depth and dropout: own partial with the held-out family\n(3 seeds each)", fontsize=8, loc="left"); ax.legend(fontsize=7, loc="upper right")
    taus = [kendalltau([r["partial_val"] for r in sorted([x for x in dre if x["seed"] == a], key=lambda r: r["scenario"])], [r["partial_val"] for r in sorted([x for x in dre if x["seed"] == b], key=lambda r: r["scenario"])])[0] for a, b in ((0, 1), (0, 2), (1, 2))]
    out.num("drebin.fold_order_tau_between_seeds", [round(float(t), 2) for t in taus])
    fig.tight_layout(); out.figure("drebin_folds", fig, "Figure 3. DREBIN by cluster: reproducible heterogeneity across held-out families and sign reversal of the depth and dropout effects.", ["E0 JSONs (DREBIN)"]); plt.close(fig)


def fig_heatmap(L, out):
    plt = _mpl(); from matplotlib.patches import Patch
    R = L["measures_replication"][0]["payload"]; order = ["nslkdd", "kronodroid"] + [f"drebin_cluster fold {k}" for k in range(5)]
    keys = ["curvature", "lmax", "curv2", "curvb", "sharp_0.1", "sharp_0.5"]; cols = []
    for sc in order:
        e = [x for x in R["scenarios"] if x["scenario"] == sc][0]
        for s in e["seeds"]:
            cols.append((sc, s, e))
    colors = {"0": GRID, "P": "#bfe9d8", "S": BLUE, "C": ORANGE}; txt = {"0": INK, "P": INK, "S": SURF, "C": SURF}
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.grid(False)
    for j, (sc, s, e) in enumerate(cols):
        for i, k in enumerate(keys):
            m = e["measures"].get(k)
            code = m["codes"][e["seeds"].index(s)] if m else "—"
            ax.add_patch(plt.Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, color=colors.get(code, SURF), lw=0))
            ax.text(j + 0.5, i + 0.5, code, ha="center", va="center", fontsize=7.5, color=txt.get(code, INK), fontweight="bold" if code in "SC" else "normal")
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(keys), 0); ax.set_yticks(np.arange(len(keys)) + 0.5); ax.set_yticklabels([MEASURE_LABELS[k] for k in keys])
    ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels([f"s{s}" for _, s, _ in cols], fontsize=7)
    starts = {}
    for j, (sc, s, e) in enumerate(cols):
        starts.setdefault(sc, j)
    for sc, j0 in starts.items():
        n = sum(1 for c in cols if c[0] == sc); ax.text(j0 + n / 2, -0.25, sc.replace("drebin_cluster fold ", "DREBIN f").replace("nslkdd", "NSL-KDD").replace("kronodroid", "KronoDroid"), ha="center", va="bottom", fontsize=7.5)
        if j0 > 0:
            ax.axvline(j0, color=SURF, lw=2.5)
    ax.legend(handles=[Patch(color=colors[c], label=l) for c, l in (("S", "S: sustains (both partials > 0)"), ("P", "P: hyperparameter proxy"), ("C", "C: contradicts"), ("0", "0: null"))], fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    out.figure("measures_heatmap", fig, "Figure 4. Verdict per geometric measure, scenario and seed by the pre-registered per-measure rule (two families of measures, two estimators of Tr(H²)).", ["measures_replication"]); plt.close(fig)


def fig_within(L, out):
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    panels = (("within_krono", "KronoDroid", "cvar_test"), ("within_drebin_f0", "DREBIN fold 0", "cvar_test_novel"), ("within_drebin_f3", "DREBIN fold 3", "cvar_test_novel"))
    sds = []
    for ax, (key, lab, tgt) in zip(axes[:3], panels):
        J = L[key][0]; rows = [r for r in J["payload"]["rows"] if r["reached"]]; A = J["payload"]["analysis"]
        cfgs = sorted({r["cfg"] for r in rows}); X, Y = [], []
        for c in cfgs:
            rr = [r for r in rows if r["cfg"] == c]; x = np.log10([r["sharp_0.1"] for r in rr]); y = np.array([r[tgt] for r in rr])
            X.append(x - x.mean()); Y.append(y - y.mean())
            b = np.polyfit(x - x.mean(), y - y.mean(), 1)[0]; xs = np.array([min(x - x.mean()), max(x - x.mean())]); ax.plot(xs, b * xs, color=GRAY, lw=0.7, alpha=0.7, zorder=2)
        X = np.concatenate(X); Y = np.concatenate(Y); ax.scatter(X, Y, s=9, color=BLUE, alpha=0.55, edgecolor="none", zorder=3)
        m = A["measures"]["sharp_0.1"]; ax.set_title(f"{lab}\nρ_w = {m['partial_within']:+.2f} {fci(m['ci95_within'])}", fontsize=7.5, loc="left")
        ax.set_xlabel("log sharpness ρ=0.1\n(centred per configuration)", fontsize=7); ax.axhline(0, color=INK, lw=0.6); ax.axvline(0, color=INK, lw=0.6); ax.set_axisbelow(True)
        if key == "within_krono":
            ax.set_ylabel("shifted CVaR (centred)", fontsize=7)
        sds.append((lab, A["sd_within"]["target"], A["sd_within"]["val"]))
    ax = axes[3]; x = np.arange(3)
    ax.bar(x - 0.18, [s[1] for s in sds], width=0.34, color=BLUE, label="shifted target"); ax.bar(x + 0.18, [s[2] for s in sds], width=0.34, color=GRAY, label="validation")
    for i, s in enumerate(sds):
        ax.text(i - 0.18, s[1] + 0.01, f"{s[1]:.2f}", ha="center", fontsize=6.5); ax.text(i + 0.18, s[2] + 0.01, f"{s[2]:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(["Krono-\nDroid", "DREBIN\nf0", "DREBIN\nf3"], fontsize=7); ax.set_title("sd across the 20 seeds,\nsame configuration", fontsize=7.5, loc="left"); ax.legend(fontsize=6.5, loc="upper left"); ax.set_ylim(0, max(s[1] for s in sds) * 1.3); ax.set_axisbelow(True)
    fig.tight_layout(); out.figure("within_config", fig, "Figure 5. Same configuration, 20 training seeds: within-configuration relation between adaptive sharpness and the shifted CVaR (thin lines = per-configuration fits), and the seed lottery — spread of the shifted target versus validation.", ["e0_within_config (-p200)"]); plt.close(fig)


def fig_f1(L, out):
    plt = _mpl()
    d2 = L["f1_d2_drebin"][0]["payload"]["d2"]["results"]["cvar :: cvar_c101 (deslocado) :: log Tr(H2)"]; dc = L["f1_d2_drebin"][0]["payload"]["e0"]["drebin_cluster"]["combined"]["cvar"]; ns = L["f1_nsl"][0]["payload"]["e0"]["nslkdd"]["combined"]["cvar"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for res, col, lab in ((d2, BLUE, "CIFAR-10.1"), (dc, ORANGE, "DREBIN cluster (RE, 5 folds)"), (ns, AQUA, "NSL-KDD (RE, 3 seeds)")):
        nv = [p["n_val"] for p in res["per_size"]]
        for ax, key, ci in ((axes[0], "P", "P_ci95"), (axes[1], "P_arch", "P_arch_ci95")):
            v = [p[key] for p in res["per_size"]]; lo = [p[ci][0] for p in res["per_size"]]; hi = [p[ci][1] for p in res["per_size"]]
            ax.plot(nv, v, color=col, lw=1.8, marker="o", ms=4, mec=SURF, mew=0.8, label=lab); ax.fill_between(nv, lo, hi, color=col, alpha=0.12, lw=0)
    for ax, t in zip(axes, ["a. given a validation CVaR on n_v samples", "b. given validation, depth and width"]):
        ax.set_xscale("log"); ax.axhline(0, color=INK, lw=0.8); ax.set_xlabel("validation size n_v"); ax.set_title(t, fontsize=8, loc="left"); ax.set_axisbelow(True)
    axes[0].set_ylabel("partial correlation with the shifted target"); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(); out.figure("f1_scarcity", fig, "Figure 6. Validation scarcity: the partial correlation of log Tr(H²) with the shifted target as the validation set shrinks (bands: two-level bootstrap 95% CI).", ["f1_val_size"]); plt.close(fig)


def fig_r2(L, out):
    plt = _mpl()
    rows_ = L["curvature_vs_hparams"][0]["payload"]["rows"]; by = {}
    for r in rows_:
        by.setdefault((r["run"].split(" ")[0], r["measure"]), []).append(r["r2_linear"])
    measures = ["curvature", "lmax", "sharp_0.1", "sharp_0.5"]; datasets = [("drebin_cluster", "DREBIN cluster", BLUE), ("nslkdd", "NSL-KDD", ORANGE), ("kronodroid", "KronoDroid", AQUA), ("cifar", "CIFAR-10", VIOLET)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); w = 0.19
    for i, (ds, lab, col) in enumerate(datasets):
        xs, means, lo, hi = [], [], [], []
        for j, m in enumerate(measures):
            v = by.get((ds, m))
            if not v:
                continue
            xs.append(j + (i - 1.5) * w); means.append(np.mean(v)); lo.append(np.mean(v) - min(v)); hi.append(max(v) - np.mean(v))
        ax.bar(xs, means, width=w * 0.9, color=col, label=lab); ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(measures))); ax.set_xticklabels([MEASURE_LABELS[m] for m in measures]); ax.set_ylabel("R² of the log measure on the 5 hyperparameters"); ax.set_ylim(0, 1.0); ax.set_axisbelow(True); ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.set_title("mean over runs; whiskers = min–max", fontsize=8, loc="left")
    fig.tight_layout(); out.figure("r2_hparams", fig, "Figure 7. How architectural each geometric measure is: R² on the five hyperparameters per dataset.", ["curvature_vs_hparams"]); plt.close(fig)


def fig_e1(L, out):
    plt = _mpl()
    P = L["e1_offline"][0]["payload"]; prim = P["primary"]; order = ["nslkdd", "kronodroid"] + [f"drebin_cluster fold {k}" for k in range(5)]
    by = {}
    for r in P["results"]:
        key = f"{r['dataset']} fold {r['fold']}" if r["dataset"] == "drebin_cluster" else r["dataset"]; by.setdefault(key, []).append(r)
    arms = [("V+Tr", "V + Tr(H²)", BLUE), ("V+lmax", "V + λmax", VIOLET), ("V+S", "V + sharpness ρ=0.1", ORANGE), ("V+P", "V + random objective (placebo)", GRAY)]
    labels, ys = [], []; fig, ax = plt.subplots(figsize=(7.2, 5.6)); row = 0
    for sc in order:
        rs = sorted(by[sc], key=lambda r: r["seed"])
        for r in rs:
            run = [x for x in r["runs"] if x["budget"] == min(prim["budget"], r["n_pool"]) and x["w"] == prim["w"]][0]
            for i, (arm, lab, col) in enumerate(arms):
                if arm not in run["arms"]:
                    continue
                a = run["arms"][arm]; yy = -row + (1.5 - i) * 0.18
                _errbar(ax, np.array([a["delta_vs_V"]]), np.array([yy]), np.array([a["ci95_delta"]]), col, label=lab if row == 0 else None, ms=4)
            labels.append(f"{sc.replace('drebin_cluster fold ', 'DREBIN f').replace('nslkdd', 'NSL-KDD').replace('kronodroid', 'KronoDroid')} s{r['seed']}"); ys.append(-row); row += 1
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(ys); ax.set_yticklabels(labels); ax.set_xlabel(f"Δ = shifted CVaR of the chosen configuration − validation-only choice (b = {prim['budget']}, w = {prim['w']:g}); negative = better")
    ax.legend(fontsize=7, loc="upper right"); ax.set_xlim(-0.25, 1.05); ax.set_axisbelow(True)
    fig.tight_layout(); out.figure("e1_offline", fig, "Figure 8. Offline tuner criterion: adding a geometric second objective never selects better than validation alone; bars are pool-bootstrap 95% CIs.", ["e1_offline"]); plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Tabelas e figuras do artigo a partir dos JSONs oficiais.")
    ap.add_argument("--root", default=str(ROOT)); ap.add_argument("--out", default="results/paper"); ap.add_argument("--B2", type=int, default=1000)
    ap.add_argument("--no-check", action="store_true", help="nao confere commits/dirty (desenvolvimento)"); ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args(argv); t0 = time.time(); root = pathlib.Path(args.root)
    L, manifest = check_sources(root, check=not args.no_check)
    out = Out(root / args.out if not os.path.isabs(args.out) else pathlib.Path(args.out))
    runs = e0_runs(L, root, args, measures_index(L))
    tab_scenarios(L, runs, out); tab_main(L, runs, out); tab_channels(L, runs, out); tab_measures(L, runs, out); tab_within(L, out); tab_halves(L, out); tab_e1(L, out); tab_f1(L, out); tab_cifar(L, out); tab_r2(L, out)
    if not args.no_figs:
        fig_forest(runs, L, out); fig_drebin(runs, out); fig_heatmap(L, out); fig_within(L, out); fig_f1(L, out); fig_r2(L, out); fig_e1(L, out)
    out.num("_provenance.two_level_ci", {r["src"]: r["ci_two_level_prov"] for r in runs}); out.num("_provenance.B2", args.B2); out.num("_provenance.sources_checked", not args.no_check)
    (out.dir / "numbers.json").write_text(json.dumps({"numbers": out.numbers, "tables": out.tables}, indent=1, ensure_ascii=False), encoding="utf-8")
    man = ["# Manifest — results/paper", "", f"Gerado por `scripts/paper_tables.py` em {time.strftime('%Y-%m-%d %H:%M')} (conferencia de commits: {'sim' if not args.no_check else 'NAO'}; B2 = {args.B2}).", "", "## Fontes (JSON, commit, dirty)", ""]
    man += [f"- `{key}`: `{path}` — {commit}{' SUJO' if dirty else ''}" for key, path, commit, dirty in manifest]
    man += ["", "## Saidas", ""] + [f"- `{name}` — {cap} — fontes: {', '.join(srcs)}" for name, cap, srcs in out.manifest]
    man += ["", "## IC em dois niveis: origem por rodada", ""] + [f"- {r['src']}: {r['ci_two_level_prov']}" for r in runs]
    (out.dir / "MANIFEST.md").write_text("\n".join(man) + "\n", encoding="utf-8")
    print(f"[paper_tables] {len(out.tables)} tabelas, {sum(1 for n, _, _ in out.manifest if n.startswith('fig_'))} figuras, {len(out.numbers)} numeros em {out.dir} ({time.time() - t0:.0f} s)")
    missing = [r["src"] for r in runs if r["ci_two_level_prov"] != "npz"]
    if missing:
        print(f"[paper_tables] AVISO: {len(missing)} rodada(s) sem perdas gravadas (results/raw): IC em dois niveis vindo do e0_measures ou ausente -> " + ", ".join(missing))


if __name__ == "__main__":
    main()
