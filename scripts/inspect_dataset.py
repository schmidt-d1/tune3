#!/usr/bin/env python
# scripts/inspect_dataset.py
"""
F3, passo (a): INSPECAO de um dataset tabular novo (ex.: KronoDroid) SEM treinar nada.

Pelo pre-registro da fase F, a regra de divisao temporal (coluna de tempo, ano de corte, atributos,
tamanho da validacao) so' e' fixada DEPOIS de ver a estrutura do dataset -- e antes de qualquer
modelo. Este script imprime e grava o que e' preciso para essa decisao:
  - arquivos encontrados (CSV, CSV dentro de ZIP), forma de cada um;
  - colunas: total, as primeiras N, tipos, fracao de faltantes;
  - candidatas a ROTULO (nome contendo label/class/malware/family/category, ou colunas binarias com
    poucos valores) e a distribuicao de valores;
  - candidatas a TEMPO (nome contendo time/date/year/first/last/seen/stamp, ou valores que parecem
    datas/epochs) e, para cada uma, a contagem por ANO x rotulo;
  - grupos de atributos por prefixo do nome (para escolher "estaticos", "dinamicos" etc.).
Grava results/inspect/<ts>_<commit>_<tag>.json com o mesmo conteudo.

Uso:
    python scripts/inspect_dataset.py data/kronodroid --tag kronodroid
    python scripts/inspect_dataset.py data/kronodroid/real_device.csv --label-hint malware --time-hint first --max-rows 200000
"""
from __future__ import annotations

import argparse, io, os, re, time, zipfile
from collections import Counter

import numpy as np
import pandas as pd

from tune3.experiments.results_io import save_result

LABEL_PAT = re.compile(r"label|class|malware|family|categor|target|benign|type", re.I)
TIME_PAT = re.compile(r"time|date|year|first|last|seen|stamp|dex|modif|submi|creat", re.I)


def _files(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, names in os.walk(path):
        for n in sorted(names):
            if n.lower().endswith((".csv", ".zip", ".csv.gz", ".tsv")):
                out.append(os.path.join(root, n))
    return out


def _read(path, max_rows):
    """Devolve lista de (nome, DataFrame). ZIPs sao lidos sem extrair."""
    if path.lower().endswith(".zip"):
        out = []
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith((".csv", ".tsv")):
                    with z.open(n) as f:
                        out.append((f"{os.path.basename(path)}::{n}", pd.read_csv(io.BytesIO(f.read()), nrows=max_rows, low_memory=False,
                                                                                    sep="\t" if n.lower().endswith(".tsv") else ",")))
        return out
    sep = "\t" if path.lower().endswith(".tsv") else ","
    return [(os.path.basename(path), pd.read_csv(path, nrows=max_rows, low_memory=False, sep=sep))]


def _to_datetime(s: pd.Series):
    """Tenta interpretar a coluna como data: epoch (s ou ms) ou texto de data."""
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if v.empty:
            return None
        med = float(v.median())
        if 1e11 < med < 1e13:                       # epoch em ms
            return pd.to_datetime(s, unit="ms", errors="coerce")
        if 1e8 < med < 1e11:                        # epoch em s
            return pd.to_datetime(s, unit="s", errors="coerce")
        if 1990 <= med <= 2100 and v.nunique() < 200:   # ja' e' um ano
            return pd.to_datetime(s.astype("Int64").astype(str), format="%Y", errors="coerce")
        return None
    try:
        d = pd.to_datetime(s.astype(str), errors="coerce", utc=True, format="mixed")
        return d if d.notna().mean() > 0.5 else None
    except Exception:  # noqa: BLE001
        return None


def inspect_df(name, df, args):
    info = {"file": name, "shape": list(df.shape), "n_columns": int(df.shape[1]),
            "first_columns": list(map(str, df.columns[: args.max_cols])),
            "dtypes": dict(Counter(str(t) for t in df.dtypes)),
            "missing_frac_max": float(df.isna().mean().max()) if df.shape[1] else 0.0}
    print(f"\n### {name}: {df.shape[0]} linhas x {df.shape[1]} colunas; tipos {info['dtypes']}; faltantes max {info['missing_frac_max']:.1%}")
    print("  primeiras colunas:", ", ".join(info["first_columns"]), ("..." if df.shape[1] > args.max_cols else ""))

    # rotulo
    cands = [c for c in df.columns if LABEL_PAT.search(str(c)) or (args.label_hint and args.label_hint.lower() in str(c).lower())]
    cands += [c for c in df.columns if c not in cands and df[c].nunique(dropna=True) == 2 and str(c).lower() in ("y", "target", "is_malware")]
    info["label_candidates"] = {}
    for c in cands[:12]:
        vc = df[c].value_counts(dropna=False).head(12)
        info["label_candidates"][str(c)] = {str(k): int(v) for k, v in vc.items()}
        print(f"  rotulo? {c!s:30s} n_unicos={df[c].nunique(dropna=True)}  top: {dict(list(info['label_candidates'][str(c)].items())[:6])}")

    # tempo
    info["time_candidates"] = {}
    tcands = [c for c in df.columns if TIME_PAT.search(str(c)) or (args.time_hint and args.time_hint.lower() in str(c).lower())]
    label_col = args.label_col or (cands[0] if cands else None)
    for c in tcands[:12]:
        d = _to_datetime(df[c])
        if d is None:
            print(f"  tempo? {c!s:30s} nao parece data (dtype {df[c].dtype}; ex.: {df[c].dropna().astype(str).head(3).tolist()})"); continue
        yrs = d.dt.year
        ok = yrs.notna().mean(); lo, hi = (int(yrs.min()), int(yrs.max())) if ok else (None, None)
        tab = {}
        if label_col is not None and label_col in df.columns:
            ct = pd.crosstab(yrs.fillna(-1).astype(int), df[label_col].astype(str))
            tab = {str(y): {str(k): int(v) for k, v in row.items()} for y, row in ct.iterrows()}
        else:
            tab = {str(int(y)): int(n) for y, n in yrs.value_counts().sort_index().items()}
        info["time_candidates"][str(c)] = {"parseable_frac": float(ok), "year_min": lo, "year_max": hi, "by_year": tab}
        print(f"  tempo  {c!s:30s} interpretavel em {ok:.0%}; anos {lo}-{hi}")
        for y, row in list(tab.items())[:40]:
            print(f"      {y}: {row}")

    # grupos de atributos por prefixo
    pref = Counter()
    for c in df.columns:
        s = str(c); m = re.match(r"([A-Za-z]+)[_\.:-]", s)
        pref[(m.group(1).lower() if m else s.split("_")[0].lower()[:12])] += 1
    info["feature_groups_by_prefix"] = dict(pref.most_common(25))
    print("  grupos por prefixo (top 25):", info["feature_groups_by_prefix"])
    num = df.select_dtypes(include=[np.number])
    if num.shape[1]:
        binary = int(((num.min() == 0) & (num.max() == 1)).sum())
        info["numeric_columns"] = int(num.shape[1]); info["binary_columns"] = binary
        print(f"  colunas numericas: {num.shape[1]} (binarias 0/1: {binary})")
    return info


def main():
    ap = argparse.ArgumentParser(description="Inspeciona um dataset tabular novo sem treinar nada (F3, passo a).")
    ap.add_argument("path"); ap.add_argument("--tag", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="limite de linhas lidas por arquivo (None = tudo)")
    ap.add_argument("--max-cols", type=int, default=40)
    ap.add_argument("--label-hint", default=None); ap.add_argument("--label-col", default=None); ap.add_argument("--time-hint", default=None)
    args = ap.parse_args(); t0 = time.time()
    files = _files(args.path)
    if not files:
        print(f"nenhum CSV/ZIP em {args.path}"); return
    print(f"[inspecao] {len(files)} arquivo(s) em {args.path}")
    out = {"path": args.path, "files": []}
    for f in files:
        try:
            for name, df in _read(f, args.max_rows):
                out["files"].append(inspect_df(name, df, args))
        except Exception as e:  # noqa: BLE001
            print(f"  [erro] {f}: {e}"); out["files"].append({"file": f, "error": str(e)})
    save_result(out, experiment="inspect", tag=args.tag, args=vars(args), started_at=t0)


if __name__ == "__main__":
    main()
