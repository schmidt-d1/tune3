# tune3/experiments/results_io.py
"""
Registro padronizado de resultados (para conferencia posterior pelo coordenador).

Todo script de experimento grava em:
    results/<experimento>/<AAAAMMDD-HHMMSS>_<commit>_<tag>.json

com um bloco "meta" fixo (quem rodou, quando, em qual commit, com qual ambiente,
com quais argumentos) e um bloco "payload" com o resultado bruto do script.
Os JSONs de results/ SAO versionados no git (excecao no .gitignore), de modo
que o aluno entrega o resultado com `git add results/ && git commit && git push`
e o coordenador confere no repositorio, sem depender de Drive/e-mail.

Uso:
    from tune3.experiments.results_io import save_result
    path = save_result(payload, experiment="s2_zeroday", tag="aluno-joao", args=vars(args))

`scripts/collect_results.py` varre results/ e monta a tabela-resumo.
"""
from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _git(cmd: list) -> Optional[str]:
    try:
        return subprocess.check_output(["git"] + cmd, stderr=subprocess.DEVNULL,
                                       text=True, timeout=5).strip()
    except Exception:
        return None


def git_state() -> Dict[str, Any]:
    """Commit, branch e se a arvore de trabalho esta suja (resultado nao reproduzivel)."""
    commit = _git(["rev-parse", "HEAD"])
    short = _git(["rev-parse", "--short", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "commit_short": short, "branch": branch,
            "dirty": bool(status) if status is not None else None,
            "dirty_files": status.splitlines() if status else []}


def environment_state() -> Dict[str, Any]:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "hostname": socket.gethostname(), "user": getpass.getuser()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    for mod in ("numpy", "scipy", "botorch", "gpytorch", "statsmodels", "sklearn"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def build_meta(experiment: str, tag: Optional[str], args: Optional[Dict], started_at: float) -> Dict:
    return {"experiment": experiment, "tag": tag,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - started_at, 1),
            "args": args or {}, "git": git_state(), "env": environment_state(),
            "schema_version": 1}


def save_result(payload: Any, experiment: str, tag: Optional[str] = None,
                args: Optional[Dict] = None, started_at: Optional[float] = None,
                out_dir: str = "results", extra_path: Optional[str] = None) -> str:
    """Grava results/<experiment>/<ts>_<commit>_<tag>.json e devolve o caminho.
    Se extra_path for dado (ex.: --out antigo), grava tambem uma copia la."""
    started_at = started_at or time.time()
    meta = build_meta(experiment, tag, args, started_at)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = meta["git"]["commit_short"] or "nogit"
    safe_tag = (tag or getpass.getuser()).replace(" ", "-").replace("/", "-")
    d = Path(out_dir) / experiment
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{ts}_{commit}_{safe_tag}.json"
    doc = {"meta": meta, "payload": payload}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=_json_default)
    if extra_path:
        with open(extra_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, default=_json_default)
    if meta["git"]["dirty"]:
        print(f"[AVISO] arvore git SUJA -- resultado gravado, mas nao e' reproduzivel a partir "
              f"do commit {commit}. Faca commit antes de rodar experimentos oficiais.")
    print(f"[salvo] {path}")
    return str(path)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_results(experiment: Optional[str] = None, out_dir: str = "results"):
    """Itera (path, doc) sobre os JSONs gravados por save_result."""
    root = Path(out_dir)
    if not root.exists():
        return
    pattern = f"{experiment}/*.json" if experiment else "*/*.json"
    for p in sorted(root.glob(pattern)):
        try:
            with open(p, encoding="utf-8") as f:
                yield str(p), json.load(f)
        except Exception as e:
            print(f"[ignorado] {p}: {e}")
