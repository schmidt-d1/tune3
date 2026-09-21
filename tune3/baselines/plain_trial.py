# tune3/baselines/plain_trial.py
"""Trial SIMPLES para baselines -- early stopping, GPU-resident, best-epoch CVaR.

CORRECAO (set/2026, auditoria): separacao estrita validacao x teste.
  - `data` = (X_tr, y_tr, X_val, y_val): o early stopping e a escolha da melhor
    epoca usam APENAS a validacao. O "cvar" devolvido e' o CVaR de VALIDACAO na
    melhor epoca (e' o que RS/ASHA/B4 usam para selecionar configuracoes).
  - `test_data` = (X_te, y_te) OPCIONAL: se fornecido, o modelo da MELHOR EPOCA
    DE VALIDACAO (estado restaurado) e' avaliado UMA vez no teste, devolvendo
    "cvar_test", "test_loss" e (se return_scores) "scores_test".
    O teste NUNCA influencia early stopping, escolha de epoca ou curvatura.
  - A curvatura Tr(H^2) e' medida no modelo da melhor epoca, em model.eval()
    (dropout desligado -- o estimador mede a rede determinística, nao uma
    sub-rede aleatoria), sobre um batch de TREINO.

Antes desta correcao, protocol.py/security_protocol.py passavam o TESTE no lugar
da validacao, de modo que a melhor epoca era escolhida olhando o teste
(vazamento). Os numeros produzidos por aquela versao nao devem ser reportados.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tune3.models.factory import build_model
from tune3.curvature import HutchinsonEstimator, HutchinsonConfig
from tune3.core.objectives import cvar
from tune3.baselines.sam import SAM


@dataclass
class PlainTrialConfig:
    max_epochs: int = 100
    batch_size: int = 4096
    arch: str = "mlp"
    gamma: float = 0.95
    device: str = "cpu"
    seed: int = 0
    optimizer: str = "sgd"
    momentum: float = 0.9
    sam_rho: float = 0.05
    patience: int = 10
    min_delta: float = 1e-4
    enable_tf32: bool = True
    use_class_weight: bool = True
    curvature_probes: int = 5


def _class_weights(y, k, device):
    c = np.bincount(np.asarray(y).astype(int), minlength=k).astype(float)
    c = np.clip(c, 1.0, None)
    return torch.tensor(c.sum() / (k * c), dtype=torch.float32, device=device)


@torch.no_grad()
def _eval_split(model, crit, crit_ps, X, y, want_scores: bool):
    """Avalia um split em model.eval(): (loss media, perdas por amostra, prob. classe 1)."""
    model.eval()
    out = model(X)
    loss = float(crit(out, y))
    ps = crit_ps(out, y).cpu().numpy()
    scores = F.softmax(out, dim=1)[:, 1].cpu().numpy() if want_scores else None
    return loss, ps, scores


def plain_trial(hparams, data, config=None, max_epochs_override=None,
                report_intermediate=False, return_scores=False,
                test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
    cfg = config or PlainTrialConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    if cfg.enable_tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    n_epochs = max_epochs_override or cfg.max_epochs

    X_tr, y_tr, X_val, y_val = data
    in_dim = X_tr.shape[1]
    k = int(max(y_tr.max(), y_val.max())) + 1
    lr = float(hparams.get("learning_rate", 1e-3)); wd = float(hparams.get("weight_decay", 0.0))
    arch_hp = {"hidden_dim": int(hparams.get("hidden_dim", 128)),
               "n_layers": int(hparams.get("n_layers", 2)),
               "dropout": float(hparams.get("dropout", 0.1))}

    model = build_model(in_dim, k, arch=cfg.arch, hparams=arch_hp).to(device)
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=wd); is_sam = False
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd); is_sam = False
    elif cfg.optimizer == "sam":
        opt = SAM(model.parameters(), torch.optim.SGD, rho=cfg.sam_rho, lr=lr, momentum=cfg.momentum, weight_decay=wd); is_sam = True
    else:
        raise ValueError(f"optimizer desconhecido: {cfg.optimizer}")

    w = _class_weights(y_tr, k, device) if cfg.use_class_weight else None
    crit = nn.CrossEntropyLoss(weight=w); crit_ps = nn.CrossEntropyLoss(weight=w, reduction="none")

    Xtr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=device)
    n = Xtr_t.shape[0]; bs = min(cfg.batch_size, n)

    inter: List[float] = []
    best_val = np.inf; best_val_ps = None; best_scores = None; best_state = None
    best_epoch = -1; wait = 0
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]; xb, yb = Xtr_t[idx], ytr_t[idx]
            if is_sam:
                crit(model(xb), yb).backward(); opt.first_step(zero_grad=True)
                crit(model(xb), yb).backward(); opt.second_step(zero_grad=True)
            else:
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        vl, vps, vprobs = _eval_split(model, crit, crit_ps, Xv, yv, return_scores)
        if report_intermediate:
            inter.append(vl)
        if vl < best_val - cfg.min_delta:
            best_val = vl; best_val_ps = vps; best_scores = vprobs; best_epoch = ep; wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is None:           # nenhuma epoca melhorou (ex.: NaN desde o inicio)
        best_val_ps = vps; best_val = vl; best_scores = vprobs; best_epoch = ep
        best_state = copy.deepcopy(model.state_dict())

    # --- restaura o modelo da MELHOR EPOCA DE VALIDACAO ---
    model.load_state_dict(best_state)

    # curvatura no modelo da melhor epoca, em eval() (sem dropout), batch de TREINO
    perm = torch.randperm(n, device=device)[:bs]
    model.eval()
    curv = HutchinsonEstimator(HutchinsonConfig(num_probes=cfg.curvature_probes)).estimate(
        model, crit(model(Xtr_t[perm]), ytr_t[perm]))

    cv = float(cvar(best_val_ps, cfg.gamma)) if np.isfinite(best_val) else 1e3
    out = {"cvar": cv, "curvature": float(curv) if np.isfinite(curv) else 1e3,
           "final_val_loss": float(best_val), "best_epoch": int(best_epoch),
           "intermediate_val_losses": inter}
    if return_scores:
        out["scores"] = best_scores

    # --- avaliacao UNICA no teste, so' se pedida, com o modelo ja escolhido ---
    if test_data is not None:
        X_te, y_te = test_data
        Xte = torch.as_tensor(X_te, dtype=torch.float32, device=device)
        yte = torch.as_tensor(y_te, dtype=torch.long, device=device)
        tl, tps, tscores = _eval_split(model, crit, crit_ps, Xte, yte, return_scores)
        out["test_loss"] = float(tl)
        out["cvar_test"] = float(cvar(tps, cfg.gamma)) if np.isfinite(tl) else 1e3
        if return_scores:
            out["scores_test"] = tscores
    return out
