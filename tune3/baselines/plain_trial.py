# tune3/baselines/plain_trial.py
"""
Trial SIMPLES para baselines -- treina SEM os componentes MICRO do Tune3
(sem DDKF, sem gating, sem EoS; lr FIXO). Mesma fabrica de modelo, mesmos dados,
mesmo objetivo (CVaR) e mesma curvatura final -> comparacao justa com o Tune3.

Suporta otimizadores: 'sgd', 'adam', 'sam'. Reporta perdas intermediarias de
validacao por epoca (necessario para o early-stopping do ASHA).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tune3.models.factory import build_model
from tune3.curvature import HutchinsonEstimator, HutchinsonConfig
from tune3.core.objectives import cvar
from tune3.baselines.sam import SAM


@dataclass
class PlainTrialConfig:
    max_epochs: int = 50
    batch_size: int = 256
    arch: str = "mlp"
    gamma: float = 0.95
    device: str = "cpu"
    seed: int = 0
    optimizer: str = "sgd"       # 'sgd' | 'adam' | 'sam'
    momentum: float = 0.9
    sam_rho: float = 0.05
    use_class_weight: bool = True
    curvature_probes: int = 5


def _loader(X, y, bs, shuffle):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def _class_weights(y, k, device):
    c = np.bincount(np.asarray(y).astype(int), minlength=k).astype(float)
    c = np.clip(c, 1.0, None)
    return torch.tensor(c.sum() / (k * c), dtype=torch.float32, device=device)


def plain_trial(
    hparams: Dict[str, float],
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: Optional[PlainTrialConfig] = None,
    max_epochs_override: Optional[int] = None,
    report_intermediate: bool = False,
) -> Dict:
    cfg = config or PlainTrialConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    n_epochs = max_epochs_override or cfg.max_epochs

    X_tr, y_tr, X_val, y_val = data
    in_dim = X_tr.shape[1]
    k = int(max(y_tr.max(), y_val.max())) + 1
    lr = float(hparams.get("learning_rate", 1e-3))
    wd = float(hparams.get("weight_decay", 0.0))
    arch_hp = {"hidden_dim": int(hparams.get("hidden_dim", 128)),
               "n_layers": int(hparams.get("n_layers", 2)),
               "dropout": float(hparams.get("dropout", 0.1))}

    model = build_model(in_dim, k, arch=cfg.arch, hparams=arch_hp).to(device)
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=wd)
        is_sam = False
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        is_sam = False
    elif cfg.optimizer == "sam":
        opt = SAM(model.parameters(), torch.optim.SGD, rho=cfg.sam_rho,
                  lr=lr, momentum=cfg.momentum, weight_decay=wd)
        is_sam = True
    else:
        raise ValueError(f"optimizer desconhecido: {cfg.optimizer}")

    w = _class_weights(y_tr, k, device) if cfg.use_class_weight else None
    crit = nn.CrossEntropyLoss(weight=w)
    crit_ps = nn.CrossEntropyLoss(weight=w, reduction="none")
    tr_loader = _loader(X_tr, y_tr, cfg.batch_size, True)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=device)

    inter: List[float] = []
    for ep in range(n_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            if is_sam:
                crit(model(xb), yb).backward()
                opt.first_step(zero_grad=True)
                crit(model(xb), yb).backward()
                opt.second_step(zero_grad=True)
            else:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
        if report_intermediate:
            model.eval()
            with torch.no_grad():
                inter.append(float(crit(model(Xv), yv)))

    model.eval()
    with torch.no_grad():
        val_ps = crit_ps(model(Xv), yv).cpu().numpy()
        val_loss = float(crit(model(Xv), yv))

    # curvatura final (mesma metrica do Tune3)
    xb, yb = next(iter(tr_loader)); xb, yb = xb.to(device), yb.to(device)
    model.train()
    curv = HutchinsonEstimator(HutchinsonConfig(num_probes=cfg.curvature_probes)).estimate(
        model, crit(model(xb), yb))

    cv = float(cvar(val_ps, cfg.gamma)) if np.isfinite(val_loss) else 1e3
    return {"cvar": cv, "curvature": float(curv) if np.isfinite(curv) else 1e3,
            "final_val_loss": val_loss, "intermediate_val_losses": inter}
