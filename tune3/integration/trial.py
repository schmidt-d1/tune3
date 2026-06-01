# tune3/integration/trial.py
"""
Trial de treino do Tune3 -- integracao de todos os componentes.
ATUALIZADO (Fase 7): early stopping, dataset residente na GPU, batches grandes,
TF32, e CVaR avaliado na MELHOR epoca (nao na ultima).

Dado hiperparametros (do MACRO/BO), treina o modelo com:
  - DDKFController ajustando o log-lr ONLINE (gated por RegimeGate);
  - HutchinsonEstimator medindo Tr(H^2) (treino, nunca teste);
  - GSNREstimator (3a observacao do DDKF);
  - CantelliGuard (aborto em spike catastrofico);
  - EoSDetector (reduz lr em progressive sharpening);
  - class_weight (desbalanceamento, coerente com CVaR);
  - EARLY STOPPING (paciencia): para se val_loss nao melhora.
Retorna (CVaR_gamma da MELHOR epoca, Tr(H^2)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import structlog

from tune3.models.factory import build_model
from tune3.curvature import HutchinsonEstimator, HutchinsonConfig, GSNREstimator, GSNRConfig
from tune3.micro import DDKFController, DDKFConfig, RegimeGate, RegimeGatingConfig
from tune3.safety import CantelliGuard, CantelliConfig, EoSDetector, EoSConfig
from tune3.core.objectives import cvar

logger = structlog.get_logger()


@dataclass
class TrialConfig:
    max_epochs: int = 100
    batch_size: int = 4096          # grande: dataset cabe na GPU (A100)
    arch: str = "mlp"
    gamma: float = 0.95
    device: str = "cpu"
    seed: int = 0
    optimizer: str = "sgd"
    momentum: float = 0.9
    patience: int = 10              # early stopping: epocas sem melhora ate parar
    min_delta: float = 1e-4         # melhora minima de val_loss p/ resetar paciencia
    enable_tf32: bool = True        # tensor cores da A100
    gpu_resident: bool = True       # mantem dataset na GPU (evita transferencias)
    # controle Tune3 (FIXOS; variados so na Frente F)
    ddkf: DDKFConfig = field(default_factory=lambda: DDKFConfig(rho=0.95, sigma_eta=0.05))
    gating: RegimeGatingConfig = field(default_factory=RegimeGatingConfig)
    cantelli: CantelliConfig = field(default_factory=lambda: CantelliConfig(gamma=0.95, window_size=10))
    eos: EoSConfig = field(default_factory=EoSConfig)
    hutchinson: HutchinsonConfig = field(default_factory=lambda: HutchinsonConfig(num_probes=5))
    gsnr_batches: int = 4
    abort_penalty: float = 1e3
    use_class_weight: bool = True
    curvature_every: int = 5


def _class_weights(y, k, device):
    c = np.bincount(np.asarray(y).astype(int), minlength=k).astype(float)
    c = np.clip(c, 1.0, None)
    return torch.tensor(c.sum() / (k * c), dtype=torch.float32, device=device)


def _flat_grad(model) -> np.ndarray:
    gs = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    return torch.cat(gs).cpu().numpy() if gs else np.zeros(1)


def _current_lr(opt): return opt.param_groups[0]["lr"]
def _set_lr(opt, lr):
    for g in opt.param_groups: g["lr"] = lr


def run_trial(hparams, data, config=None, wandb_run=None) -> Dict:
    cfg = config or TrialConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    if cfg.enable_tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # tensor cores A100

    X_tr, y_tr, X_val, y_val = data
    in_dim = X_tr.shape[1]
    k = int(max(y_tr.max(), y_val.max())) + 1
    lr0 = float(hparams.get("learning_rate", 1e-3)); wd = float(hparams.get("weight_decay", 0.0))
    arch_hp = {"hidden_dim": int(hparams.get("hidden_dim", 128)),
               "n_layers": int(hparams.get("n_layers", 2)),
               "dropout": float(hparams.get("dropout", 0.1))}

    model = build_model(in_dim, k, arch=cfg.arch, hparams=arch_hp).to(device)
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=cfg.momentum, weight_decay=wd)
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=wd)
    else:
        raise ValueError(f"optimizer desconhecido: {cfg.optimizer}")

    weight = _class_weights(y_tr, k, device) if cfg.use_class_weight else None
    crit = nn.CrossEntropyLoss(weight=weight)
    crit_ps = nn.CrossEntropyLoss(weight=weight, reduction="none")

    # --- dataset RESIDENTE na GPU (otimizacao A100) ---
    Xtr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=device)
    n = Xtr_t.shape[0]
    bs = min(cfg.batch_size, n)

    ddkf = DDKFController(x0=np.log(lr0), config=cfg.ddkf, rng=np.random.default_rng(cfg.seed))
    gate = RegimeGate(cfg.gating); cantelli = CantelliGuard(cfg.cantelli)
    eos = EoSDetector(cfg.eos); hutch = HutchinsonEstimator(cfg.hutchinson)
    gsnr_est = GSNREstimator(GSNRConfig(num_batches=cfg.gsnr_batches))

    last_curv = 0.0
    best_val = np.inf; best_val_ps = None; best_curv = 0.0; wait = 0
    aborted = False

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_losses = []; grads_for_gsnr = []
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            if len(grads_for_gsnr) < cfg.gsnr_batches:
                grads_for_gsnr.append(_flat_grad(model))
            opt.step()
            epoch_losses.append(float(loss.detach()))

        train_loss = float(np.mean(epoch_losses))
        if not np.isfinite(train_loss):
            aborted = True; break

        model.eval()
        with torch.no_grad():
            val_out = model(Xv)
            val_loss = float(crit(val_out, yv).detach())
            val_ps = crit_ps(val_out, yv).cpu().numpy()

        if epoch % cfg.curvature_every == 0 or epoch == cfg.max_epochs - 1:
            idx = perm[:bs]
            model.train()
            last_curv = hutch.estimate(model, crit(model(Xtr_t[idx]), ytr_t[idx]))

        log_gsnr = gsnr_est.log_gsnr(grads_for_gsnr) if len(grads_for_gsnr) >= 2 else 0.0
        ddkf.update(np.array([train_loss, val_loss, log_gsnr]))
        r2 = ddkf.information_ratio(); factor = gate.gating_factor(r2); gate.observe(r2)
        new_log_lr = ddkf.x_hat if factor > 0 else np.log(lr0)
        applied = (1 - factor) * np.log(max(_current_lr(opt), 1e-12)) + factor * new_log_lr
        new_lr = float(np.exp(applied))
        if eos.update(last_curv):
            new_lr *= eos.suggested_lr_factor()
        _set_lr(opt, new_lr)

        if cantelli.should_abort(val_loss):
            aborted = True; break

        # --- EARLY STOPPING: rastreia melhor epoca ---
        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss; best_val_ps = val_ps; best_curv = last_curv; wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                logger.info("early stopping", epoch=epoch, best_val=round(best_val, 4))
                break

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                           "lr": new_lr, "trace_h2": last_curv, "R2": r2})

    if aborted or best_val_ps is None:
        return {"cvar": cfg.abort_penalty, "curvature": cfg.abort_penalty,
                "aborted": True, "regime_fractions": gate.fractions()}

    # CVaR na MELHOR epoca (consistente com early stopping)
    return {"cvar": float(cvar(best_val_ps, cfg.gamma)),
            "curvature": float(best_curv),
            "aborted": False, "final_val_loss": float(best_val),
            "regime_fractions": gate.fractions(), "n_eos_triggers": eos.n_triggers}
