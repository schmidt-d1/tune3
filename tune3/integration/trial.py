# tune3/integration/trial.py
"""
Trial de treino do Tune3 -- a INTEGRACAO de todos os componentes.

Dado um conjunto de hiperparametros (sugeridos pelo MACRO/BO), este modulo:
  1. constroi o modelo (fabrica agnostica de arquitetura);
  2. treina no dataset com:
       - DDKFController ajustando o log-lr ONLINE (gated por RegimeGate);
       - HutchinsonEstimator medindo Tr(H^2) (TREINO, nunca teste);
       - GSNREstimator fornecendo a 3a observacao do DDKF;
       - CantelliGuard abortando em spikes catastroficos de val_loss;
       - EoSDetector reduzindo o lr ao detectar progressive sharpening;
       - class_weight no criterio (coerente com CVaR / desbalanceamento);
  3. retorna (CVaR_gamma das perdas por amostra de validacao, Tr(H^2) final).

Saidas para o MACRO: ambas MINIMIZADAS (CVaR e curvatura).
Em caso de aborto (Cantelli) ou divergencia (NaN), retorna penalidade alta.

Agnostico de dataset e arquitetura: recebe tensores e um config de arquitetura.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import structlog

from tune3.models.factory import build_model
from tune3.curvature import HutchinsonEstimator, HutchinsonConfig, GSNREstimator, GSNRConfig
from tune3.micro import DDKFController, DDKFConfig, RegimeGate, RegimeGatingConfig
from tune3.safety import CantelliGuard, CantelliConfig, EoSDetector, EoSConfig
from tune3.core.objectives import cvar

logger = structlog.get_logger()


@dataclass
class TrialConfig:
    # treino
    max_epochs: int = 50
    batch_size: int = 256
    arch: str = "mlp"
    gamma: float = 0.95            # nivel do CVaR
    device: str = "cpu"
    seed: int = 0
    optimizer: str = "sgd"         # "sgd" (lr e passo direto -> DDKF significativo) ou "adam"
    momentum: float = 0.9          # usado se optimizer="sgd"
    # controle Tune3 (FIXOS; variados so na Frente F de sensibilidade)
    ddkf: DDKFConfig = field(default_factory=lambda: DDKFConfig(rho=0.95, sigma_eta=0.05))
    gating: RegimeGatingConfig = field(default_factory=RegimeGatingConfig)
    cantelli: CantelliConfig = field(default_factory=lambda: CantelliConfig(gamma=0.95, window_size=10))
    eos: EoSConfig = field(default_factory=EoSConfig)
    hutchinson: HutchinsonConfig = field(default_factory=lambda: HutchinsonConfig(num_probes=5))
    gsnr_batches: int = 4
    abort_penalty: float = 1e3     # CVaR retornado em caso de aborto/divergencia
    use_class_weight: bool = True
    curvature_every: int = 5       # medir Tr(H^2) a cada N epocas (custo controlado)


def _make_loader(X, y, batch_size, shuffle, device):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _class_weights(y, num_classes, device):
    counts = np.bincount(np.asarray(y).astype(int), minlength=num_classes).astype(float)
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / (num_classes * counts)   # peso inverso a frequencia
    return torch.tensor(w, dtype=torch.float32, device=device)


def _flat_grad(model) -> np.ndarray:
    gs = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    return torch.cat(gs).cpu().numpy() if gs else np.zeros(1)


def run_trial(
    hparams: Dict[str, float],
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: Optional[TrialConfig] = None,
    wandb_run=None,
) -> Dict:
    """
    hparams: {'learning_rate','weight_decay','dropout','hidden_dim','n_layers'}
    data: (X_train, y_train, X_val, y_val)
    Retorna dict com cvar, curvature, e telemetria.
    """
    cfg = config or TrialConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    X_tr, y_tr, X_val, y_val = data
    input_dim = X_tr.shape[1]
    num_classes = int(max(y_tr.max(), y_val.max())) + 1

    lr0 = float(hparams.get("learning_rate", 1e-3))
    wd = float(hparams.get("weight_decay", 0.0))
    arch_hp = {
        "hidden_dim": int(hparams.get("hidden_dim", 128)),
        "n_layers": int(hparams.get("n_layers", 2)),
        "dropout": float(hparams.get("dropout", 0.1)),
    }

    model = build_model(input_dim, num_classes, arch=cfg.arch, hparams=arch_hp).to(device)
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=cfg.momentum, weight_decay=wd)
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=wd)
    else:
        raise ValueError(f"optimizer desconhecido: {cfg.optimizer}")

    weight = _class_weights(y_tr, num_classes, device) if cfg.use_class_weight else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    criterion_persample = nn.CrossEntropyLoss(weight=weight, reduction="none")

    train_loader = _make_loader(X_tr, y_tr, cfg.batch_size, True, device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=device)

    # componentes Tune3
    ddkf = DDKFController(x0=np.log(lr0), config=cfg.ddkf, rng=np.random.default_rng(cfg.seed))
    gate = RegimeGate(cfg.gating)
    cantelli = CantelliGuard(cfg.cantelli)
    eos = EoSDetector(cfg.eos)
    hutch = HutchinsonEstimator(cfg.hutchinson)
    gsnr_est = GSNREstimator(GSNRConfig(num_batches=cfg.gsnr_batches))

    last_curv = 0.0
    prev_val = None
    aborted = False

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses = []
        grads_for_gsnr = []
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            # coletar gradientes p/ GSNR (alguns minibatches)
            if len(grads_for_gsnr) < cfg.gsnr_batches:
                grads_for_gsnr.append(_flat_grad(model))
            opt.step()
            epoch_losses.append(float(loss.detach()))

        train_loss = float(np.mean(epoch_losses))
        if not np.isfinite(train_loss):
            aborted = True
            logger.warning("Trial divergiu (NaN/inf)", epoch=epoch)
            break

        # validacao (perda media e por amostra)
        model.eval()
        with torch.no_grad():
            val_out = model(Xv)
            val_loss = float(criterion(val_out, yv).detach())
            val_persample = criterion_persample(val_out, yv).cpu().numpy()

        # curvatura Tr(H^2) em TREINO (a cada curvature_every epocas)
        if epoch % cfg.curvature_every == 0 or epoch == cfg.max_epochs - 1:
            xb, yb = next(iter(train_loader))
            xb, yb = xb.to(device), yb.to(device)
            model.train()
            curv_loss = criterion(model(xb), yb)  # COM grafo
            last_curv = hutch.estimate(model, curv_loss)

        # GSNR (3a observacao)
        log_gsnr = gsnr_est.log_gsnr(grads_for_gsnr) if len(grads_for_gsnr) >= 2 else 0.0

        # ----- DDKF: ajusta log-lr ONLINE, gated por regime -----
        ddkf.update(np.array([train_loss, val_loss, log_gsnr]))
        r2 = ddkf.information_ratio()
        factor = gate.gating_factor(r2)
        gate.observe(r2)
        # aplica a correcao do DDKF proporcional ao gating
        new_log_lr = ddkf.x_hat if factor > 0 else np.log(lr0) + 0.0
        # interpola: em Regime A (factor=0) mantem; em B/C usa o DDKF
        applied_log_lr = (1 - factor) * np.log(max(_current_lr(opt), 1e-12)) + factor * new_log_lr
        new_lr = float(np.exp(applied_log_lr))

        # ----- EoS: reduz lr se houver progressive sharpening -----
        if eos.update(last_curv):
            new_lr *= eos.suggested_lr_factor()

        _set_lr(opt, new_lr)

        # ----- Cantelli: aborto em spike catastrofico de val_loss -----
        if cantelli.should_abort(val_loss):
            aborted = True
            logger.warning("Cantelli abortou o trial", epoch=epoch, val_loss=val_loss)
            break

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                           "lr": new_lr, "trace_h2": last_curv, "R2": r2, "log_gsnr": log_gsnr})
        prev_val = val_loss

    # ----- objetivos para o MACRO -----
    if aborted:
        return {"cvar": cfg.abort_penalty, "curvature": cfg.abort_penalty,
                "aborted": True, "regime_fractions": gate.fractions()}

    cvar_val = cvar(val_persample, cfg.gamma)
    return {
        "cvar": float(cvar_val),
        "curvature": float(last_curv),
        "aborted": False,
        "final_val_loss": float(val_loss),
        "regime_fractions": gate.fractions(),
        "n_eos_triggers": eos.n_triggers,
    }


def _current_lr(opt) -> float:
    return opt.param_groups[0]["lr"]


def _set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr
