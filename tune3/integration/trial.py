# tune3/integration/trial.py
"""
Trial de treino do Tune3 -- integracao de todos os componentes.
ATUALIZADO (Fase 7): early stopping, dataset residente na GPU, batches grandes,
TF32, e CVaR avaliado na MELHOR epoca (nao na ultima).
CORRIGIDO (set/2026, auditoria):
  (a) EoSDetector so' recebe a curvatura quando ela e' RE-ESTIMADA. Antes, recebia
      o mesmo valor repetido `curvature_every` vezes, e a condicao de "subidas
      estritamente consecutivas" nunca era satisfeita: o detector NUNCA disparava
      (n_eos_triggers == 0 em todos os experimentos anteriores).
  (b) Hutchinson roda em model.eval(): com dropout ativo o estimador media a
      curvatura de uma sub-rede aleatoria (viés e variancia extras).
  (c) Flag `ddkf_enabled` (ablacao E2) + telemetria de engajamento do controlador.

Dado hiperparametros (do MACRO/BO), treina o modelo com:
  - DDKFController ajustando o log-lr ONLINE (gated por RegimeGate);
  - HutchinsonEstimator medindo Tr(H^2) (batch de treino, nunca teste);
  - GSNREstimator (3a observacao do DDKF);
  - CantelliGuard (aborto em spike catastrofico);
  - EoSDetector (reduz lr em progressive sharpening);
  - class_weight (desbalanceamento, coerente com CVaR);
  - EARLY STOPPING (paciencia): para se val_loss nao melhora.
Retorna (CVaR_gamma da MELHOR epoca de validacao, Tr(H^2)) + telemetria.

OBSERVACAO DO DDKF (decisao de projeto, documentada): z_t = (train_loss,
val_loss, log GSNR). A curvatura Tr(H^2) NAO entra no vetor de observacao do
filtro; ela atua (i) como objetivo do nivel MACRO e (ii) via EoSDetector.
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
    # A curvatura e' re-estimada a cada `curvature_every` epocas. O EoSDetector
    # conta "passos" em unidades de ESTIMATIVAS (window=6, patience=4 => precisa
    # de >= 6 estimativas = 6*curvature_every epocas antes de poder disparar).
    curvature_every: int = 5
    # Ablacao E2: False => lr fixo em lr0 (Cantelli/EoS continuam ativos).
    ddkf_enabled: bool = True


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
    best_val = np.inf; best_val_ps = None; best_curv = 0.0; best_epoch = -1; wait = 0
    aborted = False
    # telemetria de engajamento (E2): o controlador realmente mexeu no lr?
    lr_traj = []; n_ddkf_active = 0; n_curv_estimates = 0
    curv_traj = []

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

        # --- curvatura: re-estimada esparsamente; EoS so' ve' valores NOVOS ---
        eos_fired = False
        if epoch % cfg.curvature_every == 0 or epoch == cfg.max_epochs - 1:
            idx = perm[:bs]                    # batch de TREINO (novo a cada epoca)
            model.eval()                       # sem dropout: curvatura da rede determinística
            last_curv = hutch.estimate(model, crit(model(Xtr_t[idx]), ytr_t[idx]))
            model.train()
            n_curv_estimates += 1; curv_traj.append(float(last_curv))
            eos_fired = eos.update(last_curv)

        # --- controle MICRO do lr ---
        log_gsnr = gsnr_est.log_gsnr(grads_for_gsnr) if len(grads_for_gsnr) >= 2 else 0.0
        ddkf.update(np.array([train_loss, val_loss, log_gsnr]))
        r2 = ddkf.information_ratio(); factor = gate.gating_factor(r2); gate.observe(r2)
        cur_lr = max(_current_lr(opt), 1e-12)
        if cfg.ddkf_enabled:
            new_log_lr = ddkf.x_hat if factor > 0 else np.log(lr0)
            applied = (1 - factor) * np.log(cur_lr) + factor * new_log_lr
            if factor > 0: n_ddkf_active += 1
        else:
            applied = np.log(cur_lr)           # ablacao: DDKF nao atua
        new_lr = float(np.exp(applied))
        if eos_fired:
            new_lr *= eos.suggested_lr_factor()
        _set_lr(opt, new_lr)
        lr_traj.append(new_lr)

        if cantelli.should_abort(val_loss):
            aborted = True; break

        # --- EARLY STOPPING: rastreia melhor epoca (VALIDACAO) ---
        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss; best_val_ps = val_ps; best_curv = last_curv
            best_epoch = epoch; wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                logger.info("early stopping", epoch=epoch, best_val=round(best_val, 4))
                break

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                           "lr": new_lr, "trace_h2": last_curv, "R2": r2})

    lr_arr = np.asarray(lr_traj, dtype=float)
    telemetry = {
        "regime_fractions": gate.fractions(),
        "n_eos_triggers": eos.n_triggers,
        "n_epochs_run": len(lr_traj),
        "n_ddkf_active_epochs": n_ddkf_active,
        "n_curvature_estimates": n_curv_estimates,
        "lr0": lr0,
        "lr_final": float(lr_arr[-1]) if lr_arr.size else lr0,
        "lr_min": float(lr_arr.min()) if lr_arr.size else lr0,
        "lr_max": float(lr_arr.max()) if lr_arr.size else lr0,
        # fracao de epocas em que o lr efetivamente mudou (>0.1% relativo)
        "lr_change_fraction": float(np.mean(np.abs(np.diff(np.log(lr_arr))) > 1e-3)) if lr_arr.size > 1 else 0.0,
        "curvature_trajectory": curv_traj,
    }

    if aborted or best_val_ps is None:
        return {"cvar": cfg.abort_penalty, "curvature": cfg.abort_penalty,
                "aborted": True, **telemetry}

    # CVaR na MELHOR epoca de validacao (consistente com early stopping)
    return {"cvar": float(cvar(best_val_ps, cfg.gamma)),
            "curvature": float(best_curv),
            "aborted": False, "final_val_loss": float(best_val),
            "best_epoch": int(best_epoch), **telemetry}
