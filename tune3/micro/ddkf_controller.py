# tune3/micro/ddkf_controller.py
"""
DDKF -- Data-Driven Kalman Filter para o log-learning-rate (nivel MICRO).

Estado escalar:  x_t = log(eta_t)  (a variavel de CONTROLE que escolhemos).
Dinamica AR(1):  x_t = x_nom + rho*(x_{t-1} - x_nom) + ruido,  |rho|<1.
Observacao:      z_t in R^{3x1} = (loss, val_loss, log-GSNR).

NATUREZA DO FILTRO (importante):
    Diferente de um Kalman classico que RASTREIA um estado externo desconhecido,
    aqui o estado x_t E' a variavel de controle (log-lr). O DDKF identifica
    empiricamente a relacao lr -> observacoes via Cov(x, z) e ajusta o lr para
    levar as observacoes na direcao desejada. E' um CONTROLADOR adaptativo.

EXCITACAO PERSISTENTE (system identification):
    Para estimar Cov(x, z) e' preciso que x (log-lr) tenha VARIADO. Sem variacao,
    Cov(x,z)=0 e o controlador fica inerte (corretamente: sem informacao sobre o
    efeito do lr). Por isso injetamos exploracao (exploration_std) durante a fase
    de identificacao (warmup). Padrao explore-then-exploit.

CONVENCAO DIMENSIONAL CANONICA (Kalman):
    P_xz in R^{1x3} (linha),  P_zz in R^{3x3},  K = P_xz P_zz^{-1} in R^{1x3},
    e in R^{3x1},  u = K e in R (escalar, SEM transpose). u via solve.

Estabilidade: shrinkage isotropico + floor em P_zz; clip de |u| a multiplos de
sqrt(S_inf), S_inf = sigma_eta^2/(1-rho^2). R^2 = S_u/S_inf (S_u=P_xz P_zz^{-1} P_xz^T).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class DDKFConfig:
    rho: float = 0.95
    sigma_eta: float = 0.05
    obs_dim: int = 3
    window_size: int = 100
    shrinkage: float = 0.10
    floor: float = 1e-6
    max_correction: float = 3.0
    min_samples: int = 30
    exploration_std: float = 0.03   # excitacao do log-lr na fase de identificacao
    exploration_decay: float = 0.0  # 0 => exploracao so no warmup; >0 => decai apos

    def __post_init__(self):
        if not (abs(self.rho) < 1.0):
            raise ValueError(f"|rho| deve ser < 1; rho={self.rho}")


class DDKFController:
    def __init__(self, x0: float, config: Optional[DDKFConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = config or DDKFConfig()
        self.x_hat: float = float(x0)
        self.x_nominal: float = float(x0)
        self.x_buffer: deque = deque(maxlen=self.cfg.window_size)
        self.z_buffer: deque = deque(maxlen=self.cfg.window_size)
        self.S_inf: float = self.cfg.sigma_eta**2 / (1.0 - self.cfg.rho**2)
        self.last_u: float = 0.0
        self.last_S_u: float = 0.0
        self._t: int = 0
        self._rng = rng or np.random.default_rng()

    def predict(self) -> float:
        return self.x_nominal + self.cfg.rho * (self.x_hat - self.x_nominal)

    def _estimate_covariances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(self.x_buffer, dtype=float)
        Z = np.asarray(self.z_buffer, dtype=float)
        n = len(X)
        x_mean = X.mean(); z_mean = Z.mean(axis=0)
        Xc = X - x_mean; Zc = Z - z_mean
        P_xz = (Xc[:, None] * Zc).sum(axis=0) / (n - 1)
        P_zz = (Zc.T @ Zc) / (n - 1)
        d = self.cfg.obs_dim
        target = (np.trace(P_zz) / d) * np.eye(d)
        P_zz = (1.0 - self.cfg.shrinkage) * P_zz + self.cfg.shrinkage * target
        P_zz = P_zz + self.cfg.floor * np.eye(d)
        return P_xz, P_zz, z_mean

    def _exploration_noise(self) -> float:
        """Excitacao do log-lr para identificacao do sistema."""
        if self.cfg.exploration_std <= 0:
            return 0.0
        if len(self.z_buffer) >= self.cfg.min_samples and self.cfg.exploration_decay == 0:
            return 0.0  # exploracao so no warmup
        std = self.cfg.exploration_std
        if self.cfg.exploration_decay > 0:
            std *= np.exp(-self.cfg.exploration_decay * self._t)
        return float(self._rng.normal(0.0, std))

    def update(self, observation) -> float:
        z = np.asarray(observation, dtype=float).reshape(-1)
        if z.shape[0] != self.cfg.obs_dim:
            raise ValueError(f"obs dim {self.cfg.obs_dim}, recebido {z.shape[0]}")
        self._t += 1
        x_pred = self.predict()

        if len(self.z_buffer) >= self.cfg.min_samples:
            P_xz, P_zz, z_mean = self._estimate_covariances()
            e = z - z_mean
            u = float(P_xz @ np.linalg.solve(P_zz, e))
            self.last_S_u = float(P_xz @ np.linalg.solve(P_zz, P_xz))
            u_max = self.cfg.max_correction * np.sqrt(self.S_inf)
            u = float(np.clip(u, -u_max, u_max))
            x_new = x_pred - u
        else:
            u = 0.0
            self.last_S_u = 0.0
            x_new = x_pred

        # excitacao (warmup): permite identificar Cov(x,z)
        x_new = x_new + self._exploration_noise()

        self.last_u = u
        self.x_hat = x_new
        self.x_buffer.append(x_new)
        self.z_buffer.append(z.copy())
        return x_new

    def information_ratio(self) -> float:
        if self.S_inf <= 0.0:
            return 0.0
        return self.last_S_u / self.S_inf

    @property
    def learning_rate(self) -> float:
        return float(np.exp(self.x_hat))
