# tune3/baselines/optuna_optimizer.py
import optuna
import torch
import torch.nn as nn
import structlog
import wandb
from typing import Any
from torch.utils.data import DataLoader as TorchDataLoader

from tune3.models import TabularMLP
from tune3.utils import ModelTrainer, hutchinson_trace_estimator

logger = structlog.get_logger()


class CurvatureAwareOptuna:
    def __init__(self, train_loader: TorchDataLoader, test_loader: TorchDataLoader, input_dim: int, cfg: Any):
        """
        Otimizador Bayesiano com Optuna que utiliza o traço de Hutchinson
        como fator de penalidade na função objetivo e integra com W&B.
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_dim = input_dim
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Constantes extraídas do YAML (Metodologia DDD)
        self.lambda_trace = self.cfg.experiment.ref_point_trace_factor
        self.num_trials = self.cfg.experiment.bo.num_trials

    def objective(self, trial: optuna.Trial) -> float:
        # Inicializa o rastreamento W&B
        run = wandb.init(
            project="tune3-v2",
            group="bayesian-optimization",
            name=f"trial_{trial.number}",
            config=trial.params,
            mode="online"
        )

        # 1. Amostragem Bayesiana dos Hiperparâmetros
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        n_layers = trial.suggest_int("n_layers", 1, 3)

        hidden_dims = []
        for i in range(n_layers):
            hidden_dims.append(trial.suggest_int(f"n_units_l{i}", 16, 128))

        # 2. Inicializar Modelo e Otimizador
        model = TabularMLP(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=2,
            dropout_rate=dropout
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = ModelTrainer(model, criterion, optimizer, self.device)

        # 3. Treinamento Base (Nível 1)
        epochs = 10
        for epoch in range(epochs):
            trainer.train_epoch(self.train_loader)

        # 4. Avaliação Base
        val_metrics = trainer.evaluate(self.test_loader)
        val_loss = val_metrics["val_loss"]

        # 5. Penalidade de Curvatura (Nível 2)
        X_val_batch, y_val_batch = next(iter(self.test_loader))
        X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

        model.train()
        outputs = model(X_val_batch)
        base_val_loss = criterion(outputs, y_val_batch)

        trace_penalty = hutchinson_trace_estimator(model, base_val_loss, num_vectors=5)

        # Score Bi-nível
        final_score = val_loss + (self.lambda_trace * trace_penalty)

        # Log no W&B para auditoria científica
        wandb.log({
            "val_loss": val_loss,
            "trace_hessiana": trace_penalty,
            "final_score": final_score,
            "lr": lr,
            "dropout": dropout
        })

        logger.info(
            f"Trial {trial.number} Concluída",
            val_loss=round(val_loss, 4),
            trace=round(trace_penalty, 4),
            score=round(final_score, 4)
        )

        run.finish() # Finaliza a sessão da trial
        return final_score

    def optimize(self) -> optuna.Study:
        logger.info("Iniciando Loop de Otimização Bayesiana", trials=self.num_trials)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.num_trials)

        logger.info(
            "Otimização Concluída com Sucesso!",
            melhor_trial=study.best_trial.number,
            melhor_score=round(study.best_value, 4),
            melhores_params=study.best_params
        )

        return study