# tune3/utils/trainer.py
import torch
import torch.nn as nn
import structlog
from torch.utils.data import DataLoader
from typing import Dict

logger = structlog.get_logger()


class ModelTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str):
        """
        Gerencia o loop de treinamento e avaliação dos modelos.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

        logger.info("ModelTrainer inicializado", device=str(self.device))

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Executa uma única época de treinamento.
        """
        self.model.train()
        total_loss = 0.0
        num_samples = len(dataloader.dataset)

        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            # Zerar gradientes
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # Backward pass e otimização
            loss.backward()
            self.optimizer.step()

            # Acumula a loss (ponderada pelo tamanho do batch)
            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / num_samples
        logger.debug("Época de treinamento finalizada", avg_train_loss=avg_loss)
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Avalia o modelo sem calcular gradientes, economizando memória.
        Retorna dicionário com métricas de validação.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_samples = len(dataloader.dataset)

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)

                # Assumindo tarefa de classificação (verifica se target é inteiro)
                if y_batch.dtype in [torch.int64, torch.long]:
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

        avg_loss = total_loss / num_samples
        metrics = {"val_loss": avg_loss}

        if total > 0:
            metrics["val_accuracy"] = correct / total

        logger.info("Avaliação concluída", **metrics)
        return metrics