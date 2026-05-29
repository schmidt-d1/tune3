# tune3/models/resnet.py
import torch
import torch.nn as nn
import torchvision.models as models
import structlog

logger = structlog.get_logger()


class VisionResNet(nn.Module):
    def __init__(self, num_classes: int, resnet_version: str = "resnet18", pretrained: bool = False):
        """
        Wrapper para ResNet, facilitando a troca de backbones durante os experimentos.
        """
        super().__init__()

        logger.info("Inicializando modelo de Visão", arquitetura=resnet_version, pre_treinado=pretrained)

        if resnet_version == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif resnet_version == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Versão de ResNet '{resnet_version}' não configurada no wrapper.")

        # Ajuste da camada linear final (fully connected) para o número de classes específico do dataset
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)