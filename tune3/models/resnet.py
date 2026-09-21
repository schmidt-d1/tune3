# tune3/models/resnet.py
"""Wrapper de ResNet (trilha Plano B: CIFAR-10). `torchvision` e' dependencia
OPCIONAL (extra `vision`): o import e' feito dentro do construtor para que
`import tune3.models` nao quebre em ambientes sem torchvision."""
import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger()


class VisionResNet(nn.Module):
    def __init__(self, num_classes: int, resnet_version: str = "resnet18", pretrained: bool = False):
        """Wrapper para ResNet, facilitando a troca de backbones durante os experimentos."""
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as e:  # pragma: no cover
            raise ImportError("VisionResNet requer torchvision: pip install -e '.[vision]'") from e

        logger.info("Inicializando modelo de Visão", arquitetura=resnet_version, pre_treinado=pretrained)
        if resnet_version == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif resnet_version == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Versão de ResNet '{resnet_version}' não configurada no wrapper.")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
