# tune3/tests/test_models.py
import torch
from tune3.models import TabularMLP, VisionResNet


def test_tabular_mlp_forward_shape():
    """Garante que a MLP processa os dados tabulares e devolve a dimensão correta."""
    batch_size = 16
    input_dim = 10
    output_dim = 2

    model = TabularMLP(input_dim=input_dim, hidden_dims=[32, 16], output_dim=output_dim)

    # Tensor dummy
    x = torch.randn(batch_size, input_dim)
    output = model(x)

    assert output.shape == (batch_size, output_dim), "A dimensão de saída da MLP está incorreta."


def test_vision_resnet_forward_shape():
    """Garante que o wrapper da ResNet se adapta ao número de classes especificado."""
    batch_size = 2
    num_classes = 5

    model = VisionResNet(num_classes=num_classes, resnet_version="resnet18", pretrained=False)

    # Tensor dummy simulando uma imagem RGB (3 canais, 224x224)
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)

    assert output.shape == (batch_size, num_classes), "A dimensão de saída da ResNet está incorreta."