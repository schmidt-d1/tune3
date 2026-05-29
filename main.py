# main.py
import os
import torch
import structlog
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from tune3.data import DataLoader, DataPreprocessor, validate_dataset
from tune3.baselines import CurvatureAwareOptuna

logger = structlog.get_logger()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    logger.info("Iniciando Pipeline Tune3 v2 com Loop Bayesiano", config=OmegaConf.to_container(cfg, resolve=True))

    # 1. Geração de Dados Sintéticos (Dummy)
    data_path = "data/sample.csv"
    target_col = "target"
    if not os.path.exists(data_path):
        os.makedirs("data", exist_ok=True)
        df = pd.DataFrame(np.random.randn(200, 10), columns=[f"feat_{i}" for i in range(10)])
        df[target_col] = np.random.randint(0, 2, 200)
        df.to_csv(data_path, index=False)

    # 2. Pipeline de Dados
    loader = DataLoader(data_path=data_path, target_column=target_col)
    X_train, X_test, y_train, y_test = loader.load_and_split(test_size=0.2)
    validate_dataset(pd.concat([X_train, y_train], axis=1), target_col)

    preprocessor = DataPreprocessor(scaling_method="robust")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled.values), torch.LongTensor(y_train.values))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled.values), torch.LongTensor(y_test.values))

    train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Executar o Otimizador Bayesiano Bi-nível
    input_dim = X_train_scaled.shape[1]

    optimizer = CurvatureAwareOptuna(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        cfg=cfg
    )

    # Executa as 30 trials definidas no tune3_full.yaml
    study = optimizer.optimize()

    return study.best_value


if __name__ == "__main__":
    main()