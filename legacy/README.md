# legacy/ — código anterior ao Tune3 bi-nível

Estes arquivos pertencem à primeira encarnação do repositório (pipeline Optuna +
Breast Cancer + Hydra) e **não fazem parte do Tune3 atual**. Estão aqui apenas
para histórico; nenhum script ou teste do projeto os importa.

| Arquivo | O que era | Por que saiu |
|---|---|---|
| `main.py` | entrypoint Hydra para `CurvatureAwareOptuna` no Breast Cancer | imports quebrados (`tune3.data.DataLoader`, `tune3.baselines.CurvatureAwareOptuna` não existem mais); confunde quem chega ao repo |
| `configs/` | YAMLs Hydra (`model: rf`, `ref_point_*`) | não usados por nenhum script atual |
| `optuna_optimizer.py` | BO com Optuna + penalidade de Tr(H) | substituído pelo nível MACRO (BoTorch) e pelos baselines em `tune3/baselines/` |
| `trainer.py`, `loader.py`, `preprocessing.py`, `validation.py` | utilitários do pipeline antigo | só eram usados pelos itens acima |
| `Dockerfile`, `docker-compose.yml` | ambiente CUDA 11.8 / torch 2.1.2 | inconsistente com `torch>=2.2`; ninguém do projeto usa Docker hoje |
| `tune3_COLAB_completo.ipynb` | notebook Colab (68 testes, zip manual) | fluxo substituído pelo guia dos alunos (`docs/`) |
| `AUDITORIA_E_ROADMAP.md` | auditoria da Fase 0 (mai/2026) | histórico; a auditoria vigente é a de set/2026 (`docs/AUDITORIA_2026_09.md`) |

Para usar o Tune3 atual, veja o `README.md` da raiz.
