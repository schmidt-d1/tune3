# Tune3 — HPO Bi-nível Consciente de Curvatura para SecOps

Framework de otimização de hiperparâmetros que combina busca Bayesiana
multi-objetivo (CVaR + curvatura) no nível MACRO com controle online de
learning-rate (DDKF) e guardas de segurança no nível MICRO.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Estrutura

```
tune3/
  core/          objetivo CVaR (Rockafellar-Uryasev)
  macro/         BO multi-objetivo (BoTorch GP + cEHVI) + seleção de Pareto
  micro/         DDKF (controle de lr) + RegimeGate
  safety/        CantelliGuard (aborto) + EoSDetector (edge-of-stability)
  curvature/     Hutchinson Tr(H²) + GSNR
  integration/   trial (treino integrado) + runner (Tune3 completo)
  baselines/     SAM, Random Search, ASHA, plain_trial
  data/          loader DREBIN + desbalanceamento + deslocamento por cluster
  experiments/   estatística, protocolo, protocolo de segurança, métricas
  models/         fábrica de modelos (agnóstica de arquitetura)
scripts/         pontos de entrada (pilotos, S1, S2)
```

## Uso

```bash
# verificar setup
python scripts/verify_setup.py

# Tune3 ponta a ponta no DREBIN
python scripts/run_tune3_drebin.py --csv data/drebin215.csv --epochs 100 --device cuda

# piloto estatístico (Tune3 vs baselines, N seeds)
python scripts/run_pilot.py --csv data/drebin215.csv --seeds 0 1 2 3 4 --epochs 100 --device cuda

# piloto paralelo (mais rápido na A100)
python scripts/run_pilot_parallel.py --csv data/drebin215.csv --seeds 0 1 2 3 4 5 6 7 8 9 --workers 4 --device cuda

# S1: desbalanceamento (H2)
python scripts/run_s1_imbalance.py --csv data/drebin215.csv --ratios 0.05 0.02 0.01 --device cuda

# S2: zero-day por cluster (H3)
python scripts/run_s2_zeroday.py --csv data/drebin215.csv --n-clusters 5 --device cuda
```

## Testes

```bash
pytest tune3/tests/ -v          # 68 testes
pytest tune3/tests/ -m "not slow"   # pula os testes lentos de BO
```

## Dados

Baixe o DREBIN-215 e salve em `data/drebin215.csv`:
- figshare: Android_malware_dataset_for_machine_learning_2 (id 5854653)
- Kaggle: drebin 215 dataset
