# Tune3 — HPO Bi-nível Consciente de Curvatura para SecOps

Framework de otimização de hiperparâmetros que combina busca Bayesiana
multi-objetivo (CVaR + curvatura Tr(H²)) no nível **MACRO** com controle online de
learning-rate (DDKF) e guardas de segurança (Cantelli, EoS) no nível **MICRO**,
aplicado à detecção de malware Android (DREBIN-215).

> **Estado (set/2026):** código auditado e corrigido (ver `docs/AUDITORIA_2026_09.md`).
> Os experimentos H1–H3 precisam ser **reexecutados** a partir deste commit; o protocolo
> vinculante está em `PREREGISTRATION.md`. Guia dos alunos: `docs/`.

## Instalação (Windows PowerShell / Linux / macOS)

```bash
git clone https://github.com/schmidt-d1/tune3.git
cd tune3
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1      Linux/macOS:  source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,vision]"        # dev = pytest; vision = torchvision (trilha CIFAR)
# GPU NVIDIA: instale o torch com CUDA ANTES do comando acima, ex.:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Requer Python ≥ 3.10 (testado em 3.11 e 3.14). `requirements.txt` é um espelho de
`pyproject.toml` para quem preferir `pip install -r`.

## Verificação em três níveis (faça nesta ordem)

```bash
pytest -q                                   # 1) suíte de testes (~20-40 s; 89 testes)
python scripts/validate_theory.py           # 2) teoria x implementação, dados sintéticos (~2 min)
python scripts/validate_theory.py --csv data/drebin215.csv --device cuda --tag seu-nome   # 3) idem, DREBIN
```

O nível 2/3 imprime um veredito por componente (PROVADO / VERIFICADO(escopo) /
REFUTADO / EM ABERTO) e grava `results/validation/…json`.

## Dados

Baixe o DREBIN-215 e salve em `data/drebin215.csv` (15 036 amostras × 215 features):
- figshare: *Android malware dataset for machine learning 2* (id 5854653)
- Kaggle: "drebin 215 dataset"

`data/*.csv` é ignorado pelo git — cada máquina baixa o seu.

## Experimentos

```bash
# H1 — piloto in-distribution (todos os 8 métodos)
python scripts/run_pilot.py --csv data/drebin215.csv --seeds 0 1 2 --epochs 20 --tag seu-nome
python scripts/run_pilot_parallel.py --csv data/drebin215.csv --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --epochs 100 --device cuda --workers 4 --tag seu-nome

# H2 — desbalanceamento (S1)
python scripts/run_s1_imbalance.py --csv data/drebin215.csv --ratios 0.05 0.02 0.01 --device cuda --tag seu-nome

# H3 — zero-day por cluster (S2); análise primária POR SEED (média sobre folds)
python scripts/run_s2_zeroday.py --csv data/drebin215.csv --n-clusters 5 --device cuda --tag seu-nome

# subconjunto de métodos (ex.: só a Linha 1 / E1)
python scripts/run_s2_zeroday.py ... --methods tune3_bestcvar tune3_placebo_bestcvar bo_mono_cvar

# E0 — reanálise (correlação parcial CVaR x log Tr(H²) dentro dos folds de H3) sobre results/s2_zeroday/
python scripts/analyze_e0.py --tag seu-nome

# consolidar tudo o que está em results/
python scripts/collect_results.py
```

Métodos disponíveis (`--methods`): `tune3`, `tune3_bestcvar` (primário),
`tune3_noddkf_bestcvar` (E2), `tune3_placebo_bestcvar` (E1), `bo_mono_cvar` (B4, E1),
`random_search`, `asha`, `sam`.

## Como os resultados são entregues

Cada script grava `results/<experimento>/<data-hora>_<commit>_<tag>.json` com metadados
(commit, árvore suja?, GPU, versões, argumentos, duração) + resultado bruto. **Esses JSONs
são versionados**: `git add results/ && git commit -m "resultado: s2 seeds 0-9 (seu-nome)" && git push`.
Preencha também `docs/TEMPLATE_RELATORIO_ALUNO.md` (copie para `results/relatorios/`).
Rode sempre a partir de uma árvore limpa (`git status` vazio); resultados com
`meta.git.dirty = true` não são aceitos como oficiais.

## Estrutura

```
tune3/
  core/          objetivo CVaR (quantil e Rockafellar-Uryasev)
  macro/         BO multi-objetivo (BoTorch GP + qLogNEHVI) + seleção knee / best-CVaR
  micro/         DDKF (controle de log-lr) + RegimeGate (gating por R²)
  safety/        CantelliGuard (aborto) + EoSDetector (edge-of-stability)
  curvature/     Hutchinson Tr(H²) + GSNR
  integration/   trial (treino integrado, telemetria) + runner (Tune3 completo, placebo)
  baselines/     plain_trial, Random Search, ASHA, SAM, B4 (bo_mono)
  data/          loader DREBIN + desbalanceamento (S1) + deslocamento por cluster (S2)
  experiments/   estatística (Wilcoxon/Holm/d_z/BCa), protocolos, métricas, results_io
  models/        fábrica de modelos (MLP; ResNet opcional via extra `vision`)
  tests/         89 testes (inclui os testes-sentinela da auditoria de set/2026)
scripts/         validate_theory, run_pilot(_parallel), run_s1_imbalance, run_s2_zeroday,
                 run_tune3_drebin, analyze_e0, collect_results, verify_setup
docs/            auditoria, guia dos alunos, teoria
legacy/          código da fase Optuna/Breast-Cancer (não usado)
```

## Testes

```bash
pytest -q                     # tudo (89)
pytest -q -m "not slow"       # pula os 4 testes lentos de BO
```
