# Tune3 — Auditoria, Correções e Roadmap de Implementação

> Documento de orientação (professor sênior). Lê-se de cima para baixo.
> Os arquivos `.py` corrigidos/novos acompanham este documento.

---

## 1. Veredito da auditoria

**Engenharia (boa):** Hydra, W&B, estrutura de pacote, testes, Docker, pre-commit — acima da média.

**Ciência (incompleta + com bugs):** o núcleo do Tune3 ainda não existe; o único
componente científico (`utils/curvature.py`) calcula a quantidade errada.

### Bugs encontrados

| # | Severidade | Onde | Problema | Correção |
|---|---|---|---|---|
| 1 | 🔴 Crítico | `utils/curvature.py` | Calcula `Tr(H)` (via `vᵀHv`), não `Tr(H²)` | Usar `‖Hv‖²` → `tune3/curvature/hutchinson.py` (incluído) |
| 2 | 🔴 Crítico | `baselines/optuna_optimizer.py` | Curvatura medida no **test set** (data leak) | Medir em treino/val dedicado |
| 3 | 🟠 Alto | `requirements.txt` | Vazio → não reprodutível | Substituído (incluído) |
| 4 | 🟠 Alto | `configs/` | `model: rf` mas código usa MLP; `ref_point_*` mal usado | Realinhar configs (Fase 3) |
| 5 | 🟡 Médio | `tests/test_curvature.py` | Testa a quantidade errada (espera 6=Tr H) | Substituído: espera 12=Tr(H²) (incluído) |

### Componentes do manual ausentes

| Componente | Arquivo | Status |
|---|---|---|
| Estimador Tr(H²) | `curvature/hutchinson.py` | ✅ **incluído (corrigido)** |
| Filtro DDKF (MICRO) | `micro/ddkf_controller.py` | ❌ Fase 2 |
| Guarda de Cantelli | `safety/cantelli.py` | ❌ Fase 2 |
| Gating de regime (R²) | `micro/regime_gating.py` | ❌ Fase 2 |
| Detector Edge-of-Stability | `safety/eos_detector.py` | ❌ Fase 2 |
| Objetivo CVaR | `core/objectives.py` | ❌ Fase 3 |
| MACRO: GP + cEHVI | `macro/bo_loop.py` | ❌ Fase 3 |

> O `optuna_optimizer.py` **não é desperdício**: vira o **baseline BOHB/TPE** (B-class).
> Apenas renomeie e remova a penalidade de curvatura dele (baseline não usa curvatura).

---

## 2. Recomendação MACRO: **BoTorch** (não Optuna, não Ax)

**Decisão: usar BoTorch diretamente para o nível MACRO.**

**Justificativa:**
1. **Fidelidade ao manual.** O manual especifica *GP + cEHVI* (constrained Expected
   Hypervolume Improvement). BoTorch implementa isso nativamente
   (`qNoisyExpectedHypervolumeImprovement` / `qLogNEHVI` com `constraints`).
   As referências do próprio artigo (Daulton 2020/2021) **são** os papers do BoTorch.
2. **Optuna multi-objetivo NÃO é cEHVI.** O multi-objetivo do Optuna usa NSGA-II/MOTPE
   (evolucionário), não hipervolume esperado sobre um GP. Usá-lo e chamar de "cEHVI"
   seria deturpação metodológica — um revisor top-tier rejeitaria.
3. **Controle do GP.** O acoplamento com o objetivo CVaR e a restrição de aborto exige
   acesso ao posterior do GP; BoTorch dá esse controle, Ax esconde atrás de uma API de serviço.

**Trade-off honesto:** BoTorch tem curva de aprendizado maior que Optuna. Mitigação:
fornecerei um scaffold mínimo na Fase 3. **Fallback:** se BoTorch travar o progresso,
Ax (`ax-platform`, que usa BoTorch por baixo) é aceitável e bem mais simples — mas
prefira BoTorch puro.

**Optuna permanece no projeto como baseline** (correto pelo protocolo experimental).

---

## 3. Ordem de implementação recomendada (7 fases)

A ordem segue o princípio: **validar a fundação antes de construir sobre ela.**

```
FASE 0  Correções da fundação ........... [arquivos incluídos] <- FAÇA AGORA
FASE 1  Pipeline DREBIN-215 ............. [arquivo incluído]
FASE 2  Núcleo MICRO (a ciência):
          2a. hutchinson (✓ feito na Fase 0)
          2b. cantelli.py (CantelliGuard)
          2c. ddkf_controller.py (Kalman)
          2d. regime_gating.py (R²)
          2e. eos_detector.py
FASE 3  Objetivo CVaR + MACRO (BoTorch GP+cEHVI)
FASE 4  Integração: o loop Tune3 completo
FASE 5  Baselines B1–B9 (Optuna vira um deles)
FASE 6  Protocolo estatístico (Wilcoxon + Holm + dz + bootstrap)
```

Cada fase termina com **testes-sentinela** que devem passar antes de avançar.

---

## 4. FASE 0 — Aplicar agora (arquivos incluídos)

1. **Substitua** `requirements.txt` pelo incluído e rode:
   ```bash
   pip install -r requirements.txt
   ```
2. **Adicione** o pacote `tune3/curvature/` (com `__init__.py` e `hutchinson.py`).
3. **Substitua** `tune3/tests/test_curvature.py` pelo incluído.
4. **Deprecie** `utils/curvature.py`: deixe um shim que importa do novo módulo e emite
   `DeprecationWarning`, ou apague e ajuste os imports. (NÃO mantenha a versão com bug.)
5. **Corrija o data leak** no optimizer: a curvatura deve ser medida sobre um batch de
   **treino** (ou um split de validação dedicado), nunca do `test_loader`.
6. Rode os testes:
   ```bash
   pytest tune3/tests/test_curvature.py -v
   ```
   Os 4 testes devem passar. `test_trace_h2_diagonal` falhando com ~6.0 significa que o
   módulo antigo ainda está sendo importado.

**Commit sugerido:** `fix(curvature): estimar Tr(H^2) via ||Hv||^2 (corrige Tr(H)); +deps`

---

## 5. FASE 1 — Pipeline DREBIN-215 (arquivo incluído)

### 5.1 Esclarecimento importante sobre o dataset
O artigo cita ~129.013 amostras (DREBIN **original**, features esparsas, requer pedido
de acesso). Para reprodutibilidade, usamos **DREBIN-215**: 15.036 amostras × 215 features
binárias. **Declare isto na seção de dados** ("usamos a representação DREBIN-215 de 215
features extraídas por análise estática"). É a versão padrão na literatura recente.

### 5.2 Download (escolha uma)
- **figshare** (fonte do DroidFusion): `figshare.com/articles/dataset/Android_malware_dataset_for_machine_learning_2/5854653`
- **Kaggle**: procure "Drebin-215 dataset"
- Salve como `data/drebin215.csv`.

### 5.3 Uso
```python
from tune3.data.drebin import DrebinLoader, DrebinConfig
loader = DrebinLoader(DrebinConfig(csv_path="data/drebin215.csv"))
X_tr, X_val, X_te, y_tr, y_val, y_te = loader.load_splits()
```
O loader trata: `?` faltantes, classe `S`/`B` ou `1`/`0`, dtype object, split
**estratificado** (preserva proporção malware/benigno: ~37%/63%).

### 5.4 Pré-processamento (DREBIN é binário!)
- **NÃO** use `RobustScaler`/`StandardScaler` nas features binárias 0/1 — elas já estão
  na escala certa e escalonar destrói a esparsidade. Use as features como estão (float32).
  Seu `DataPreprocessor` atual é adequado para o Breast Cancer, mas para DREBIN passe-as direto.
- Imbalanceamento: use `class_weight` no `CrossEntropyLoss` (`weight=[1, n_benign/n_malware]`)
  ou `WeightedRandomSampler`. Isto é coerente com o foco em CVaR (cauda = falsos negativos).

---

## 6. Visão geral das Fases 2–6 (detalhamento virá a seguir)

### FASE 2 — Núcleo MICRO
- **`cantelli.py`**: `CantelliGuard` com `C_γ = sqrt(γ/(1-γ))`, MAD × 1.4826,
  janela deslizante, aborto quando `loss > μ̂ + C_γ·σ̂_MAD`. Testes-sentinela:
  `test_cantelli_fisher_const` (1.4826) e `C_γ(0.95)≈4.36`.
- **`ddkf_controller.py`**: filtro de Kalman escalar (estado = log-lr), observação 3-D
  `(loss, val_loss, log-GSNR)`, ganho `K = P_xz P_zz⁻¹ ∈ R^{1×3}`, correção `u = K·e` (escalar).
  Atenção à convenção dimensional (P_xz como linha 1×3).
- **`regime_gating.py`**: razão `R² = Ŝ_u/Ŝ_∞`; gating predict-only quando `R² < 0.10`.
- **`eos_detector.py`**: detecta crescimento monotônico de Tr(H²) (edge of stability),
  reduz lr por fator 0.7.

### FASE 3 — CVaR + MACRO
- **`objectives.py`**: CVaR empírico ao nível γ sobre as perdas por época/fold.
- **`bo_loop.py`** (BoTorch): GP multi-saída sobre `(CVaR, Tr(H²))`, aquisição `qLogNEHVI`
  com restrição de aborto. Ponto de referência adaptativo.

### FASE 4 — Integração
- Loop Tune3: BO (MACRO) sugere ξ → treina com DDKF+Cantelli+gating (MICRO) →
  retorna `(CVaR, curvatura)` ao GP. Checkpoint/resume.

### FASE 5 — Baselines (B1–B9)
- Random Search, ASHA, BOHB(=seu Optuna), GP-EI, PBT, AdaHessian, SAM, SWA, Adam puro.

### FASE 6 — Protocolo estatístico
- Wilcoxon pareado (signed-rank) por seed; correção Bonferroni-Holm na família de
  comparações primárias; tamanho de efeito `d_z` de Cohen; IC 95% por bootstrap;
  N=20 seeds nas 3 comparações primárias, N=10 nas demais. Pré-registrar hipóteses.

---

## 7. Próximo passo concreto

1. Aplique a Fase 0 (arquivos incluídos) e rode `pytest`.
2. Baixe o DREBIN-215 e rode um `load_splits()` para confirmar shapes.
3. Me avise quando passar — entrego a **Fase 2** completa (`cantelli.py` + `ddkf_controller.py`
   + testes), que é onde o Tune3 começa a existir de verdade.
