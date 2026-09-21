# PRÉ-REGISTRO — Tune3, Linha 1 (E0 / E1 / E2) e reexecução de H1–H3

**Status:** RASCUNHO para assinatura do coordenador. Torna-se vinculante quando este
arquivo for commitado com a data e o hash abaixo preenchidos e nenhuma execução
"oficial" tiver começado antes desse commit.

| Campo | Valor |
|---|---|
| Coordenador | Prof. Dionatan R. Schmidt (UNIPAMPA / AI Horizon Labs) |
| Data de congelamento | `____/____/2026` |
| Commit vinculante | `________` (`git rev-parse --short HEAD` após este arquivo ser commitado) |
| Repositório | github.com/schmidt-d1/tune3 |
| Substitui | nenhum pré-registro anterior versionado (não existia `PREREGISTRATION.md` no repositório) |

> Regra do projeto: **ajustes de protocolo post-hoc são proibidos.** Qualquer mudança
> depois do congelamento entra na Seção 9 (registro de desvios), com data e motivo, e o
> resultado passa a ser reportado como *exploratório*.

---

## 1. Por que reexecutar H1–H3

A auditoria de set/2026 (`docs/AUDITORIA_2026_09.md`) encontrou, na avaliação final de
todos os métodos, **early stopping e escolha de época feitos sobre o conjunto de teste**
(`protocol.py` / `security_protocol.py` passavam o teste no lugar da validação). Os
números de H1/H2/H3 produzidos por aquela versão **não podem ser reportados**. Também
foi encontrado que o EoSDetector nunca disparava (valor de curvatura repetido) e que o
Hutchinson rodava com dropout ativo. As correções estão no commit vinculante. Tudo
abaixo se refere a execuções **a partir dele**.

## 2. Hipóteses (declaradas antes de rodar)

- **H1 (in-distribution):** Tune3 (seleção best-CVaR) **não** apresenta vantagem
  estatisticamente significativa em CVaR de teste sobre RS/ASHA/SAM. (Resultado anterior;
  esperamos replicar a ausência de efeito.)
- **H2 (desbalanceamento 5%/2%/1%):** a diferença de CVaR a favor do Tune3 cresce com a
  raridade do malware. Predição falsificável: ordenação monotônica de `d_z` nas três razões.
- **H3 (zero-day, leave-one-cluster-out):** Tune3 best-CVaR vence RS, ASHA e SAM em CVaR de
  teste (`prereg_win` nos três).
- **H_G vs H_M (Linha 1):**
  - **E0** — correlação parcial CVaR × log Tr(H²) *dentro dos folds de H3*, controlando por
    log η e weight decay (protocolo V2). Predição de H_G: correlação parcial **positiva**
    e com IC BCa excluindo zero. *Nota:* os JSONs antigos têm o vazamento da Seção 1; E0
    só é válido sobre os JSONs gerados a partir do commit vinculante.
  - **E1** — dois controles com o **mesmo orçamento** do Tune3 (n_init + n_iter avaliações):
    **B4 = `bo_mono_cvar`** (GP + logEI só em CVaR) e **placebo = `tune3_placebo_bestcvar`**
    (2º objetivo = ruído N(0,1)). Predição de H_G: Tune3 best-CVaR vence **ambos**.
    Predição de H_M: Tune3 empata com o placebo e/ou perde para B4.
  - **E2** — `tune3_noddkf_bestcvar` (DDKF desligado) vs `tune3_bestcvar`, mais a telemetria
    de engajamento (`n_ddkf_active_epochs`, `regime_fractions`, `lr_change_fraction`).
    Predição a registrar aqui pelo coordenador: ☐ o DDKF melhora CVaR ☐ é neutro.
    Se `n_ddkf_active_epochs / n_epochs_run < 0,05` na mediana dos trials, o DDKF será
    declarado **inerte** no relatório, independentemente do CVaR.

## 3. Dados

DREBIN-215 (`data/drebin215.csv`, 215 features binárias). Splits estratificados
80/20 teste e 80/20 val dentro do restante (`DrebinLoader`, `random_state = seed`).
S1: `make_imbalanced` só no treino. S2: `cluster_holdout_split` com K = 5 clusters
(k-means, `random_state = seed`), malware do cluster retido = teste.
**O conjunto de teste não é usado em nenhuma decisão** (early stopping, escolha de
época, seleção de hiperparâmetros, curvatura). Verificado por `test_A1_*`.

## 4. Métodos e orçamento

| Método | Papel | Orçamento |
|---|---|---|
| `tune3_bestcvar` | **método primário** | 8 Sobol + 20 iterações qLogNEHVI = 28 trials |
| `tune3` (knee) | seleção alternativa (anomalia do knee) | idem (mesma busca) |
| `bo_mono_cvar` (B4) | E1 | 8 + 20 = 28 trials, mesmo GP, mesmo trial interno |
| `tune3_placebo_bestcvar` | E1 | 28 trials |
| `tune3_noddkf_bestcvar` | E2 | 28 trials |
| `random_search` | baseline | 28 trials |
| `asha` | baseline | 32 configs, r0 = 4, η = 2 (orçamento em épocas reportado) |
| `sam` | baseline de planura em θ | RS de 28 trials com otimizador SAM (ρ = 0,05) |

Trial: MLP, `max_epochs = 100`, `patience = 10`, `batch_size = 4096`, SGD momentum 0,9,
`class_weight` ativo, γ = 0,95. Espaço de busca: `log_lr ∈ [−5, −2]`, `log_wd ∈ [−6, −2]`,
`dropout ∈ [0, 0,5]`, `hidden_dim ∈ [32, 256]`, `n_layers ∈ [1, 4]`.
Parâmetros de controle fixos: DDKF (ρ = 0,95, σ_η = 0,05, `min_samples = 30`,
`exploration_std = 0,03`), gating (`r2_low = 0,10`), Cantelli (γ = 0,95, janela 10),
EoS (janela 6, paciência 4, fator 0,7), Hutchinson (5 sondas), `curvature_every = 5`.

## 5. Métrica primária e secundárias

Primária: **CVaR₀,₉₅ das perdas por amostra no teste**, do modelo da melhor época de
validação. Secundárias (reportadas, sem decisão): AUC-PR, FPR@95%TPR, TPR@1%FPR, AUC-ROC,
F1 e MCC (limiar 0,5 — degenerados sob forte desbalanceamento; priorizar AUC-PR e FPR@95TPR).

## 6. Protocolo estatístico (vinculante) — implementado em `tune3/experiments/stats.py`

1. Comparações **pareadas por seed** (mesmos splits para todos os métodos).
2. **Wilcoxon signed-rank** bilateral, α = 0,05.
3. **Correção de Holm** sobre a família {primário vs cada comparador} dentro de cada experimento.
4. Tamanho de efeito **d_z de Cohen** (pareado) com **IC 95% bootstrap BCa** (10 000 reamostras).
5. **Regra de decisão:** vitória pré-registrada ⇔ `p_Holm < 0,05` **e** `|d_z| ≥ 0,30` **e**
   direção favorável (`prereg_win` no JSON).
6. **Unidade estatística em S2:** a seed (média sobre os K folds). A análise "pooled"
   (K×N pares) é exploratória (pseudo-replicação declarada).
7. **Seeds:** H1/H2: 20 (`0..19`); H3: 10 (`0..9`) × 5 folds. Caso inconclusivo
   (p_Holm ∈ [0,05; 0,20] ou IC de d_z contendo 0 com |d_z| ≥ 0,30): ampliar para 40 seeds,
   **uma única vez**, declarado aqui a priori.
8. Equivalência prática: |d_z| < 0,30 com IC BCa dentro de (−0,50; 0,50) ⇒ "sem diferença prática".

## 7. Árvore de decisão (Linha 1) — declarada antes de rodar

| Cenário | Condição observada | Consequência |
|---|---|---|
| **C1** | Tune3 vence B4 **e** placebo (prereg_win em ambos) e E0 tem correlação parcial > 0 | H_G sobrevive → seguir para H4 (robustez adversarial) |
| **C2** | Placebo empata com Tune3 (equivalência prática) | ganho vem da exploração multi-objetivo → reposicionar tese; priorizar Linha 3 |
| **C3** | Empate geral | artigo de metodologia de avaliação |
| **C4** | B4 vence Tune3 (`prereg_win` de B4) | **parada obrigatória** e reunião de coordenação |

Nenhum cenário é fracasso; cada um leva a um artigo diferente.

## 8. Registro e entrega dos resultados

Todo script grava `results/<experimento>/<AAAAMMDD-HHMMSS>_<commit>_<tag>.json` com bloco
`meta` (commit, árvore suja?, ambiente, GPU, argumentos, duração) e `payload`. Os alunos
entregam com `git add results/ && git commit && git push`. Resultados de árvore **suja**
(`meta.git.dirty = true`) não contam como oficiais. `scripts/collect_results.py` consolida.

## 9. Registro de desvios (preencher só depois do congelamento)

| Data | O que mudou | Motivo | Impacto na interpretação |
|---|---|---|---|
| | | | |
