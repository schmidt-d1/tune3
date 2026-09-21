# Auditoria do repositório — 16/09/2026

Auditoria linha a linha de `HEAD 517cd9b` (16 commits, 29/05–02/06/2026) contra a
documentação do projeto (Handoff, Estado Consolidado, manual de implementação), com
verificação computacional independente onde aplicável. Taxonomia: REFUTADO / PROVADO /
VERIFICADO(escopo) / PLAUSÍVEL / EM ABERTO.

## 1. Achados bloqueantes (corrigidos neste commit)

| # | Achado | Evidência | Correção |
|---|---|---|---|
| B1 | **Vazamento do teste na avaliação final.** `protocol.py::_eval_on_test` e `security_protocol.py::_eval_full` chamavam `plain_trial(hp, (X_tr, y_tr, X_te, y_te))`: early stopping e melhor época escolhidos **no teste**; `X_val` não era usado. Todos os números de H1/H2/H3 foram produzidos assim. | leitura do código; `test_A1_*` reproduz | `plain_trial(..., test_data=)`: escolhe época na validação, restaura o estado da melhor época e avalia o teste **uma vez**. Protocolos reescritos. **H1–H3 devem ser reexecutados.** |
| B2 | **EoSDetector nunca disparava.** `trial.py` chamava `eos.update(last_curv)` toda época, mas a curvatura só mudava a cada 5; a regra "4 subidas estritamente consecutivas em janela 6" é insatisfazível com valores repetidos. | simulação com o código real: curvatura ×524 288 em 100 épocas → 0 disparos; por época → 16 (`test_A2_*`) | `eos.update` só quando a curvatura é re-estimada. |
| B3 | **Hutchinson com dropout ativo** (`model.train()` antes de `estimate`) no trial e no `plain_trial`: mede a curvatura de uma sub-rede aleatória. Patch documentado (`model.eval()`) não estava aplicado. | MLP pequeno, dropout 0,3: razão train/exato = 1,28–1,32 (`validate_theory V1b`) | `model.eval()` durante a estimativa (`test_A3`). |
| B4 | **`torchvision` usada mas não declarada** (`tune3/models/resnet.py` importa no topo; `tune3.models` importa `resnet`). `pip install -e .` seguido de `pytest` falha na coleta; a CI (`requirements.txt`) também. | `ModuleNotFoundError` reproduzido em Windows/3.14 e Linux/3.11 | import lazy dentro de `VisionResNet`; extra `vision`; `pytest.importorskip` no teste; extra `dev` com pytest; CI corrigida. |
| B5 | **`PREREGISTRATION.md` ausente**, apesar de o protocolo declarar pré-registro vinculante. | `Get-Content` / listagem | rascunho criado para assinatura (`PREREGISTRATION.md`). |
| B6 | **B4 (BO mono-objetivo em CVaR) e o placebo bi-objetivo não existiam** — o E1 inteiro estava por implementar; sem eles não se separa H_G de H_M. | `hpo.py` só tem RS/ASHA; `protocol.py` roda 5 métodos | `tune3/baselines/bo_mono.py` (GP + qLogEI, mesmo orçamento/trial do Tune3); `Tune3RunConfig.objective2="random"`; métodos `bo_mono_cvar`, `tune3_placebo_bestcvar`, `tune3_noddkf_bestcvar` (E2) nos protocolos. |

## 2. Achados maiores (corrigidos)

| # | Achado | Correção |
|---|---|---|
| M1 | `stats.py`: bootstrap **percentílico**; protocolo exige **BCa**; regra `|d_z| ≥ 0,30` não codificada. | `bootstrap_ci(..., ci_method="bca")` (scipy), `dz_ci95`, campos `practical`/`prereg_win`. |
| M2 | `run_s2_zeroday.py` agregava 5 folds × N seeds como 5N pares independentes (pseudo-replicação). | análise primária **por seed** (`aggregate_per_seed`); pooled reportada como exploratória. |
| M3 | Sem registro padronizado de resultados (JSONs soltos, ignorados pelo git, sem commit/ambiente). | `results_io.save_result` + `results/**/*.json` versionado + `collect_results.py`. |
| M4 | `main.py`, `configs/`, `optuna_optimizer.py`, `trainer.py`, `loader.py`, `preprocessing.py`, `validation.py`, `Dockerfile` — legado quebrado (imports inexistentes), confunde novos usuários. | movidos para `legacy/` com README explicativo. |
| M5 | README dizia 68 testes (real: 74 antes, 88 agora); instruções de instalação incompletas. | README reescrito. |
| M6 | `test_trial_aborta_lr_explosivo` passava por acaso do caminho aleatório (lr = 50 colapsa em ReLUs mortas, não diverge). | lr = 1e3 (overflow determinístico). |

## 3. Divergências documentação × código que NÃO são bugs (decisões a registrar)

| # | Documentado | Implementado | Decisão sugerida |
|---|---|---|---|
| D1 | Vetor de observação do DDKF `z = [L, log10(1+Tr(H²)), log GSNR]` (Handoff §3.2, manuscrito) | `z = [train_loss, val_loss, log GSNR]` — **o DDKF não observa curvatura**; ela entra só no MACRO e no EoS | manter o código (resultados anteriores usaram isso) e **corrigir o manuscrito**; testar a variante com curvatura como ablação futura. |
| D2 | Log-AR(1) acoplado a um scheduler base (Cosine) | `x_nominal = log lr0` constante — AR(1) em torno de lr fixo; não há scheduler | idem: documentar; a "meta-scheduler sobre Cosine" não existe no código. |
| D3 | cEHVI com restrição operacional L*; falha → (∞, ∞) | `qLogNEHVI` **sem restrição**; aborto → penalidade 1e3; GP homoscedástico | documentar como "EHVI sem restrição"; Fix 1 (ruído heteroscedástico) segue pendente. |
| D4 | GSNR por duplo EMA O(1) | estimador direto de K = 4 minibatches (Liu et al. 2020) | manter; corrigir manuscrito (não há termo `1/B` a remover — C3 não se aplica). |
| D5 | `maybe_estimate` com cooldown + `emergency_override` (C8) | `maybe_estimate` **não é usado**; curvatura em cronograma fixo (`curvature_every`) | remover código morto ou documentar; C8 como descrito não existe, mas o deadlock também não. |
| D6 | DDKF como controlador ativo | `exploration_decay = 0` ⇒ excitação só no warm-up ⇒ R² decai a ~0 e o filtro fica em regime A (predict-only) — o próprio `regime_gating.py` admite "rede de segurança, não otimizador". `validate_theory V6b`/`V9`: 0/85 épocas ativas. | é **a pergunta do E2**; telemetria agora reportada. |

## 4. Verificações independentes (sobreviventes)

- **Hutchinson: PROVADO.** HVP do repositório vs `H·v` com Hessiana exata (`torch.autograd.functional.hessian`, 194 parâmetros): diferença máx. 1,5·10⁻⁷ em 20 sondas; média de 20 000 sondas vs ‖H‖²_F exato: z = −0,77. Hoisting do gradiente (C7) correto.
- **CVaR: PROVADO.** Quantil ≡ Rockafellar-Uryasev (< 0,02 %); valor teórico lognormal a 2,3 %.
- **GSNR: PROVADO** contra a definição; **VERIFICADO** GSNR ∝ B (razão 14,5 para B×8).
- **Cantelli: VERIFICADO(t₃, janela 30):** alarme falso 0,9 % ≤ 5 %.
- **Estatística: VERIFICADO(300 sim.):** FWER 2 % sob H₀; poder 92 % em d_z = 0,8, N = 20.
- **Suíte:** 74/74 antes das correções (Windows 3.14 e Linux 3.11); **88/88** depois; **89/89** após a revisão 2 (teste A10: `curvature_every` atravessa até o `TrialConfig`).

## 5. Itens em aberto para o coordenador

1. Assinar e congelar `PREREGISTRATION.md` (data + hash) **antes** de qualquer execução oficial.
2. Decidir D1–D6 (recomendação: manter código, corrigir manuscrito; E2 decide D6).
3. `curvature_every`: 5 (padrão) ou 1? Com 5, o EoS precisa de 30 épocas de histórico e a
   paciência de early stopping é 10 — na prática dificilmente dispara. Decisão de protocolo.
4. Revogar o token do GitHub exposto (Handoff §11), se ainda não feito.
5. Os JSONs antigos (Drive) têm o vazamento B1: **E0 "de custo zero" não é válido sobre eles.**
