# Tune3 — passo a passo para fechar a Parte 1 (git pronto para os alunos)

Os 33 arquivos corrigidos/novos **já estão gravados na sua pasta**
`C:\Users\diona\PycharmProjects\Tune3\tune3` (nada foi commitado — o git é seu).
Falta o que só o git faz: mover o legado, apagar o `pytest.ini`, escrever o `ci.yml`
(arquivo protegido, não pude gravar), rodar os testes e commitar.

## Passo 1 — Conferir o que mudou (PyCharm, Alt+F12, com `(.venv)` no prompt)

```powershell
cd C:\Users\diona\PycharmProjects\Tune3\tune3
git status
git diff --stat
```

Esperado: ~23 arquivos modificados (`M`) e ~10 novos (`??`): `tune3/baselines/bo_mono.py`,
`tune3/experiments/results_io.py`, `tune3/tests/test_auditoria_2026_09.py`,
`scripts/validate_theory.py`, `scripts/collect_results.py`, `PREREGISTRATION.md`,
`docs/AUDITORIA_2026_09.md`, `legacy/README.md`, `results/README.md`.

## Passo 2 — Prompt para o Claude Code (cole inteiro)

```
Estamos no repositório Tune3 (C:\Users\diona\PycharmProjects\Tune3\tune3). Uma auditoria
externa já gravou 33 arquivos corrigidos na árvore de trabalho (veja `git status`).
NÃO reescreva nem "melhore" esses arquivos. Faça exatamente o seguinte, um passo por vez,
mostrando a saída de cada comando:

1. Mover o código legado para legacy/ preservando histórico:
   git mv main.py legacy/main.py
   git mv Dockerfile legacy/Dockerfile
   git mv docker-compose.yml legacy/docker-compose.yml
   git mv tune3_COLAB_completo.ipynb legacy/tune3_COLAB_completo.ipynb
   git mv AUDITORIA_E_ROADMAP.md legacy/AUDITORIA_E_ROADMAP.md
   git mv configs legacy/configs
   git mv tune3/baselines/optuna_optimizer.py legacy/optuna_optimizer.py
   git mv tune3/utils/trainer.py legacy/trainer.py
   mkdir legacy/tune3_data
   git mv tune3/data/loader.py legacy/tune3_data/loader.py
   git mv tune3/data/preprocessing.py legacy/tune3_data/preprocessing.py
   git mv tune3/data/validation.py legacy/tune3_data/validation.py
   git rm pytest.ini        # a configuração do pytest agora está em pyproject.toml
   (se o .pre-commit-config.yaml estiver vazio — 0 bytes —, git rm nele também)

2. Substituir TODO o conteúdo de .github/workflows/ci.yml por:
---
name: Tune3 Continuous Integration
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install (CPU torch)
      run: |
        python -m pip install --upgrade pip
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install -e ".[dev]"
    - name: Run tests
      run: pytest tune3/tests/ -v --cov=tune3 -m "not slow"
---

3. Reinstalar o pacote com os novos extras e rodar a suíte completa:
   pip install -e ".[dev,vision]"
   pytest -q
   Esperado: 88 passed (1 warning do torch.jit sob Python 3.14 é normal). Se algo falhar,
   PARE e me mostre o erro completo — não tente corrigir sozinho.

4. Rodar a validação teoria x implementação em dados sintéticos:
   python scripts/validate_theory.py --tag dionatan
   Esperado: 11 itens, 0 REFUTADO (V6b fica "EM ABERTO" por desenho). Me mostre a tabela.

5. Conferir que nada de legado ficou importado:
   git grep -n "optuna_optimizer\|ModelTrainer\|data.loader\|CurvatureAwareOptuna" -- tune3 scripts
   Esperado: nenhuma linha (só comentários, se houver).

6. Commitar em DOIS commits, sem --amend e sem --no-verify:
   git add -A
   git commit -m "fix(protocolo): separa validacao/teste na avaliacao final; EoS recebe so curvaturas novas; Hutchinson em eval()

   Auditoria de 16/09/2026 (docs/AUDITORIA_2026_09.md). A avaliacao final usava o TESTE
   para early stopping e escolha de epoca (vazamento) -> H1-H3 devem ser reexecutados.
   EoSDetector nunca disparava (valor repetido). Hutchinson rodava com dropout ativo.
   Adiciona B4 (BO mono-objetivo em CVaR), placebo bi-objetivo e ablacao do DDKF (E1/E2),
   bootstrap BCa + regra pre-registrada, analise por seed em S2, registro padronizado de
   resultados (results/), PREREGISTRATION.md (rascunho), 14 testes-sentinela novos (88 total).
   Move o pipeline legado Optuna/Breast-Cancer para legacy/; declara torchvision/pytest
   como extras; corrige CI."
   git tag -a auditoria-2026-09-16 -m "estado corrigido pos-auditoria; base para reexecucao de H1-H3"
   git push origin main --tags

7. Me mostre `git log --oneline -3` e `git status` (deve estar limpo).
```

## Passo 3 — Depois do push (você, no navegador)

1. Abra o repositório no GitHub → aba **Actions**: a CI deve ficar verde (antes ela quebrava
   na coleta por falta de `torchvision`).
2. Revogue o token antigo do GitHub, se ainda não fez (Settings → Developer settings → Tokens).
3. Abra `PREREGISTRATION.md`, preencha **data** e **hash** (`git rev-parse --short HEAD`),
   marque a predição do E2, e commite: `git commit -am "prereg: congela protocolo" && git push`.
   A partir desse commit, nenhuma execução "oficial" pode alterar o protocolo.

## Decisões que só você pode tomar (estão listadas em `docs/AUDITORIA_2026_09.md`, §3 e §5)

- **D1** — o DDKF observa `(train_loss, val_loss, log GSNR)`, **não** Tr(H²). Manter e corrigir o
  manuscrito, ou acrescentar a curvatura à observação (muda o método; vira ablação).
- **D2** — não existe scheduler base (Cosine); o Log-AR(1) gira em torno de `lr0` fixo.
- **D3** — o MACRO usa qLogNEHVI **sem** a restrição L*; GP homoscedástico (Fix 1 pendente).
- **`curvature_every`**: 5 (padrão) ou 1? Com 5 o EoS precisa de 30 épocas de histórico.
- Os JSONs antigos do Drive têm o vazamento: **E0 "de custo zero" só vale sobre resultados novos.**
