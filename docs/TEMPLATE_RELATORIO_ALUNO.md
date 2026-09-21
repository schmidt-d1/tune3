# Relatório de execução — Tune3

> Copie este arquivo para `results/relatorios/<seu-nome>_<AAAA-MM-DD>.md`, preencha e
> commite junto com os JSONs. Um relatório por rodada de experimentos.

| Campo | Preencha |
|---|---|
| Aluno | |
| Data(s) de execução | |
| Máquina (CPU / GPU / RAM / SO) | |
| Commit do repositório (`git rev-parse --short HEAD`) | |
| Árvore limpa antes de rodar? (`git status`) | sim / não |
| Python / torch (saída de `scripts/verify_setup.py`) | |

## 1. Teste 0 — sanidade (obrigatório)
- `pytest -q` → `___ passed` em `___ s`  (esperado: 89 passed)
- `python scripts/validate_theory.py --csv data/drebin215.csv --device cuda --tag <nome>`
  → itens REFUTADO: `___` (esperado: 0). Cole a tabela de vereditos abaixo.

```
(cole aqui)
```

## 2. Experimentos executados
| Experimento | Comando exato | Seeds | Épocas | Métodos | Duração | Arquivo em results/ |
|---|---|---|---|---|---|---|
| | | | | | | |

## 3. O que observei (fatos, não interpretação)
- Medianas de CVaR de teste por método (copie do terminal):
- Comparações pré-registradas (`d_z`, IC95 BCa, `p_holm`, `prereg_win`):
- Telemetria: fração de épocas com DDKF ativo; nº de disparos do EoS; nº de trials abortados:

## 4. O que deu errado / avisos
(erros, `[AVISO] arvore git SUJA`, trials abortados, OOM, tempo maior que o esperado…)

## 5. Dúvidas para o coordenador
