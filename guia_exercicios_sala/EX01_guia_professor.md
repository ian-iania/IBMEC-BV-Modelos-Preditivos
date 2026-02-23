# Versão Professor (cole como uma célula Markdown no topo do notebook)

## 👨‍🏫 Guia do Professor — EX1 Comitê de Crédito FP&A (Cutoff por Custo)

### Objetivo pedagógico (em 30s)

- Fixar a diferença entre:
  - **MODELO** (separação: AUC/PR-AUC)
  - **POLÍTICA** (decisão: cutoff → Precision/Recall/F1)
  - **RESULTADO** (payoff em R$: FN vs FP)
- Mostrar que **premissas mudam a decisão**: cenário macro e apetite de risco alteram o cutoff ótimo.

---

## Como conduzir em sala (roteiro rápido)

### 1. Setup (1 minuto)

Diga algo como:

> "Hoje vocês são o Comitê de Crédito. O modelo só dá probabilidade de default. Quem decide é a política (cutoff). E quem paga a conta é o FP&A em R$."

### 2. Execução em grupos (10–15 minutos)

- Forme grupos de 3–4 pessoas.
- Uma pessoa abre o notebook no Colab e muda **só 3 variáveis**:
  - `scenario` (`BASE` / `STRESS` / `EXPANSAO`)
  - `risk_appetite` (`CONSERVADOR` / `BALANCEADO` / `AGRESSIVO`)
  - `chosen_model` (`LOGISTICA` / `GBM`)
- Rodar **Run all**.
- Preencher a seção **"✅ Resposta do Grupo (copiar e colar)"**.

### 3. Coleta de respostas (5–8 minutos)

- Chame 3 grupos (um de cada cenário) para ler seus 3 bullets.

### 4. Debrief (5 minutos)

Perguntas guiadas:

1. **Qual cutoff vocês escolheram e por quê?**
2. **O que pesou mais: FN ou FP?** (perda vs margem)
3. **Se o cenário mudasse, vocês mudariam o cutoff?**
4. **Por que escolheram logística ou GBM?** (governança vs performance)

---

## Distribuição de premissas (para garantir diversidade)

### Opção 1 (recomendada): distribuição fixa por grupo

- **Grupos 1–2:** `scenario="STRESS"` + `risk_appetite="CONSERVADOR"`
- **Grupos 3–4:** `scenario="BASE"` + `risk_appetite="BALANCEADO"`
- **Grupos 5–6:** `scenario="EXPANSAO"` + `risk_appetite="AGRESSIVO"`

E divida modelos assim:

- metade dos grupos usa `chosen_model="LOGISTICA"`
- metade usa `chosen_model="GBM"`

✅ Resultado: respostas diferentes e comparáveis.

### Opção 2: escolha livre (se a turma estiver madura)

Deixe cada grupo escolher, mas exija:

- 1 grupo precisa ser `STRESS`
- 1 grupo precisa ser `EXPANSAO`
- metade usa `LOGISTICA` e metade `GBM`

---

## Regras de avaliação (simples e transparentes)

A resposta do grupo é "boa" se tiver:

1. **Cutoff + % aprovação** (volume)
2. **Perda (FN), margem perdida (FP) e custo total** (R$)
3. **Justificativa coerente** com:
   - 1 métrica (AUC ou PR-AUC)
   - 1 trade-off (FN vs FP)
   - 1 frase de governança (logística explica / GBM performa)

---

## Mensagens-chave para reforçar (para conectar com os slides)

- **AUC/PR-AUC** → mede **separação/ranking** (não decide política).
- **Cutoff** → é **política** (transforma probabilidade em decisão).
- **FN vs FP** → têm **custos diferentes em R$** (FP&A decide pelo payoff).
- **Cenário e apetite** mudam o cutoff ótimo.
- **Logística** costuma ser mais explicável; **GBM** pode ganhar performance (com governança maior).

---

## Dicas práticas (para evitar travas)

- Se o Colab estiver lento, peça para:
  - só 1 pessoa rodar o notebook por grupo
  - fechar abas
- Se houver discussão "sem fim", force a decisão:
  - "Escolham o cutoff que minimiza custo total" **OU**
  - "Escolham cutoff que bate uma meta mínima de aprovação (ex.: ≥ 60%)"

---

## Pergunta bônus (se sobrar tempo)

> "Se a diretoria exigir uma meta mínima de aprovação (volume), como isso muda a decisão do cutoff?"

(Isso introduz o conceito real de **restrições** em otimização de política.)
