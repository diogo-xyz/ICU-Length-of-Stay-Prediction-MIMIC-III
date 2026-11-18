# Previsão do Tempo de Permanência em UCI (MIMIC-III)

Pipeline completo de Machine Learning para prever o **tempo de permanência
(*Length of Stay*, LOS) de doentes em Unidades de Cuidados Intensivos (UCI)**,
usando apenas a informação das **primeiras 24 horas** de internamento. O
projeto vai desde a ingestão de centenas de milhões de registos clínicos até à
interpretabilidade dos modelos finais.

Projeto académico desenvolvido na Licenciatura em Inteligência Artificial e
Ciência de Dados da Faculdade de Ciências da Universidade do Porto (FCUP),
ano letivo 2025/2026.

---

## Contexto e objetivo

Uma estimativa fiável do LOS na UCI ajuda a gestão hospitalar a:

- **otimizar a alocação de camas**, planeando admissões e altas;
- **planear recursos**: equipas, equipamentos e medicação;
- **identificar doentes de alto risco**, com previsão de internamento prolongado;
- **reduzir custos operacionais**.

O objetivo é prever o LOS de duas formas complementares:

- **Regressão:** número de dias na UCI.
- **Classificação:** 4 classes de LOS, definidas pelos quartis da distribuição.

---

## Dados

O projeto usa a base de dados clínica **[MIMIC-III](https://physionet.org/content/mimiciii/)**,
com dados de doentes internados em cuidados intensivos.

>  **Os dados não estão incluídos neste repositório.** O acesso ao MIMIC-III
> exige credenciação na PhysioNet e a assinatura de um acordo de utilização
> de dados (DUA), que proíbe a redistribuição. Para reproduzir o projeto, é
> necessário obter acesso próprio.

As principais tabelas utilizadas são:

| Tabela | Conteúdo | Dimensão aproximada |
|---|---|---|
| `CHARTEVENTS` | Sinais vitais e monitorização | ~330 milhões de linhas |
| `LABEVENTS` | Análises laboratoriais | ~27 milhões de linhas |
| `ICUSTAYS`, `ADMISSIONS`, `PATIENTS` | Estadias, admissões e dados demográficos | — |

Dataset final: **61 522 estadias em UCI**, de **46 467 doentes**, com **93 variáveis**.

---

## Pipeline

### Fase 1 — Ingestão e processamento em larga escala (BigQuery)

Com centenas de milhões de linhas, o processamento local com `pandas` seria
impraticável. Toda a agregação foi feita no **Google BigQuery**:

- **Queries de frequência** sobre `CHARTEVENTS` e `LABEVENTS` para escolher
  sinais vitais e análises que são, ao mesmo tempo, frequentes, clinicamente
  relevantes e não redundantes.
- **Query de agregação** que gera **uma linha por estadia na UCI**, com média,
  mínimo, máximo, desvio padrão e amplitude de cada sinal nas primeiras 24 h.
- **Tabela auxiliar** com as séries temporais de frequência cardíaca, usada na
  análise com *Dynamic Time Warping*.

| Query | Tempo real | Dados processados |
|---|---|---|
| Top 50 `CHARTEVENTS` | 0,45 s | 4,37 GB |
| Top 50 `LABEVENTS` | 0,41 s | 0,40 GB |
| Dataset final | 3,72 s | 12,74 GB |
| Séries temporais (DTW) | 2,36 s | 9,30 GB |

### Fase 2 — Análise exploratória, *feature engineering* e modelação (Python)

- **EDA:** distribuição do LOS (fortemente assimétrica, mediana ≈ 2,1 dias),
  análise demográfica, correlações e readmissões.
- ***Feature engineering*:** variabilidade dos sinais vitais, SOFA *score*
  parcial, variáveis temporais da admissão (hora, dia, turno) e indicadores de
  valores em falta.
- **Imputação** com `IterativeImputer`.
- **Clustering:**
  - K-Means (K = 5) sobre variáveis clínicas, que identifica perfis com
    diferenças significativas de LOS e mortalidade;
  - *Dynamic Time Warping* (K = 2) sobre as séries de frequência cardíaca,
    com associação estatisticamente significativa aos clusters clínicos
    (χ² = 248,59; p < 0,001).
- **Modelação:**
  - regressão com Ridge (*baseline*), Random Forest e XGBoost;
  - classificação com Regressão Logística (*baseline*), Random Forest e XGBoost;
  - divisão treino/teste **agrupada por doente** (`GroupShuffleSplit`), para
    evitar *data leakage* entre estadias da mesma pessoa;
  - otimização de hiperparâmetros com `RandomizedSearchCV`.
- **Interpretabilidade** com **SHAP**.

### Fase 3 — *Profiling* e análise de desempenho

- Medição do tempo de execução e do pico de memória de cada fase do pipeline.
- Paralelismo com `joblib` (*threading* e *loky*) na EDA, no clustering e na
  otimização de hiperparâmetros.
- Justificação das ferramentas escolhidas (por exemplo, BigQuery em vez de
  PySpark).
- Tempo de inferência num cenário de produção.

---

## Resultados

| Tarefa | Melhor modelo | Métricas |
|---|---|---|
| Regressão (LOS em dias) | XGBoost | MAE = 3,18 dias; R² = 0,34 |
| Classificação (4 classes) | XGBoost | F1 = 0,49; AUC = 0,76; κ quadrático = 0,53 |

**Principais conclusões:**

- Mesmo com apenas 24 h de dados, o modelo tem valor discriminativo real
  (AUC = 0,76; κ moderado), sobretudo nas classes extremas: altas muito
  rápidas e internamentos longos.
- Segundo o SHAP, as variáveis mais preditivas são o **número de eventos
  laboratoriais e de monitorização** nas primeiras 24 h (um *proxy* da
  gravidade do doente) e a **idade**.
- A otimização de hiperparâmetros ocupa cerca de 91% do tempo total do
  pipeline.
- Todos os modelos fazem uma previsão em menos de 60 ms por doente, o que os
  torna viáveis para suporte à decisão clínica em tempo real.

---

## Estrutura do repositório

```
.
├── icu_length_of_stay.ipynb      # Notebook com todo o pipeline (fases 1, 2 e 3)
├── requirements.txt  # Dependências Python
└── README.md
```

Os ficheiros de dados gerados pelo BigQuery (`chartevents_50.csv`,
`labevents_50.csv`, `dataset_final_6.csv` e `timeseries_hr.csv`) **não são
distribuídos**, pelas restrições de utilização do MIMIC-III.

---

## Como executar

1. Obter acesso ao MIMIC-III através da [PhysioNet](https://physionet.org/content/mimiciii/)
   e disponibilizar os dados no BigQuery.
2. Correr as queries SQL da Fase 1 (incluídas no notebook) e exportar os
   resultados para CSV, na mesma pasta do notebook.
3. Criar o ambiente e instalar as dependências (desenvolvido em Python 3.12):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Abrir e correr o notebook:

   ```bash
   jupyter notebook icu_length_of_stay.ipynb
   ```

---

## Trabalho futuro

- **Remover variáveis administrativas** (contagens de eventos), que refletem
  decisões médicas e não apenas o estado clínico do doente.
- **Incorporar notas clínicas** com técnicas de NLP/LLM.
- **Modelação temporal** com redes recorrentes (RNN) sobre as séries completas.

---

## Autores

- João Pedro Nunes Rocha
- Paulo Diogo Lopes Pinto
- Rui Filipe Rocha Alvarilhão

Faculdade de Ciências da Universidade do Porto (FCUP), 2025/2026.
