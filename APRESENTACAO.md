<!--
COMO USAR: abra este arquivo no PyCharm com a tela dividida (editor + pré-visualização).
As linhas que começam com 🎙️ são marcas da narração: elas aparecem só no editor,
nunca na pré-visualização. Quando o áudio chegar na frase marcada, role a
pré-visualização até a seção logo abaixo da marca.
-->

# Ray_DB - Diagnóstico de Câncer de Mama com Machine Learning

**Tech Challenge — Fase 1 · PÓS TECH · Davi Valério**

---

<!-- 🎙️ "Olá! Meu nome é Davi Valério..." -->

## 1. O problema

- Um hospital universitário precisa **acelerar a triagem**, reduzir erros e otimizar o tempo dos profissionais.
- Objetivo: classificar tumores de mama como **benignos** ou **malignos**.

<!-- 🎙️ "Para a parte principal, usamos o dataset Breast Cancer Wisconsin..." -->

### Dataset — Breast Cancer Wisconsin

| Característica | Valor |
|---|---|
| Pacientes | 569 |
| Variáveis | 30 medidas das células do tumor |
| Exemplos de medidas | raio, área, textura, concavidade |
| Benignos | 357 (63%) |
| Malignos | 212 (37%) |

<!-- 🎙️ "Como extra, também treinamos uma rede neural convolucional..." -->

| Parte | Dados | Técnica |
|---|---|---|
| Principal | 30 medidas (tabela) | Regressão Logística e Random Forest |
| Extra | Imagens de mamografia | Rede neural convolucional (CNN) |

> O sistema é uma **ferramenta de apoio**. Quem decide o diagnóstico é **sempre o médico**.

---

<!-- 🎙️ "Vamos começar pela organização do projeto..." -->

## 2. Estrutura do projeto

```
Rain_DB/
├── data/data.csv           → dataset
├── main.py                 → pipeline da parte principal
├── notebooks/              → parte extra (CNN)
├── outputs/                → gráficos e resultados
├── Dockerfile              → execução em container
└── requirements.txt        → bibliotecas com versões fixadas
```

<!-- 🎙️ "O arquivo main está dividido em funções..." -->

**Uma função para cada etapa do `main.py`:**

1. `load_data` → carregar e limpar
2. `perform_eda` → análise exploratória
3. `split_data` → separar os dados
4. `build_models` → construir os modelos
5. `train_and_select_model` → treinar e escolher o melhor
6. `evaluate_model` → avaliar no teste
7. `analyze_feature_importance` e `analyze_with_shap` → interpretar

<!-- 🎙️ "Para garantir que qualquer pessoa consiga rodar o projeto..." -->

### Docker

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

<!-- 🎙️ "Com um único comando, o pipeline inteiro roda..." -->

**Um único comando executa o pipeline inteiro:**

```bash
docker run --rm -v "${PWD}/outputs:/app/outputs" tech-challenge
```

---

<!-- 🎙️ "Antes de treinar qualquer modelo, exploramos os dados..." -->

## 3. Análise exploratória

### Qualidade dos dados

| Verificação | Resultado |
|---|---|
| Valores ausentes | Nenhum |
| Coluna `id` | Removida (sem valor para o diagnóstico) |
| Coluna `Unnamed: 32` | Removida (totalmente vazia) |
| Diagnóstico | **B → 0** (benigno) · **M → 1** (maligno) |

<!-- 🎙️ "Este gráfico mostra quais medidas têm mais relação com o diagnóstico..." -->

### Correlação com o diagnóstico

![Correlação com o diagnóstico](outputs/correlacao_diagnostico.png)

- Mais fortes: **tamanho** (perímetro, raio, área) e **concavidade**.
- Maior correlação: pontos côncavos, com **0,79**.

<!-- 🎙️ "Aqui, na distribuição do raio médio, isso fica bem visível..." -->

### Distribuição do raio médio

![Distribuição de radius_mean](outputs/radius_mean_diagnostico.png)

- **Azul:** benignos · **Laranja:** malignos.
- Os malignos têm o raio **claramente maior**.
- **Zona de dúvida** entre 13 e 16: uma medida só não basta para decidir.

<!-- 🎙️ "Também notamos que as medidas estão em escalas muito diferentes..." -->

### Escalas diferentes

| Medida | Ordem de grandeza |
|---|---|
| `area_mean` | mais de 1.000 |
| `smoothness_mean` | cerca de 0,1 |

➡️ É preciso **padronizar** as medidas para a Regressão Logística.

---

<!-- 🎙️ "Para treinar e avaliar de forma justa, separamos os dados em três partes..." -->

## 4. Separação dos dados

| Conjunto | Proporção | Registros | Uso |
|---|---|---|---|
| Treino | 60% | 341 | O modelo aprende |
| Validação | 20% | 114 | Comparar e escolher o melhor modelo |
| Teste | 20% | 114 | Prova final, usada uma única vez |

A separação é **estratificada**: cada parte mantém a mesma proporção de benignos e malignos.

---

<!-- 🎙️ "Comparamos dois modelos de natureza diferente..." -->

## 5. Modelos

| Modelo | Característica |
|---|---|
| **Regressão Logística** | Linear, simples e fácil de interpretar |
| **Random Forest** | 200 árvores de decisão, capta relações mais complexas |

<!-- 🎙️ "A Regressão Logística precisa que todas as medidas estejam na mesma escala..." -->

### Pipeline e vazamento de dados

```python
Pipeline([
    ("scaler", StandardScaler()),       # aprende média e desvio só no treino
    ("modelo", LogisticRegression()),
])
```

| ❌ Padronizar a base inteira antes de separar | ✅ Padronizar dentro do pipeline |
|---|---|
| O modelo "espia" a validação e o teste | A padronização aprende só com o treino |
| Resultados otimistas demais | Avaliação honesta |

---

<!-- 🎙️ "Agora, a escolha mais importante do projeto: qual métrica usar..." -->

## 6. A métrica: por que o recall?

| Erro | O que acontece | Gravidade |
|---|---|---|
| Falso positivo (benigno → maligno) | Exames adicionais e biópsia | Custo reversível |
| **Falso negativo** (maligno → benigno) | **A paciente fica sem tratamento** | **O erro mais grave** |

<!-- 🎙️ "A acurácia sozinha esconde essa diferença..." -->

### A armadilha da acurácia

> Um modelo que dissesse **"benigno" para todo mundo** acertaria **63%** dos casos e não serviria para nada.

- **Recall (maligno):** quantos casos malignos o modelo encontra → **critério de escolha**.
- **F1-score:** equilíbrio entre precisão e recall.

| Modelo | Recall na validação |
|---|---|
| **Regressão Logística** ✅ | **97,7%** |
| Random Forest | 95,3% |

---

<!-- 🎙️ "No teste final, o modelo acertou 97,4% dos casos..." -->

## 7. Resultado final no teste — Regressão Logística

| Acurácia | Recall (maligno) | Precisão (maligno) | F1-score (maligno) |
|---|---|---|---|
| **97,4%** | **95,2%** | 97,6% | 96,4% |

![Matriz de confusão](outputs/matriz_confusao.png)

- Encontrou **40 de 42** tumores malignos.
- 3 erros em 114 casos: 1 falso positivo e **2 falsos negativos**.

---

<!-- 🎙️ "Um modelo que acerta muito não basta..." -->

## 8. Interpretação do modelo

### SHAP — quanto cada medida contribui para cada previsão

<!-- 🎙️ "Neste gráfico, cada ponto é um paciente..." -->

![SHAP summary](outputs/shap_summary.png)

- Cada ponto é um paciente.
- 🔴 Vermelho = valor alto · 🔵 Azul = valor baixo.
- Direita → **maligno** · Esquerda → **benigno**.

<!-- 🎙️ "E este é o ponto mais útil para o médico: a explicação de um único paciente..." -->

### SHAP — explicação de um único paciente

![SHAP individual](outputs/shap_individual.png)

- Cada barra mostra o quanto uma medida empurrou a decisão.
- O médico recebe **a resposta e o motivo**, e pode discordar.

<!-- 🎙️ "Também geramos a importância das variáveis do Random Forest..." -->

### Feature importance — Random Forest

![Feature importance](outputs/feature_importance.png)

**Por que o ranking é diferente do SHAP?**
1. São **modelos diferentes**.
2. Raio, perímetro e área medem **quase a mesma coisa**, e a Regressão Logística divide o peso entre elas.

➡️ As duas técnicas concordam no essencial: **tamanho e concavidade** são os principais sinais de malignidade.

---

<!-- 🎙️ "Como parte extra, tentamos fazer o mesmo diagnóstico de outro jeito..." -->

## 9. Extra — CNN com mamografias (CBIS-DDSM)

![Exemplos de recortes](outputs/cnn/exemplos_imagens.png)

| Imagens | Tipos de lesão | Malignos |
|---|---|---|
| 3.566 recortes | Massas e calcificações | 41% |

<!-- 🎙️ "Um cuidado importante: treino e validação foram divididos por paciente..." -->

### Separação por paciente

| Conjunto | Imagens | Como foi formado |
|---|---|---|
| Treino | 2.277 | Pacientes do treino oficial |
| Validação | 585 | **Outras pacientes** do treino oficial |
| Teste | 704 | Divisão oficial do dataset |

➡️ A rede não pode **reconhecer a paciente** em vez de aprender o padrão da lesão.

<!-- 🎙️ "Treinamos no Google Colab, com GPU..." -->

### Transfer learning

- **EfficientNetB0**, pré-treinada em mais de **1 milhão de imagens** (ImageNet).
- Só a parte final foi trocada, para responder **benigno ou maligno**.
- Treino em **2 fases**: 1) só a parte nova · 2) ajuste fino das últimas camadas.
- Executado no **Google Colab com GPU**.

<!-- 🎙️ "As curvas de treino e de validação andam juntas..." -->

### Curvas de treinamento

![Curvas de treinamento](outputs/cnn/curvas_treinamento.png)

- Treino e validação juntos → **sem overfitting**.
- A validação **ainda subia** no final → mais treino deve melhorar.

<!-- 🎙️ "O resultado foi uma AUC de 0,73 no teste..." -->

### Resultados no teste — AUC 0,73

![Matriz de confusão e curva ROC da CNN](outputs/cnn/matriz_confusao_roc_cnn.png)

| Limiar | Recall (maligno) | Precisão | Acurácia |
|---|---|---|---|
| 50% (padrão) | 52,9% → só metade dos malignos | 62,1% | 68,9% |
| **27% (ajustado)** | **89,9%** | 46,9% | 56,1% |

- Com o limiar ajustado: **248 de 276** lesões malignas encontradas.
- O preço: **281 de 428** benignos marcados como suspeitos.

<!-- 🎙️ "Por fim, o Grad-CAM, que é o equivalente do SHAP para imagens..." -->

### Grad-CAM — onde a rede olha

![Grad-CAM](outputs/cnn/grad_cam.png)

- ✅ Em alguns casos, ela foca **na lesão**.
- ❌ Em outros, inclusive nos erros, ela foca **nas bordas do recorte**.

➡️ A rede **ainda não é confiável**, e foi a interpretação que mostrou isso.

---

<!-- 🎙️ "Comparando as duas abordagens: o modelo tabular foi muito melhor..." -->

## 10. Comparação

| Aspecto | Modelo tabular | CNN com imagens |
|---|---|---|
| Entrada | Medidas já extraídas por especialistas | Mamografias digitalizadas de filme |
| Recall (maligno) | **95,2%** | 89,9% |
| Falsos positivos | 1 em 72 | 281 em 428 |
| Interpretação | SHAP coerente com a clínica | Grad-CAM às vezes fora da lesão |

<!-- 🎙️ "E esse modelo pode ser usado na prática?..." -->

## 11. Uso na prática

| Uso | Como funciona |
|---|---|
| **Triagem** | Exames com maior chance de malignidade vão para o topo da fila |
| **Segunda opinião** | Probabilidade + explicação do SHAP para o médico |
| **Alerta** | Revisão dos casos em que o laudo e o modelo discordam |

<!-- 🎙️ "Mas há limitações importantes..." -->

### Limitações

- Dataset pequeno, de uma **única instituição** → precisa de validação com outros hospitais.
- Mesmo com 95% de recall, **2 casos malignos** passaram.

> **Em nenhum momento o modelo decide sozinho. A palavra final é sempre do médico.**

<!-- 🎙️ "Já a rede neural ainda não está pronta para uso..." -->

### Próximos passos da CNN

- Treinar por mais tempo.
- Usar imagens com mais resolução.
- Garantir que ela olhe para a lesão.

---

<!-- 🎙️ "O código completo, o Dockerfile, os resultados e o relatório técnico..." -->

## Obrigado!

**No repositório do GitHub:** código completo · Dockerfile · resultados · relatório técnico
