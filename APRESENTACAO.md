# Roteiro de apresentação — Tech Challenge Fase 1

Guia para o vídeo de demonstração. São 7 blocos e cerca de **13 minutos**, abaixo do limite de 15.

Não precisa decorar as falas: leia algumas vezes, entenda a ideia e fale com as suas palavras. As frases citadas são só uma base.

---

## Visão geral

| Bloco | Tempo | O que aparece na tela |
|---|---|---|
| 1. Abertura e problema | 0:00 – 1:30 | README no GitHub |
| 2. Projeto e execução com Docker | 1:30 – 3:30 | PyCharm e terminal rodando o Docker |
| 3. Análise exploratória | 3:30 – 5:00 | Gráficos da pasta `outputs/` |
| 4. Modelos e resultados | 5:00 – 7:00 | Código dos modelos e saída do terminal |
| 5. Interpretação com SHAP | 7:00 – 8:30 | `shap_summary.png` e `shap_individual.png` |
| 6. Extra: CNN | 8:30 – 11:30 | Notebook no Colab e gráficos de `outputs/cnn/` |
| 7. Discussão e encerramento | 11:30 – 13:00 | Relatório técnico |

---

## Antes de gravar

**Deixe aberto, nesta ordem:**

- [ ] Navegador com o repositório no GitHub (página do README)
- [ ] PyCharm com o projeto Rain_DB e o `main.py` aberto
- [ ] Terminal na pasta do projeto, com o Docker Desktop rodando
- [ ] Pasta `outputs/` aberta no gerenciador de arquivos
- [ ] Notebook da CNN no Colab, **já executado**
- [ ] Relatório técnico

**Dicas:**

- **Rode o `docker build` antes de gravar.** Assim a imagem já fica pronta e, no vídeo, você só executa o `docker run`, que leva poucos segundos.
- **Não treine a CNN ao vivo**, porque leva cerca de 15 minutos. Mostre o notebook já executado e explique os resultados.
- **Para gravar no Ubuntu**, use o OBS Studio (`sudo apt install obs-studio`). Grave a tela inteira, com o microfone ligado.
- Aumente a fonte do PyCharm e do terminal para ficarem legíveis no vídeo.
- Feche as abas que não têm a ver com o projeto e silencie as notificações.
- Se errar uma fala, pare, respire e repita a frase. Depois é só cortar o trecho. Você pode gravar bloco por bloco e juntar depois.
- Envie para o YouTube como **Não listado**.

> ⚠️ **Confira antes de gravar:** a legenda do `outputs/radius_mean_diagnostico.png` precisa mostrar **azul = Benigno** e **laranja = Maligno**. Se ainda aparecer só laranja, aplique a correção do `df_plot` no `perform_eda` e rode o Docker de novo.

---

## Bloco 1 — Abertura e problema (0:00 – 1:30)

**Tela:** a página do README no GitHub. Role devagar até a tabela do dataset.

**Fala:**

> "Olá! Meu nome é [seu nome], e este é o nosso projeto do Tech Challenge da Fase 1 da PÓS TECH.
>
> O desafio é de um hospital universitário que quer usar inteligência artificial para apoiar o diagnóstico médico. Com o volume de exames crescendo, o hospital precisa acelerar a triagem sem perder qualidade.
>
> O problema que escolhemos foi o diagnóstico de câncer de mama: a partir de um exame, dizer se o tumor é benigno ou maligno.
>
> Usamos o dataset Breast Cancer Wisconsin, que tem 569 pacientes e 30 medidas das células do tumor, como raio, área, textura e concavidade. Cerca de 63% dos casos são benignos e 37% são malignos.
>
> E, como extra, também treinamos uma rede neural para fazer o mesmo diagnóstico diretamente a partir de imagens de mamografia.
>
> Um ponto importante desde já: esse sistema é uma ferramenta de apoio. Quem decide o diagnóstico é sempre o médico."

---

## Bloco 2 — Estrutura do projeto e execução com Docker (1:30 – 3:30)

**Tela:** primeiro o PyCharm, com a árvore de pastas aberta. Depois mostre rapidamente o `Dockerfile` e passe para o terminal.

**Fala, mostrando as pastas:**

> "O projeto está organizado assim: em `data` fica o dataset, o `main.py` tem todo o pipeline, a pasta `notebooks` tem a parte extra com a rede neural e a pasta `outputs` guarda os gráficos e os resultados.
>
> O `main.py` está dividido em funções, uma para cada etapa: carregar os dados, fazer a análise exploratória, separar os dados, treinar os modelos, avaliar e interpretar."

**Fala, mostrando o `Dockerfile`:**

> "Para que qualquer pessoa consiga rodar o projeto do mesmo jeito, colocamos tudo em um container Docker. O Dockerfile usa uma imagem do Python, instala as bibliotecas do `requirements.txt` e executa o `main.py`."

**No terminal, rode:**

```bash
docker run --rm -v "${PWD}/outputs:/app/outputs" tech-challenge
```

**Fala, enquanto a saída aparece:**

> "Com um único comando, o pipeline inteiro roda: ele carrega os dados, gera os gráficos, treina os dois modelos, avalia e gera as explicações. O `-v` faz os gráficos criados dentro do container aparecerem na pasta `outputs` do computador."

---

## Bloco 3 — Análise exploratória (3:30 – 5:00)

**Tela:** abra os gráficos da pasta `outputs/`, um de cada vez.

**Com o `correlacao_diagnostico.png`:**

> "Antes de treinar qualquer modelo, exploramos os dados. Este gráfico mostra quais medidas têm mais relação com o diagnóstico. As mais fortes são as de tamanho do tumor, como perímetro e raio, e as de concavidade, que medem o quanto o contorno do tumor é irregular. Faz sentido clinicamente: tumores malignos costumam ser maiores e mais irregulares."

**Com o `radius_mean_diagnostico.png`:**

> "Aqui fica bem visível: em azul os benignos, em laranja os malignos. Os malignos têm o raio claramente maior. Mas reparem que existe uma faixa em que as duas cores se misturam. É nessa zona de dúvida que o modelo precisa combinar várias medidas para decidir."

**Fala sobre a limpeza (pode ser em cima do mesmo gráfico):**

> "Os dados já vinham bem limpos. Removemos só a coluna de identificação e uma coluna vazia, e convertemos o diagnóstico para números: zero para benigno e um para maligno."

---

## Bloco 4 — Modelos, métrica e resultados (5:00 – 7:00)

**Tela:** no PyCharm, mostre as funções `split_data` e `build_models`. Depois volte ao terminal com a saída do Docker e, por fim, abra `matriz_confusao.png`.

**Com o `split_data`:**

> "Separamos os dados em três partes: 60% para treino, 20% para validação e 20% para teste. O treino serve para o modelo aprender, a validação serve para escolher o melhor modelo, e o teste é usado uma única vez, no final, como uma prova que o modelo nunca viu. A separação é estratificada, então cada parte mantém a mesma proporção de benignos e malignos."

**Com o `build_models`:**

> "Comparamos dois modelos. A Regressão Logística, que é simples e fácil de interpretar, e o Random Forest, que combina várias árvores de decisão e capta relações mais complexas. A Regressão Logística precisa que todas as medidas estejam na mesma escala, por isso ela tem o StandardScaler dentro do pipeline. Assim, a padronização aprende só com os dados de treino, e nada da validação ou do teste vaza para o modelo."

**Com a saída do terminal (parte da comparação dos modelos):**

> "Agora, a escolha mais importante do projeto: qual métrica usar. Em diagnóstico de câncer, o pior erro é dizer que um tumor maligno é benigno, porque a paciente pode ficar sem tratamento. Por isso escolhemos o modelo pelo recall da classe maligna, que mede quantos casos malignos o modelo consegue encontrar. A Regressão Logística teve recall de 97,7% na validação, contra 95,3% do Random Forest, e foi a escolhida."

**Com o `matriz_confusao.png`:**

> "No teste final, o modelo acertou 97,4% dos casos. Dos 42 tumores malignos, encontrou 40, ou seja, um recall de 95,2%. Ele errou 3 casos em 114: um benigno classificado como maligno e dois malignos classificados como benignos. Esses dois são justamente os erros mais graves, e voltaremos a eles na discussão."

---

## Bloco 5 — Interpretação com SHAP (7:00 – 8:30)

**Tela:** abra `shap_summary.png` e depois `shap_individual.png`.

**Com o `shap_summary.png`:**

> "Um modelo que acerta muito não basta. O médico precisa entender por que o modelo chegou a uma resposta. Para isso usamos o SHAP, que mede o quanto cada medida contribuiu para cada previsão.
>
> Neste gráfico, cada ponto é um paciente. As medidas mais importantes ficam no topo, como a textura, os pontos côncavos e o raio. A cor mostra o valor da medida: vermelho é alto e azul é baixo. Pontos à direita empurram a previsão para maligno. Dá para ver que valores altos dessas medidas puxam para maligno, o que bate com o que vimos na análise exploratória."

**Com o `shap_individual.png`:**

> "E este é o ponto mais útil para o médico: a explicação de um único paciente. Cada barra mostra uma medida e o quanto ela empurrou a decisão para um lado ou para o outro. Assim o médico não recebe só um 'maligno' ou 'benigno', mas também o motivo, e pode concordar ou discordar."

**Se sobrar tempo, mencione:**

> "Também geramos a importância das variáveis do Random Forest. O ranking é um pouco diferente, porque são modelos diferentes e porque raio, perímetro e área medem quase a mesma coisa. Mesmo assim, os dois concordam que tamanho e concavidade são os sinais principais."

---

## Bloco 6 — Extra: CNN com mamografias (8:30 – 11:30)

**Tela:** o notebook já executado no Colab. Role até cada gráfico enquanto fala. Os mesmos gráficos também estão em `outputs/cnn/`.

**Com a célula de exemplos de imagens:**

> "Como parte extra, tentamos fazer o mesmo diagnóstico de outro jeito: em vez de medidas prontas, a rede recebe a própria imagem da mamografia. Usamos o dataset CBIS-DDSM, com cerca de 3.500 recortes de lesões, entre massas e calcificações. Em cima ficam exemplos benignos e embaixo, malignos. Dá para ver que, a olho nu, não é nada óbvio distinguir uns dos outros."

**Com a célula do modelo:**

> "Treinamos no Google Colab, com GPU. Como 3 mil imagens são poucas para treinar uma rede do zero, usamos transfer learning: pegamos a EfficientNet, uma rede que já aprendeu a reconhecer bordas, texturas e formas em milhões de imagens, e trocamos só a parte final, para ela responder 'benigno ou maligno'. Também tomamos o cuidado de separar treino e validação por paciente, para que a rede não reconheça a mesma paciente em vez de aprender o padrão da lesão."

**Com as curvas de treinamento:**

> "As curvas de treino e de validação andam juntas, então a rede não decorou os dados de treino. E a curva ainda estava subindo no final, o que indica que, com mais treino, o resultado provavelmente melhoraria."

**Com a matriz de confusão e a curva ROC:**

> "O resultado foi uma AUC de 0,73 no teste, bem abaixo do modelo tabular. Para não deixar passar câncer, ajustamos o limiar de decisão: a rede passa a classificar como maligno qualquer caso com mais de 27% de probabilidade. Com isso, ela encontrou 90% das lesões malignas. O preço é alto: muitos benignos foram marcados como suspeitos, o que significaria mais exames e biópsias."

**Com o Grad-CAM:**

> "Por fim, o Grad-CAM, que é o equivalente do SHAP para imagens: ele mostra em que região da imagem a rede prestou atenção. Em alguns casos ela olha para a lesão, como esperado. Mas em vários outros, inclusive nos erros, ela olha para as bordas do recorte, que não têm nada a ver com o diagnóstico. Isso mostra que a rede ainda não é confiável, e foi justamente a interpretação que nos permitiu perceber isso."

---

## Bloco 7 — Discussão crítica e encerramento (11:30 – 13:00)

**Tela:** o relatório técnico, na seção 10, com a tabela que compara os dois modelos.

**Fala:**

> "Comparando as duas abordagens: o modelo tabular foi muito melhor. Isso acontece porque ele recebe medidas que especialistas já extraíram do exame, ou seja, a parte mais difícil já foi feita. A rede neural precisa aprender sozinha o que observar, a partir de imagens antigas e com poucos exemplos.
>
> E esse modelo pode ser usado na prática? O modelo tabular pode ser testado como ferramenta de apoio, por exemplo para colocar no topo da fila os exames com maior chance de malignidade, ou como uma segunda opinião, mostrando ao médico a probabilidade junto com a explicação do SHAP.
>
> Mas há limitações importantes. O dataset é pequeno e vem de um único hospital, então o modelo precisaria ser validado com dados de outros lugares. E mesmo com 95% de recall, ele deixou passar dois casos malignos. Por isso, em nenhum momento o modelo decide sozinho. Quem tem a palavra final é sempre o médico.
>
> Já a rede neural ainda não está pronta para uso, mas mostrou o caminho para as próximas fases: treinar por mais tempo, usar imagens com mais resolução e garantir que ela olhe para a lesão.
>
> O código completo, o Dockerfile, os resultados e o relatório técnico estão no repositório do GitHub, que está no link da entrega. Obrigado!"

---

## Depois de gravar

- [ ] Conferir se o vídeo tem menos de 15 minutos
- [ ] Enviar ao YouTube como **Não listado**
- [ ] Colocar o link do vídeo no README e no relatório
- [ ] Exportar o relatório em PDF com os links do repositório e do vídeo

---

## Números para ter à mão

| Item | Valor |
|---|---|
| Pacientes (Wisconsin) | 569 — 357 benignos, 212 malignos |
| Divisão | 341 treino · 114 validação · 114 teste |
| Recall na validação | Regressão Logística 97,7% · Random Forest 95,3% |
| Teste — acurácia | 97,4% |
| Teste — recall (maligno) | 95,2% (40 de 42) |
| Teste — erros | 2 falsos negativos · 1 falso positivo |
| CNN — imagens | 3.566 recortes (2.277 treino · 585 validação · 704 teste) |
| CNN — AUC no teste | 0,73 |
| CNN — limiar ajustado | 0,274 → recall 89,9% (248 de 276) |
