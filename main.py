from pathlib import Path
import shap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25

sns.set_theme(style="whitegrid")


# ============================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================

def load_data():
    """Carrega o dataset e realiza a limpeza inicial."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {DATASET_PATH}\n"
            "Verifique se o arquivo está em data/data.csv."
        )

    df = pd.read_csv(str(DATASET_PATH))

    print("\n" + "=" * 60)
    print("INFORMAÇÕES INICIAIS DO DATASET")
    print("=" * 60)

    print(f"Quantidade de registros: {df.shape[0]}")
    print(f"Quantidade de colunas: {df.shape[1]}")

    print("\nTipos de dados e valores não nulos:")
    df.info()

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nValores ausentes por coluna:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Remove colunas que não são preditoras.
    columns_to_drop = [
        column
        for column in ["id", "Unnamed: 32"]
        if column in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    if "diagnosis" not in df.columns:
        raise ValueError(
            "A coluna 'diagnosis' não foi encontrada no dataset."
        )

    if df["diagnosis"].isnull().any():
        raise ValueError(
            "A coluna 'diagnosis' contém valores ausentes."
        )

    # Codificação do alvo:
    # B = 0 (benigno)
    # M = 1 (maligno)
    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1,
    })

    if df["diagnosis"].isnull().any():
        raise ValueError(
            "Foram encontrados valores inesperados na coluna "
            "'diagnosis'. Verifique os rótulos do dataset."
        )

    print("\nDimensões após a limpeza:", df.shape)

    print("\nDistribuição dos diagnósticos:")
    print(
        df["diagnosis"]
        .map({0: "Benigno", 1: "Maligno"})
        .value_counts()
    )

    print("\nDistribuição percentual:")
    print(
        (
            df["diagnosis"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )

    return df


# ============================================================
# 3. ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
# ============================================================

def perform_eda(df):
    """Gera estatísticas e visualizações exploratórias."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cópia só para os gráficos, com o diagnóstico em texto
    df_plot = df.copy()
    df_plot["Diagnostic"] = df_plot["diagnosis"].map({0: "Benigno", 1: "Malign"})

    print("\n" + "=" * 60)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 60)

    print("\nEstatísticas descritivas:")
    print(df.describe().round(3))

    print("\nCorrelação das variáveis com o diagnóstico:")
    correlations = (
        df.corr(numeric_only=True)["diagnosis"]
        .drop("diagnosis")
        .sort_values(key=abs, ascending=False)
    )

    print(correlations.head(10).round(4))

    # Gráfico 1: correlação das variáveis com o diagnóstico.
    plt.figure(figsize=(12, 8))

    sns.barplot(
        x=correlations.head(15).values,
        y=correlations.head(15).index,
    )

    plt.title(
        "15 variáveis com maior correlação absoluta "
        "com o diagnóstico"
    )
    plt.xlabel("Correlação com o diagnóstico")
    plt.ylabel("Variável")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlacao_diagnostico.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Gráfico 2: distribuição de radius_mean por diagnóstico.
    if "radius_mean" in df.columns:
        plt.figure(figsize=(9, 6))

        sns.histplot(
            data=df,
            x="radius_mean",
            hue="diagnosis",
            bins=25,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
        )

        plt.title("Distribuição de radius_mean por diagnóstico")
        plt.xlabel("Raio médio")
        plt.ylabel("Densidade")
        plt.legend(
            title="Diagnóstico",
            labels=["Maligno", "Benigno"],
        )
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "radius_mean_diagnostico.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Gráfico 3: distribuição de concave points_mean.
    if "concave points_mean" in df.columns:
        plt.figure(figsize=(9, 6))

        sns.histplot(
            data=df,
            x="concave points_mean",
            hue="diagnosis",
            bins=25,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
        )

        plt.title(
            "Distribuição de concave points_mean por diagnóstico"
        )
        plt.xlabel("Pontos côncavos médios")
        plt.ylabel("Densidade")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "concave_points_mean_diagnostico.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Gráfico 4: boxplot de radius_mean.
    if "radius_mean" in df.columns:
        plt.figure(figsize=(9, 6))

        sns.boxplot(
            data=df,
            x="diagnosis",
            y="radius_mean",
        )

        plt.title("Raio médio por diagnóstico")
        plt.xlabel("Diagnóstico (0 = benigno, 1 = maligno)")
        plt.ylabel("Raio médio")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "boxplot_radius_mean.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"\nGráficos da EDA salvos em: {OUTPUT_DIR}")


# ============================================================
# 4. SEPARAÇÃO DOS CONJUNTOS DE DADOS
# ============================================================

def split_data(df):
    """Divide os dados em treino, validação e teste."""

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].astype(int)

    # Primeiro, separa 20% para o teste final.
    X_train_validation, X_test, y_train_validation, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Dos 80% restantes, separa 25% para validação.
    # Resultado aproximado: 60% treino, 20% validação e 20% teste.
    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train_validation,
        )
    )

    print("\n" + "=" * 60)
    print("SEPARAÇÃO DOS DADOS")
    print("=" * 60)

    print(f"Treino: {len(X_train)} registros")
    print(f"Validação: {len(X_validation)} registros")
    print(f"Teste: {len(X_test)} registros")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# 5. CONSTRUÇÃO E TREINAMENTO DOS MODELOS
# ============================================================

def build_models():
    """Cria os modelos e seus pipelines."""

    logistic_regression = Pipeline([
        ("scaler", StandardScaler()),
        (
            "modelo",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    random_forest = Pipeline([
        (
            "modelo",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight=None,
            ),
        ),
    ])

    return {
        "Regressão Logística": logistic_regression,
        "Random Forest": random_forest,
    }


def train_and_select_model(
    models,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Treina os modelos e seleciona pelo recall da classe maligna."""

    print("\n" + "=" * 60)
    print("TREINAMENTO E COMPARAÇÃO DOS MODELOS")
    print("=" * 60)

    validation_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        malignant_recall = recall_score(
            y_validation,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        validation_results[name] = malignant_recall

        print(
            f"{name} — Recall da classe maligna "
            f"na validação: {malignant_recall:.4f}"
        )

    best_model_name = max(
        validation_results,
        key=lambda name: validation_results[name],
    )

    best_model = models[best_model_name]

    print(f"\nModelo selecionado: {best_model_name}")
    print(
        "Critério de seleção: maior recall da classe maligna "
        "no conjunto de validação."
    )

    return best_model_name, best_model, validation_results


# ============================================================
# 6. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Calcula as métricas finais no conjunto de teste."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("AVALIAÇÃO FINAL — CONJUNTO DE TESTE")
    print("=" * 60)

    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (maligno): {precision:.4f}")
    print(f"Recall (maligno): {recall:.4f}")
    print(f"F1-score (maligno): {f1:.4f}")

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benigno", "Maligno"],
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print("Matriz de confusão:")
    print(matrix)

    # Salva a matriz de confusão.
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benigno", "Maligno"],
        yticklabels=["Benigno", "Maligno"],
    )

    plt.title("Matriz de confusão — conjunto de teste")
    plt.xlabel("Diagnóstico previsto")
    plt.ylabel("Diagnóstico real")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "matriz_confusao.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return predictions


# ============================================================
# 7. IMPORTÂNCIA DAS VARIÁVEIS — RANDOM FOREST
# ============================================================

def analyze_feature_importance(model, feature_names):
    """
    Gera a importância das variáveis para o Random Forest.

    Esta análise é específica do Random Forest. Ela não deve ser
    interpretada como explicação direta da Regressão Logística.
    """

    if not isinstance(model, Pipeline):
        print(
            "\nImportância das variáveis não gerada: "
            "modelo não é um Pipeline."
        )
        return

    classifier = model.named_steps.get("modelo")

    if not isinstance(classifier, RandomForestClassifier):
        print(
            "\nImportância das variáveis não gerada: "
            "o modelo selecionado não é Random Forest."
        )
        print(
            "A importância por impureza utilizada nesta seção "
            "é específica para modelos baseados em árvores."
        )
        return

    importances = pd.Series(
        classifier.feature_importances_,
        index=feature_names,
        name="Importância",
    ).sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("IMPORTÂNCIA DAS VARIÁVEIS — RANDOM FOREST")
    print("=" * 60)

    print(importances.head(10).round(6))

    top_features = importances.head(10).sort_values()

    plt.figure(figsize=(10, 7))

    top_features.plot(
        kind="barh",
        title="Top 10 — importância das variáveis (Random Forest)",
    )

    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    importances.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        header=True,
        index_label="Variável",
    )

    print(
        "\nGráfico salvo em: "
        f"{OUTPUT_DIR / 'feature_importance.png'}"
    )



# ============================================================
# 8. EXPLICABILIDADE DO MODELO — SHAP
# ============================================================

def analyze_with_shap(model, X_train, X_test):
    """
    Gera explicações globais e individuais para o modelo selecionado.

    A explicação utiliza o modelo já treinado. O conjunto de treino
    fornece os dados de referência, e uma observação do teste é
    utilizada para a explicação individual.
    """

    print("\n" + "=" * 60)
    print("EXPLICABILIDADE DO MODELO — SHAP")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classifier = model.named_steps.get("modelo")

    if isinstance(classifier, LogisticRegression):
        scaler = model.named_steps.get("scaler")

        if scaler is None:
            raise ValueError(
                "O pipeline da Regressão Logística não possui scaler."
            )

        X_train_transformed = scaler.transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        background = shap.sample(
            X_train_transformed,
            min(100, len(X_train_transformed)),
            random_state=RANDOM_STATE,
        )

        explainer = shap.LinearExplainer(classifier, background)

        raw_values = explainer(X_test_transformed)

        # Os valores SHAP representam contribuições para a saída
        # do classificador, em escala de log-odds.
        shap_values = raw_values.values

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        base_values = raw_values.base_values

        if getattr(base_values, "ndim", 0) == 2:
            base_values = base_values[:, 1]

        elif getattr(base_values, "ndim", 0) == 0:
            base_values = float(base_values)

        feature_values = X_test.to_numpy()

    elif isinstance(classifier, RandomForestClassifier):
        explainer = shap.TreeExplainer(classifier)
        raw_values = explainer(X_test)

        shap_values = raw_values.values

        # Em classificadores binários, seleciona a classe maligna (1).
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        base_values = raw_values.base_values

        if getattr(base_values, "ndim", 0) == 2:
            base_values = base_values[:, 1]

        elif getattr(base_values, "ndim", 0) == 0:
            base_values = float(base_values)

        feature_values = X_test.to_numpy()

    else:
        raise TypeError(
            "Tipo de modelo não suportado para SHAP: "
            f"{type(classifier).__name__}"
        )

    # Normaliza os valores-base para a explicação individual.
    if getattr(base_values, "ndim", 0) == 1:
        individual_base_value = base_values[0]
    else:
        individual_base_value = base_values

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=feature_values,
        feature_names=list(X_test.columns),
    )

    # Explicação global: importância média absoluta de cada variável.
    plt.figure(figsize=(11, 8))

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=list(X_test.columns),
        max_display=15,
        show=False,
    )

    plt.title("SHAP — importância global das variáveis")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Explicação individual: primeira observação do conjunto de teste.
    individual_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=individual_base_value,
        data=feature_values[0],
        feature_names=list(X_test.columns),
    )

    shap.plots.waterfall(
        individual_explanation,
        max_display=12,
        show=False,
    )

    plt.gcf().set_size_inches(11, 8)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_individual.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Exporta os valores SHAP para análise complementar.
    shap_importance = pd.Series(
        abs(shap_values).mean(axis=0),
        index=X_test.columns,
        name="SHAP médio absoluto",
    ).sort_values(ascending=False)

    shap_importance.to_csv(
        OUTPUT_DIR / "shap_importance.csv",
        header=True,
        index_label="Variável",
    )

    print("\nTop 10 variáveis por SHAP médio absoluto:")
    print(shap_importance.head(10).round(6))

    print(f"\nGráfico global: {OUTPUT_DIR / 'shap_summary.png'}")
    print(f"Gráfico individual: {OUTPUT_DIR / 'shap_individual.png'}")
    print(f"Valores SHAP: {OUTPUT_DIR / 'shap_importance.csv'}")


# ============================================================
# 8. EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    """Executa o fluxo completo do projeto."""

    df = load_data()

    perform_eda(df)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(df)

    models = build_models()

    best_model_name, best_model, validation_results = (
        train_and_select_model(
            models,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
    )

    evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    # A análise é gerada apenas se o modelo selecionado for Random Forest.
    analyze_feature_importance(
        models["Random Forest"],
        X_train.columns,
    )

    # SHAP: explica o modelo que foi efetivamente selecionado.
    analyze_with_shap(
        best_model,
        X_train,
        X_test,
    )

    print("\n" + "=" * 60)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"Modelo selecionado: {best_model_name}")
    print(f"Resultados e gráficos: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()