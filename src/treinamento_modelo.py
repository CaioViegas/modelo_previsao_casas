import pandas as pd
from preprocessamento import processamento_dados
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def avaliar_modelo(modelo, X_train, X_test, y_train, y_test):
    """
    Avalia um modelo fornecido nos conjuntos de dados de treinamento e teste.

    Parâmetros:
    - modelo (sklearn.base.BaseEstimator): O modelo de aprendizado de máquina a ser avaliado.
    - X_train (pandas.DataFrame): As características do conjunto de dados de treinamento.
    - X_test (pandas.DataFrame): As características do conjunto de dados de teste.
    - y_train (pandas.Series): A variável alvo do conjunto de dados de treinamento.
    - y_test (pandas.Series): A variável alvo do conjunto de dados de teste.

    Retorna:
    Um dicionário contendo as seguintes métricas:
    - "MeanSquaredError_train": O erro quadrático médio do modelo no conjunto de dados de treinamento.
    - "MeanSquaredError_test": O erro quadrático médio do modelo no conjunto de dados de teste.
    - "MeanAbsoluteError_train": O erro absoluto médio do modelo no conjunto de dados de treinamento.
    - "MeanAbsoluteError_test": O erro absoluto médio do modelo no conjunto de dados de teste.
    - "R2_train": O valor R2 do modelo no conjunto de dados de treinamento.
    - "R2_test": O valor R2 do modelo no conjunto de dados de teste.
    """
    modelo.fit(X_train, y_train)

    train_pred = modelo.predict(X_train)
    test_pred = modelo.predict(X_test)

    mse_train = mean_squared_error(y_train, train_pred)
    mae_train = mean_absolute_error(y_train, train_pred)
    r2_train = r2_score(y_train, train_pred)

    mse_test = mean_squared_error(y_test, test_pred)
    mae_test = mean_absolute_error(y_test, test_pred)
    r2_test = r2_score(y_test, test_pred)

    return {
        "MeanSquaredError_train": mse_train,
        "MeanSquaredError_test": mse_test,
        "MeanAbsoluteError_train": mae_train,
        "MeanAbsoluteError_test": mae_test,
        "R2_train": r2_train,
        "R2_test": r2_test,
    }

def treinar_modelo(dataset, target):
    """
    Treina e avalia um conjunto de modelos de aprendizado de máquina no conjunto de dados e na variável alvo fornecidos.

    Parâmetros:
    - dataset (pandas.DataFrame): O conjunto de dados contendo as características a serem usadas para treinar os modelos.
    - target (str): O nome da variável alvo no conjunto de dados.

    Retorna:
    - best_model (sklearn.base.BaseEstimator): O modelo de aprendizado de máquina de melhor desempenho encontrado entre o conjunto de modelos.

    Esta função primeiro pré-processa o conjunto de dados e o divide em conjuntos de treinamento e teste. Em seguida, treina e avalia um conjunto de modelos de aprendizado de máquina, incluindo Regressão Linear, Regressão por Vetores de Suporte, Gradient Boosting, Random Forest, Árvore de Decisão e XGBoost. O modelo com a maior pontuação R2 no conjunto de teste é retornado como o modelo de melhor desempenho.
    """
    modelos = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "GradientBoosting": GradientBoostingRegressor(),
        "RandomForest": RandomForestRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
        "XGBoost": XGBRegressor()
    }

    X_train, X_test, y_train, y_test = processamento_dados(dataset, target)

    best_model = None
    best_score = 0

    for nome_modelo, modelo in modelos.items():
        resultados = avaliar_modelo(modelo, X_train, X_test, y_train, y_test)
        r2_test = resultados["R2_test"]
        
        print(f'{nome_modelo} - MSE: {resultados["MeanSquaredError_test"]}, MAE: {resultados["MeanAbsoluteError_test"]}, R2: {resultados["R2_test"]}')

        if r2_test > best_score:
            best_score = r2_test
            best_model = modelo

    print(f'\nMelhor Modelo: {best_model.__class__.__name__} com R2-Score: {best_score}')

    return best_model

if __name__ == '__main__':
    df = pd.read_csv("data/House Price India.csv")
    treinar_modelo(df, "Price")