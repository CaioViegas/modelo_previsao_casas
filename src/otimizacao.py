import pandas as pd
import joblib
from preprocessamento import processamento_dados
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBRegressor

def modelo_otimizado(dataset, target):
    """
    Esta função otimiza um modelo de regressão XGBoost usando GridSearchCV. Ela recebe um conjunto de dados e uma variável alvo como entrada, processa os dados, realiza uma busca em grade para encontrar os melhores hiperparâmetros e retorna o melhor modelo.

    Parâmetros:
    - dataset (pandas.DataFrame): O conjunto de dados de entrada contendo as características e a variável alvo.
    - target (str): O nome da variável alvo no conjunto de dados.

    Retorna:
    - best_model (XGBRegressor): O melhor modelo de regressão XGBoost otimizado.

    A função primeiro processa o conjunto de dados de entrada e a variável alvo usando a função `processamento_dados`. Em seguida, define uma grade de hiperparâmetros para o modelo XGBoost e realiza uma busca em grade usando `GridSearchCV`. O melhor modelo é então impresso junto com seus parâmetros, e as métricas de desempenho (MSE, MAE e R2-score) são calculadas e impressas. Finalmente, o melhor modelo é salvo usando `joblib.dump` e retornado.
    """
    xgb = XGBRegressor()

    X_train, X_test, y_train, y_test = processamento_dados(dataset, target)

    param_grid = {
        'n_estimators': [10000],
        'learning_rate': [0.1],
        'max_depth': [5],
        'min_child_weight': [3],
        'subsample': [0.6],
        'colsample_bytree': [1.0],
        'reg_alpha': [1.0],
        'reg_lambda': [1.5]
    }

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='r2', cv=5, verbose=1, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_ 

    print(f'Melhores parâmetros encontrados:')
    print(best_model.get_params())
    
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(best_model)
    
    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    print(f'Cross-Validation Accuracy: {cross_val_scores.mean():.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R2-score: {r2:.2f}')

    joblib.dump(best_model, 'modelo_escolhido.joblib')

    return best_model

if __name__ == '__main__':
    df = pd.read_csv("data/House Price India.csv")
    modelo_otimizado(df, "Price")