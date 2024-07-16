import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import IsolationForest

def processamento_dados(dataset, target, test_size=0.2, random_state=101, pca_components=20, k_best=20):
    """
    Esta função processa o conjunto de dados de entrada realizando várias etapas de pré-processamento, incluindo escalonamento, transformação PCA e seleção de características.

    Parâmetros:
    - dataset (pandas.DataFrame): O conjunto de dados de entrada contendo as características e a variável alvo.
    - target (str): O nome da variável alvo no conjunto de dados.
    - test_size (float, opcional): A proporção do conjunto de dados a ser usada como conjunto de teste. O padrão é 0,2.
    - random_state (int, opcional): A semente para o gerador de números aleatórios. O padrão é 101.
    - pca_components (int, opcional): O número de componentes principais a serem retidos após a transformação PCA. O padrão é 21.
    - k_best (int, opcional): O número de melhores características a serem selecionadas usando SelectKBest. O padrão é 21.

    Retorna:
    - X_train_selecionado (numpy array): As características selecionadas para o conjunto de treinamento após escalonamento, transformação PCA e seleção de características.
    - X_test_selecionado (numpy array): As características selecionadas para o conjunto de teste após escalonamento, transformação PCA e seleção de características.
    - y_train (numpy array): A variável alvo para o conjunto de treinamento.
    - y_test (numpy array): A variável alvo para o conjunto de teste.
    """
    X = dataset.drop(columns=target)
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    seletor = SelectKBest(score_func=f_classif, k=k_best)
    X_train_selecionado = seletor.fit_transform(X_train_pca, y_train)
    X_test_selecionado = seletor.transform(X_test_pca)

    return X_train_selecionado, X_test_selecionado, y_train, y_test

if __name__ == '__main__':
    df = pd.read_csv("data/House Price India.csv")
    processamento_dados(df, "Price")