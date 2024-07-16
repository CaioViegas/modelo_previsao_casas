import pandas as pd

def dataset_informacao(dataset):
    """
    Esta função fornece uma análise abrangente de um conjunto de dados fornecido. Ela imprime as primeiras linhas, informações das colunas, resumo estatístico, valores nulos e o número de valores discrepantes (outliers) em cada coluna numérica.

    Parâmetros:
    - dataset (pandas.DataFrame): O conjunto de dados a ser analisado.

    Retorna:
    Nenhum. A função imprime os resultados da análise diretamente.
    """
    print("Primeiras linhas do dataset:")
    print(dataset.head())

    print("\nInformações das colunas do dataset:")
    print(dataset.info())

    print("\nResumo estatístico das colunas:")
    print(dataset.describe())

    print("\nValores nulos:")
    print(dataset.isnull().sum())

    colunas_num = [col for col in dataset.columns if dataset[col].dtype in ['int64', 'float64']]
    
    print("\nQuantidade de outliers em cada coluna numérica:")
    for col in colunas_num:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = dataset[(dataset[col] < (Q1 - 1.5 * IQR)) | (dataset[col] > (Q3 + 1.5 * IQR))]
        print(f"{col}: {outliers.shape[0]}")

if __name__ == '__main__':
    df = pd.read_csv("data/House Price India.csv")
    dataset_informacao(df)