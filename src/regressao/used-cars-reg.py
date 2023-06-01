import os
import pandas as pd
from rich import print

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

base = pd.read_csv(PATH_DATA + '\\autos.csv', encoding='ISO-8859-1')

# quando você coloca axis = 1, quer dizer que você quer apagar a coluna inteira
# vamos apagar os dados que não tem relação direta com a precificação do carro
# o atributo name tem muita pluraridade nos nomes, e isso pode dificultar o modelo a encontrar um padrão
# o atributo seller e offerType esta desbalanceado , tem mais privat do que gewerblich , também foi removido
# para avaliar o desbalanceamento e os valores, podemos usar o atributo value_counts

## Pré-Processamento de dados
def drop_columns(columns : list):
    for column in columns:
        base.drop(column, axis=1, inplace=True)
    return base

def column_value_view(column : str):
    print(base[column].value_counts())

base = drop_columns(['dateCrawled', 'dateCreated', 'nrOfPictures', 
                     'postalCode', 'lastSeen', 'name', 'seller', 'offerType'])