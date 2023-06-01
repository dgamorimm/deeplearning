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

def shape_view(dataframe):
    shape = dataframe.shape
    records = shape[0]
    columns = shape[1]
    print(f'Registros: {records} || Colunas: {columns}')

base = drop_columns(['dateCrawled', 'dateCreated', 'nrOfPictures', 
                     'postalCode', 'lastSeen', 'name', 'seller', 'offerType'])

### vimos que há preços de carros com valor 10 e até 0
### isso atrabalha a base
inconsistencia1 = base.loc[base.price <= 10]
shape_view(inconsistencia1)
### Essa é a correção
shape_view(base)
base = base[base.price > 10]
shape_view(base)

### vimos que há preços de carros com valor muito elevado, o que não condiz com a realidade
### isso atrabalha a base
inconsistencia2 = base.loc[base.price > 350000]
shape_view(inconsistencia2)
### Essa é a correção
shape_view(base)
base = base[base.price < 350000]
shape_view(base)