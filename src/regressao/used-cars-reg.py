import os
import pandas as pd
from rich import print
from sklearn.preprocessing import LabelEncoder

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
    return base[column].value_counts()

def shape_view(title : str, dataframe):
    shape = dataframe.shape
    records = shape[0]
    columns = shape[1]
    print(f'Titulo: {title} || Registros: {records} || Colunas: {columns}')

def column_null_view(column : str):
    shape_view(f'Coluna Nula - {column}',base.loc[pd.isnull(base[column])])

def fill_na_more_appears(column : str):
    more_appears = column_value_view(column).head(1).index[0]
    base.fillna(value={column : more_appears}, inplace=True)

base = drop_columns(['dateCrawled', 'dateCreated', 'nrOfPictures', 
                     'postalCode', 'lastSeen', 'name', 'seller', 'offerType'])

### vimos que há preços de carros com valor 10 e até 0
### isso atrabalha a base
inconsistencia1 = base.loc[base.price <= 10]
shape_view('Inconsistencia 1', inconsistencia1)
### Essa é a correção
shape_view('Base Antes I1', base)
base = base[base.price > 10]
shape_view('Base Depois I1', base)

### vimos que há preços de carros com valor muito elevado, o que não condiz com a realidade
### isso atrabalha a base
inconsistencia2 = base.loc[base.price > 350000]
shape_view('Inconsistencia 2', inconsistencia2)
### Essa é a correção
shape_view('Base Antes I2',base)
base = base[base.price < 350000]
shape_view('Base Depois I2',base)

# essa outra inconsitencia mostra que temos dados ainda não preenchidos
# porém ela é de suma importancia pois da caracteristicas do carro
# aqui conseguimos ver a quantidade de valores nulos por coluna
colunas = [
    'vehicleType', 'gearbox', 'model', 'fuelType', 'notRepairedDamage'
]
for coluna in colunas:
    column_null_view(coluna)
# para corrigir essa inconsistencia, vamos subistituir os valores
# com que mais aparece na base
print('-------tratando a inconsistencia 3---------')
for coluna in colunas:
    fill_na_more_appears(coluna)
    column_null_view(coluna)
    

# Separando a base
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

# transformando dados categoricos em valores numericos
le = LabelEncoder()

colunas = [0, 1, 3, 5, 8, 9, 10]
for idx in colunas:
    previsores[:, idx] = le.fit_transform(previsores[:, idx])

print(previsores[0])