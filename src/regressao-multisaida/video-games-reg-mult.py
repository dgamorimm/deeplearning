import pandas as pd
import os
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from rich import print

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

# lendo os dados
base = pd.read_csv(PATH_DATA + '\\games.csv')

## Pré-Processamento de dados
def drop_columns(columns : list):
    for column in columns:
        base.drop(column, axis=1, inplace=True)
    return base

def drop_lines_na():
    # axis = 0 ele apaga a linha 
    base.dropna(axis=0, inplace=True)

def shape_view(title : str, dataframe):
    shape = dataframe.shape
    records = shape[0]
    columns = shape[1]
    print(f'Titulo: {title} || Registros: {records} || Colunas: {columns}')

## removendo colunas desnecessárias
colunas = ['Other_Sales', 'Global_Sales', 'Developer', 'Name']
base = drop_columns(colunas)
shape_view('Removendo colunas', base)

# o interessante aqui seria preencher com dados reais
# caso não consiga, preencher com alguma média, mediana e por ai vai
## removendo registros que não tem valor
drop_lines_na()
shape_view('Removendo linhas NaN', base)

## removendo outras inconsistencias com relação as vendas nos paises
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]
shape_view('Removendo vendas menores que 1', base)

# sparando a base entre previsores e classes
previsores = base.iloc[:, [0, 1, 2, 3, 7, 8 , 9, 10, 11]].values

venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

# transformar atributos categoricos para numeros
print(previsores[0])
le = LabelEncoder()
colunas_categoricas = [0, 2, 3, 8]
for idx in colunas_categoricas:
    previsores[:, idx] = le.fit_transform(previsores[:, idx])

print(previsores[0])

onehotencoder = ColumnTransformer(
    transformers=[
        ("OneHot", OneHotEncoder(), colunas_categoricas)
        ],
    remainder='passthrough'
)
previsores = onehotencoder.fit_transform(previsores).toarray()
print(len(previsores[0]))