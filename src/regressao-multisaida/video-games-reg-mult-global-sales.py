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
colunas = ['Other_Sales', 'Developer', 'Name',
           'NA_Sales', 'EU_Sales', 'JP_Sales']
base = drop_columns(colunas)
shape_view('Removendo colunas', base)

# o interessante aqui seria preencher com dados reais
# caso não consiga, preencher com alguma média, mediana e por ai vai
## removendo registros que não tem valor
drop_lines_na()
shape_view('Removendo linhas NaN', base)

## removendo outras inconsistencias com relação as vendas nos paises
base = base.loc[base['Global_Sales'] > 1]
shape_view('Removendo vendas menores que 1', base)

# sparando a base entre previsores e classes
previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
valor_vendas = base.iloc[:, 4].values

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

# A camada de entrada possui 99 neurônios na entrada, pois equivale a
# quantidade de atributos previsores após o pré-processamento
# A quantidade 50 é relativo a fórumula: (entradas (99) + saída (1)) / 2
# Definida somente uma camada de entrada pois estamos trabalhando somente com o valor total
## estrutura da rede neural
ativacao = Activation(activation = 'selu')

camada_entrada = Input(
    shape=99,
)

# 61 + numeros de saidas 3 / 2 = 32
camada_oculta1 = Dense(
    units=50,
    activation=ativacao
)(camada_entrada)
camada_oculta1 = Dropout(
    0.005
)(camada_oculta1)

camada_oculta2 = Dense(
    units=50,
    activation=ativacao
)(camada_oculta1)
camada_oculta2 = Dropout(
    0.002
)(camada_oculta2)


camada_saida = Dense(
    units=1,
    activation='linear'
)(camada_oculta2)


regressor = Model(
    inputs = camada_entrada, 
    outputs=[camada_saida]
)

# mse : mean squared error
regressor.compile(
    optimizer='adam',
    loss='mse',
)

regressor.fit(
    previsores,
    valor_vendas,
    epochs=4800,
    batch_size=50
)

# realizando as previsões para ver se ele acertou
previsoes = regressor.predict(previsores)

print(previsoes[:5])
print(valor_vendas[:5])