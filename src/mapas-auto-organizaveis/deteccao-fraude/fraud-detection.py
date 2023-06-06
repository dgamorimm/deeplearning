from minisom import MiniSom
from pylab import pcolor, colorbar, plot
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import numpy as np

# utils
def shape_view(title : str, dataframe):
    shape = dataframe.shape
    c = 17
    c2 = 7
    c3 = 5
    try:
        records = shape[0]
        pixel_h = shape[1]
        pixel_v = shape[2]
        channel = shape[3]
        total = pixel_h * pixel_v
        print(f'Titulo: {title}{(c-len(title))*" "}|| Registros: {records}{(c2-len(str(records)))*" "}|| Pixels: {pixel_h} X {pixel_h} = {total}{(c3-len(str(total)))*" "}|| Canais: {channel}')
    except:
        try:
            records = shape[0]
            pixel_h = shape[1]
            pixel_v = shape[2]
            total = pixel_h * pixel_v
            print(f'Titulo: {title}{(c-len(title))*" "}|| Registros: {records}{(c2-len(str(records)))*" "}|| Pixels: {pixel_h} X {pixel_h} = {total}')
        except:
            records = shape[0]
            try:
                columns = shape[1]
            except:
                columns = 0
            print(f'Titulo: {title}{(c-len(title))*" "}|| Registros: {records}{(c2-len(str(records)))*" "}|| Colunas: {columns}')
            
def column_null_view(column : str):
    shape_view(f'Coluna Nula - {column}',base.loc[pd.isnull(base[column])])
    
def drop_columns(columns : list):
    for column in columns:
        base.drop(column, axis=1, inplace=True)
    return base

def column_value_view(column : str):
    return base[column].value_counts()

def graphic_view():
    plt.axis('off')
    plt.show()

def drop_lines_na():
    # axis = 0 ele apaga a linha 
    base.dropna(axis=0, inplace=True)

def calculate_neurons(qty_records):
    square = math.sqrt(qty_records)
    value = 5 * square
    matrix_value = math.ceil(math.sqrt(value))
    return (matrix_value, matrix_value)

def qty_atributes(data):
    try:
        return data.shape[1] 
    except:
        return len(data[0])

def qty_records(data):
    try:
        return data.shape[0]
    except:
        return len(data)

# lendo os dados
PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

# Carregamento da base de dados
base = pd.read_csv(PATH_DATA + '\\credit_data.csv')
shape_view('Base Inicial', base)

# Pré processamento
drop_lines_na()
shape_view('Nulos Removidos', base)

## preenchendo idades negativas com a média das idades que não são negativas
base.loc[base.age < 0, 'age'] = base.loc[base.age > 0, 'age'].mean()
shape_view('Idades Negativas',base.loc[base.age < 0, 'age'])

## separando os dados
X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

## normalizando os dados entre 0 e 1
normalizador = MinMaxScaler(feature_range=(0, 1))
X = normalizador.fit_transform(X)

# construindo o mapa organizavel
x_matrix, y_matrix = calculate_neurons(qty_records(X))
som = MiniSom(
    x=x_matrix,
    y=y_matrix,
    input_len=qty_atributes(X),
    random_seed=0
)

# inicializar os pesos
som.random_weights_init(X)

# realizar o treinamento
som.train_random(
    data= X,
    num_iteration=100
)

# Se o gráfico estiver com um ponto mais para o amarelo
# Quer dizer que eles são muito diferentes dos outros, ou seja, são outliers
# para o nosso exemplo, indica que eles possuem uma probabilidade alta de fraude
# visualizando o mapa
pcolor(som.distance_map().T)
colorbar()

# criando os marcadores
markers = ['o', 's']

# criando as cores
colors = ['r', 'b']

# criando o mapa auto organizavel
for i, x in enumerate(X):
    w = som.winner(x)
    plot(
        w[0] + 0.5,
        w[1] + 0.5,
        markers[y[i]],
        markerfacecolor='None',
        markersize=10,
        markeredgecolor=colors[y[i]],
        markeredgewidth=2
    )
graphic_view()

# buscando clientes que realmente podem ser fraudulentos
# no mapeamento vai indicar quais registros representam qual neuronio
mapeamento = som.win_map(X)

# pegando os suspeitos
# suspeitos são os que estão na base amarela do mapa
# no mapa as linhas são horizontais e as colunas nas verticais
"""
lista de suspeitos:
LINHA X COLUNA
(5, 5)
(13,7)
"""
suspeitos = np.concatenate((mapeamento[(5, 5)], mapeamento[(13,7)]), axis=0)

# desnormalizando o dado
suspeitos = normalizador.inverse_transform(suspeitos)

# criando os valores da classe
# verificando quem são os usuários e a que classe pertencem e armazenando numa lista
classe = []
for i in range(qty_records(base)):
    for j in range(qty_records(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            classe.append(base.iloc[i, 4])

# transformando a lista em um array do numpy para manipulação
classe = np.asarray(classe)

# juntando os dados da classe com os suspeitos
suspeitos_final = np.column_stack((suspeitos, classe))

# colocando em ordem crescente
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]
print(suspeitos_final)