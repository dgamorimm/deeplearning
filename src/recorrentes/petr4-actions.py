from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from rich import print
import numpy as np
import pandas as pd
import os

# lendo os dados
PATH_DATA = os.path.join(os.path.dirname(__file__),'data')
base = pd.read_csv(PATH_DATA + '\\petr4_treinamento.csv')

#utils
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

def drop_lines_na():
    # axis = 0 ele apaga a linha 
    base.dropna(axis=0, inplace=True)

# pré processamento
shape_view('Inicio Base', base)
drop_lines_na()
shape_view('S/ Registros Nulos', base)

## selecionando os dados de treinamento
base_treinamento = base.iloc[:, 1:2].values
print('Valores Reais\n',base_treinamento[:5])

## como tulizando varias camadas, para melhorar o processamento, é bom que não utlizemos os valores reaisn  e sim normalizados
## vamos colocar os valores em uma escala de 0 até 1
normalizador = MinMaxScaler(
    feature_range=(0, 1)
)
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
print('Normalizado\n',base_treinamento_normalizada[:5])