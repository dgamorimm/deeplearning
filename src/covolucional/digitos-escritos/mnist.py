import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from rich import print

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

def image_view():
    plt.axis('off')
    plt.show()

# imprtando, lendo e separando os conjuntos de dados
(X_treinamento, y_treinamento),\
(X_teste, y_teste) = mnist.load_data()

shape_view('X - Treinamento',X_treinamento)
shape_view('X - Teste',X_teste)
shape_view('y - Treinamento',y_treinamento)
shape_view('y - Teste',y_teste)

# tem que verificar se a cor é importante para o seu resultado
# porque a cor ela aumenta a dimensionalida, ou seja, tem um maior processamento
# como são só numero não preciamos de cores
plt.imshow(X_treinamento[5], cmap='gray')
# image_view()

# mudando o conjunto de dados para que o tensorflow consiga entender
# o keras roda em cima do do tensor flow
previsores_treinamento = X_treinamento.reshape(
    X_treinamento.shape[0], # quantidade de registros
    28, # altura
    28, # largura
    1 # canal, quantos canais vamos utilizar, nesse caso como vamos usar somente na cor cinza colocamos 1. Se fosse colorido seria 3 canais devido ao RGB
)
# temos que alterar o tipo, pois esta como inteiro
previsores_treinamento = previsores_treinamento.astype('float32')
shape_view('X - Treino Alter',previsores_treinamento)

previsores_teste = X_teste.reshape(
    X_teste.shape[0],
    28,
    28,
    1
)
previsores_teste = previsores_teste.astype('float32')
shape_view('X - Teste Alter',previsores_treinamento)

""" 
precimos criar uma escala de numeros, pois se tiverem numeros muito altos
o algoritmo não vai demorar muito o processamento
poara ajustar usamos a tecninca de min/max normalization
escala de 0 até 1

exemplo da base

desnormalizado | normalizado
---------------|--------------
    0               0.
    0               0.
    200             0.358
    216             0.58
    215             0.6445
    125             0.48972
    136             0.31116
    124             0.4821
    0               0.
    0               0.
    0               0.
    0               0.
"""
# cada um desses valores ocupa 1 Byte, e 1 Byte vai de 0 a 255
# São as configurações do RGB
# foi por isso que configuramos a a coluna de registros para o tipo float, para adicionar os valores da divisão que são em casa decimais
previsores_treinamento /= 255
previsores_teste /= 255

# mesmo processo do OneHotEncoder 
# os numeros vão de 0 - 9, portanto tem 10 posições
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)