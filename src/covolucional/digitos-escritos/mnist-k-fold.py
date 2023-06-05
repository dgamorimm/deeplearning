import matplotlib.pyplot as plt
import numpy as np
import statistics
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
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

def calculate_neurons(total_pixels: int, 
                      kernel_size : tuple,
                      pooling_size : tuple):
    mp_charac = (total_pixels - kernel_size[0] + 1)
    qty = ((mp_charac // pooling_size[0]) ** 2) // 1000
    return qty

# setando um valor para ter a mesma saida com os dados
seed = 5
np.random.seed(seed)

# imprtando, lendo e separando os conjuntos de dados
(X, y),\
(X_teste, y_teste) = mnist.load_data()

shape_view('X - Treinamento',X)
shape_view('X - Teste',X_teste)
shape_view('y - Treinamento',y)
shape_view('y - Teste',y_teste)

# mesmo processo do OneHotEncoder 
# os numeros vão de 0 - 9, portanto tem 10 posições
classe = np_utils.to_categorical(y, 10)

# mudando o conjunto de dados para que o tensorflow consiga entender
# o keras roda em cima do do tensor flow
previsores = X.reshape(
    X.shape[0], # quantidade de registros
    28, # altura
    28, # largura
    1 # canal, quantos canais vamos utilizar, nesse caso como vamos usar somente na cor cinza colocamos 1. Se fosse colorido seria 3 canais devido ao RGB
)
# temos que alterar o tipo, pois esta como inteiro
previsores = previsores.astype('float32')
shape_view('Previsores',previsores)


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
previsores /= 255

# numero de vezes que ele vai percorrer os dados
# shuffle como true é para embaralhar os dados
# random state é para ter a mesma saida
# fazendo a avaliação cruzada
kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)
resultados = []

for idx_treinamento, idx_teste in kfold.split(
    previsores, np.zeros(
        shape= (classe.shape[0],1)
    )
):
    print('Índices treinamento: ', idx_treinamento, ' Índice teste: ', idx_teste)

    # Criando a estrutura da rede neural
    classificador = Sequential()

    # o ideal é começar sempre com 64 kernels
    # se for aumentar é bom seguir seus multiplos
    # strides : é o parametro que você configura para setar como que a sua janela se move dentro do seu mapar de caracteristicas
    # strides (1,1) significa que você vai mover um pixel para direita e um pixel para baixo
    kernel_size = (3,3)
    classificador.add(
        Conv2D(
            filters=64, # mapas de caracteristicas(altera os valores dos kernels)
            kernel_size=kernel_size, # tamanho do meu detector de caracterisicas. Se tiver uma imagem maior, isso aqui tem que aumentar
            input_shape=(28,28,1), # largura, comprimento, canal
            activation='relu' # função de ativação
        )
    )

    # ele tem a mesma ideia que fizemos de normalização para a camada de entrada = previsores_teste /= 255
    # só que aqui vamos aplicar para as demais camadas
    classificador.add(BatchNormalization())

    # pool_size (2,2) significa que você vai mover 2 pixels para direita e 2 pixels para baixo
    # etapa de pooling
    pool_size = (2,2)
    classificador.add(
        MaxPooling2D(
            pool_size=pool_size # tamanho da minha matriz/janela que vai capturar dentro da matriz de carcteristicas o meu maior valor
        )
    )

    # quando temos mais de uma camada, usamos o flatten apenas uma unica vez
    # etapa flattening
    classificador.add(Flatten())

    # não é muito comum usar formula de units aqui
    # geralmente usam 128, 512 e etc
    # gerar a rede neural densa 1
    classificador.add(
        Dense(
            units=calculate_neurons(784, kernel_size, pool_size),
            activation='relu'
        )
    )

    classificador.add(
        Dropout(
            0.2
        )
    )

    # 10 saidas com numeros de 0 a 9
    # utiliza o softmax pois tem mais de um classe
    # camada de saída
    classificador.add(
        Dense(
            units=10,
            activation='softmax'
        )
    )

    # compilando e avaliando o nosso modelo
    classificador.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # validation_data: para cada época que ele realizou o trenimaneto ele ja vai mostrando os resultados
    # loss = mostra a perda na base de dados de treinamento
    # accuracy = mostra o acerto na base de dados de treinamento
    # val_loss = mostra a perda na base de dados de teste
    # val_accuracy = mostra o acerto na base de dados de teste
    # treinando o modelo
    classificador.fit(
        previsores[idx_treinamento],
        classe[idx_treinamento],
        batch_size=128,
        epochs = 5
    )
    
    precisao = classificador(
        previsores[idx_teste],
        classe[idx_teste],
    )
    
    resultados.append(precisao[1])

print(resultados)
print(statistics.mean(resultados))