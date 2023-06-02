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
    

def calculate_neurons(total_pixels: int, 
                      kernel_size : tuple,
                      pooling_size : tuple):
    mp_charac = (total_pixels - kernel_size[0] + 1)
    qty = ((mp_charac // pooling_size[0]) ** 2) // 1000
    return qty

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

# pool_size (2,2) significa que você vai mover 2 pixels para direita e 2 pixels para baixo
# etapa de pooling
pool_size = (2,2)
classificador.add(
    MaxPooling2D(
        pool_size=pool_size # tamanho da minha matriz/janela que vai capturar dentro da matriz de carcteristicas o meu maior valor
    )
)

# etapa flattening
classificador.add(Flatten())

# não é muito comum usar formula de units aqui
# geralmente usam 128, 512 e etc
# gerar a rede neural densa
classificador.add(
    Dense(
        units=calculate_neurons(784, kernel_size, pool_size),
        activation='relu'
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
    previsores_treinamento,
    classe_treinamento,
    batch_size=128,
    epochs = 5,
    validation_data=(previsores_teste, classe_teste)
)