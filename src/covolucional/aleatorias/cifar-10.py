# import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import (BatchNormalization, 
                          Dense, 
                          Dropout, 
                          Flatten, 
                          Conv2D, 
                          MaxPooling2D)
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# Carregamento da base de dados (na primeira execução será feito o download)
(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

# Mostra a imagem e a respectiva classe, de acordo com o índice passado como parâmetro
# Você pode testar os seguintes índices para visualizar uma imagem de cada classe
# Avião - 650
# Pássaro - 6
# Gato - 9
# Veado - 3
# Cachorro - 813
# Sapo - 651
# Cavalo - 652
# Barco - 811
# Caminhão - 970
# Automóvel - 4
# plt.imshow(X_treinamento[4])
# plt.title('Classe '+ str(y_treinamento[4]))

# As dimensões dessas imagens é 32x32 e o número de canails é 3 pois vamos utilizar as imagens coloridas
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)

# Conversão para float para podermos aplicar a normalização
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# Normalização para os dados ficarem na escala entre 0 e 1 e agilizar o processamento
previsores_treinamento /= 255
previsores_teste /= 255

# Criação de variáveis do tipo dummy, pois teremos 10 saídas
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

# Criação da rede neural com duas camadas de convolução
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation = 'selu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3, 3), activation = 'selu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

# Rede neural densa com duas camadas ocultas
classificador.add(Dense(units = 190, activation = 'selu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 190, activation = 'selu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', 
                      optimizer="adam", metrics=['accuracy'])

# Augumentation
gerador_treinamento = ImageDataGenerator(
    rotation_range=10,  # indica o grau que será rotacionado a imagem
    horizontal_flip=True,  # conseguimos das giros horizontais na imagm
    shear_range=0.3,  # vai mudar os pixels para outra direção, vai mudar os valores do pixel
    height_shift_range=0.09,  # responsável na faixa de altura da imagem
    zoom_range= 0.3  # aplica o zoom na imagem
)

# na base de teste não se faz nenhuma alteralção
gerador_teste = ImageDataGenerator()  # passando assim ele não faz nenhuma transformação

# gerando as novas bases de treinamento e de teste
base_treinamento = gerador_treinamento.flow(
    previsores_treinamento,
    classe_treinamento,
    batch_size=190
)

base_teste = gerador_teste.flow(
    previsores_teste,
    classe_teste,
    batch_size=190
)

# steps_per_epoch: quantidade de registros dividido pelo batch size
# validation_steps : quantidade de registros na base teste dividos pelo batch size
# treinando o modelo
classificador.fit_generator(
    base_treinamento,
    steps_per_epoch= 50000 / 190, # numero total de etapas de amostras a serem geradas pelo gerador antes de declarar uma época concluida e iniciar a próxima época
    epochs=10,
    validation_data= base_teste,
    validation_steps= 10000 / 190 
)

