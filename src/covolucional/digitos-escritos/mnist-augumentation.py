from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


# Pré Processamento
(X_treinamento, y_treinamento),\
(X_teste, y_teste) = mnist.load_data()

previsores_treinamento = X_treinamento.reshape(
    X_treinamento.shape[0],
    28,
    28,
    1
)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_treinamento /= 255

previsores_teste = X_teste.reshape(
    X_teste.shape[0],
    28,
    28,
    1
)
previsores_teste = previsores_teste.astype('float32')
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

# Criando a rede convulucional
classificador = Sequential()
classificador.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        input_shape=(28,28,1),
        activation='relu'
    )
)
classificador.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)
classificador.add(
    Flatten()
)

# Criando a rede neural densa
classificador.add(
    Dense(
        units=128,
        activation='relu'
    )
)

# Criando a camada de saída
classificador.add(
    Dense(
        units=10,
        activation='softmax'
    )
)

classificador.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Augumentation
gerador_treinamento = ImageDataGenerator(
    rotation_range=7,  # indica o grau que será rotacionado a imagem
    horizontal_flip=True,  # conseguimos das giros horizontais na imagm
    shear_range=0.2,  # vai mudar os pixels para outra direção, vai mudar os valores do pixel
    height_shift_range=0.07,  # responsável na faixa de altura da imagem
    zoom_range= 0.2  # aplica o zoom na imagem
)

# na base de teste não se faz nenhuma alteralção
gerador_teste = ImageDataGenerator()  # passando assim ele não faz nenhuma transformação

# gerando as novas bases de treinamento e de teste
base_treinamento = gerador_treinamento.flow(
    previsores_treinamento,
    classe_treinamento,
    batch_size=128
)

base_teste = gerador_teste.flow(
    previsores_teste,
    classe_teste,
    batch_size=128
)

# steps_per_epoch: quantidade de registros dividido pelo batch size
# validation_steps : quantidade de registros na base teste dividos pelo batch size
# treinando o modelo
classificador.fit_generator(
    base_treinamento,
    steps_per_epoch= 60000 / 128, # numero total de etapas de amostras a serem geradas pelo gerador antes de declarar uma época concluida e iniciar a próxima época
    epochs=5,
    validation_data= base_teste,
    validation_steps= 1000 / 128 
)