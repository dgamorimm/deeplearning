from keras.models import Sequential
from keras.layers import (BatchNormalization, 
                          Dense, 
                          Dropout, 
                          Flatten, 
                          Conv2D, 
                          MaxPooling2D)
from keras.preprocessing.image import ImageDataGenerator
import os

PATH_IMAGES = os.path.join(os.path.dirname(__file__),'images')
PATH_TEST = os.path.join(PATH_IMAGES, 'test_set')
PATH_TRAINING = os.path.join(PATH_IMAGES, 'training_set')

PATH_TEST_DOG = os.path.join(PATH_TEST, 'cachorro')
PATH_TEST_CAT = os.path.join(PATH_TEST, 'gato')

PATH_TRAINING_DOG = os.path.join(PATH_TRAINING, 'cachorro')
PATH_TRAINING_CAT = os.path.join(PATH_TRAINING, 'gato')

# Estrutura da rede neural

classificador = Sequential()

# recomendado começar com 64 filtros
# Temos dimensões diferentes nas fotos, quando setamos a largura e altura, ele faz um redimensionamento automatico
# 3 canais : RGB
# recomendasse valores maiores das imagens que você vai carregar
classificador.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        input_shape = (64, 64, 3),
        activation='relu'
    )
)
classificador.add(
    BatchNormalization()
)
classificador.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)
classificador.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        input_shape = (64, 64, 3),
        activation='relu'
    )
)
classificador.add(
    BatchNormalization()
)
classificador.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

classificador.add(Flatten())

# criando a rede neural densa com duas camadas
classificador.add(
    Dense(
        units=128,
        activation='relu'
    )
)
classificador.add(
    Dropout(
        rate=0.2
    )
)

classificador.add(
    Dense(
        units=128,
        activation='relu'
    )
)
classificador.add(
    Dropout(
        rate=0.2
    )
)

# criando a camada de saida
classificador.add(
    Dense(
        units=1,
        activation='sigmoid'
    )
)

# avaliando o nosso modelo
classificador.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)