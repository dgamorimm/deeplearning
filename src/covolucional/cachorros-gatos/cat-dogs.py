from keras.models import Sequential
from keras.layers import (BatchNormalization, 
                          Dense, 
                          Dropout, 
                          Flatten, 
                          Conv2D, 
                          MaxPooling2D)
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
import os
import numpy as np

PATH_IMAGES = os.path.join(os.path.dirname(__file__),'images')
PATH_TEST = os.path.join(PATH_IMAGES, 'test_set')
PATH_TRAINING = os.path.join(PATH_IMAGES, 'training_set')

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

# realizando o processo de augumentation
gerador_treinamento = ImageDataGenerator(
    rescale=1./255,  # realiza a normalização de escala das imagens
    rotation_range= 7,
    horizontal_flip= True,
    shear_range= 0.2,
    height_shift_range= 0.07,
    zoom_range= 0.2
)

gerador_teste = ImageDataGenerator(rescale=1./255)

# criando as novas bases de treinamento e teste
base_treinamento = gerador_treinamento.flow_from_directory(
    PATH_TRAINING,
    target_size = (64, 64),
    batch_size=32,
    class_mode='binary'
)

base_teste = gerador_teste.flow_from_directory(
    PATH_TEST,
    target_size = (64, 64),
    batch_size=32,
    class_mode='binary'
)

# treinando o modelo
# se você tiver um alto indice de capacidade de processamento, o ideal não é dividir pelo batch size
classificador.fit(
    base_treinamento,
    steps_per_epoch= 4000 / 32,
    epochs=5,
    validation_data=base_teste,
    validation_steps= 1000 / 32
)

# realizando a previsão de uma imagem

# carregando a imagem no formato keras
imagem_teste = load_img(
    os.path.join(PATH_TEST, 'gato', 'cat.3500.jpg'),
    target_size = (64,64)
)

# passando as caracteristicas dos pixels da imagem para um array
imagem_teste = img_to_array(imagem_teste)

# aplicando a normalização
imagem_teste /= 255

# apliando as dimensões
# cria uma coluna com uma unica dimensão/registro, pois queremos prever uma unica imagem
imagem_teste = np.expand_dims(imagem_teste, axis=0)

#realizando a previsao
previsao = classificador.predict(imagem_teste)
print(previsao)
previsao = (previsao > 0.5)
if previsao[0][0] == True:
    print('É um gato')
else:
    print('É um cachorro')
# quanto mais o valor se aproxima do 1, o resultado será para a classe 1 .
# quanto mais o valor se aproxima do 0, o resultado será para a classe 0 .
# para verificar qual é os indices das classes,  imprima
print(base_treinamento.class_indices)