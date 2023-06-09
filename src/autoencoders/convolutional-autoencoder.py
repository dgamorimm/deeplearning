import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import (Dense,
                          Input,
                          Conv2D,
                          MaxPooling2D,
                          UpSampling2D,
                          Flatten,
                          Reshape)
from keras.models import Model, Sequential
from keras import backend as K
from rich import print

# UpSampling faz o processo inverso do MaxPooling

#utils
def graphic_view():
    plt.axis('off')
    plt.show()

@tf.autograph.experimental.do_not_convert
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# carregando as bases de dados
# como ele vai fazer o encode e o decode, não vamos utilizar base de teste, tendo em vista que a mesa entrada será a mesma saída
(previsores_treinamento, _),\
(previsores_teste, _) = mnist.load_data()
# dim treinamento = (60000, 28, 28) = 28 x 28 = 784 Pixels
# dim teste = (10000, 28, 28) = 28 x 28 = 784 Pixels

# adicionando dimensionalidade para a convolução
previsores_treinamento = previsores_treinamento.reshape(
    (
        len(previsores_treinamento), #registros
        28, # largura em pixels
        28, # altura em pixels
        1  # canais (escala em cinza) = 1  escala RGB = 3
    )
)

# adicionando dimensionalidade para a convolução
previsores_teste = previsores_teste.reshape(
    (
        len(previsores_teste), #registros
        28, # largura em pixels
        28, # altura em pixels
        1  # canais (escala em cinza) = 1  escala RGB = 3
    )
)


# normalizando o dado entre 0 e 1, mas poderiamos usar o MinMaxScaler
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

# criando o autoencoder

# Encoder
autoencoder = Sequential()

# camada de entrada e primeira camada convolucional
autoencoder.add(
    Conv2D(
        filters=16,
        kernel_size=(3,3),
        activation='relu',
        input_shape=(28, 28, 1)
    )
)

autoencoder.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

# padding = utlizado para indicar como que a imagem será passado/formato da imagem
# quando coloco o same é o formato original da imagem que colocamos na primeira camada
# segunda camada convolucional
autoencoder.add(
    Conv2D(
        filters=8,
        kernel_size=(3,3),
        activation='relu',
        padding='same' 
    )
)

autoencoder.add(
    MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )
)

# terceira camada convolucional
# informa de quanto em quantos pixels a imagem deve andar. o defaul é (1, 1)
autoencoder.add(
    Conv2D(
        filters=8,
        kernel_size=(3,3),
        activation='relu',
        padding='same',
        strides=(2,2)
    )
)

# transformando a matriz em vetores
autoencoder.add(
    Flatten()
)

autoencoder.summary()


# camada oculta de reconstrução
# depois do summart acima, você consegue ver a dimensão da ultima camda para adicionar a dimensão
# conv2d_9 (Conv2D)           (None, 4, 4, 8)           584
autoencoder.add(
    Reshape(
        (4, 4, 8)
    )
)



# Decoder

# primera camada convolucional
autoencoder.add(
    Conv2D(
        filters=8,
        kernel_size=(3,3),
        activation='relu',
        padding='same' 
    )
)

# aumenta  a dimensionalidade
autoencoder.add(
    UpSampling2D(
        size=(2,2)
    )
)

# segunda camada convolucional
autoencoder.add(
    Conv2D(
        filters=8,
        kernel_size=(3,3),
        activation='relu',
        padding='same' 
    )
)

autoencoder.add(
    UpSampling2D(
        size=(2,2)
    )
)

# terceira camada convolucional
autoencoder.add(
    Conv2D(
        filters=16,
        kernel_size=(3,3),
        activation='relu'
    )
)

autoencoder.add(
    UpSampling2D(
        size=(2,2)
    )
)

# camada de resposta
autoencoder.add(
    Conv2D(
        filters=1,
        kernel_size=(3,3),
        activation='sigmoid',
        padding='same' 
    )
)

# visualizando a rede
autoencoder.summary()

# compilando o modelo
# outras metricas = 
# 'mean_squared_error' = quanto menor melhor, 
# 'mean_absolute_error' =  quanto menor melhor, 
# 'accuracy' = quanto maior proximo de 1 (este é recomendado para redes de classifcação e não autoencoders)
# coeff_determination/r-square = quanto mais próximo de 1 melhor

autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[coeff_determination]
)

# treinando o modelo

autoencoder.fit(
    previsores_treinamento,
    previsores_treinamento,
    epochs=10,
    batch_size=256,
    validation_data=(previsores_teste, previsores_teste)
)

# codificando
encoder = Model(
    inputs=autoencoder.input,
    outputs=autoencoder.get_layer('flatten').output
)
encoder.summary()

# codificando as imagens
imagens_codificadas = encoder.predict(previsores_teste)

# decodificando as imagens
imagens_decodificadas = autoencoder.predict(previsores_teste)

# visualizando os resultados
# capturar 10 numeros aleatórios de 0 a 10000 registros
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size=numero_imagens)

# criando o gráfico
plt.figure(figsize=(18, 18))
for i, idx_img in enumerate(imagens_teste):
    # imagem original
    eixo = plt.subplot(10, 10, i +1)
    plt.imshow(previsores_teste[idx_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10, 10, i +1 + numero_imagens)
    plt.imshow(imagens_codificadas[idx_img].reshape(16,8)) # 16 * 8 = 128
    plt.xticks(())
    plt.yticks(())
    
    # imagem decodifica/reconstruida
    eixo = plt.subplot(10, 10, i +1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[idx_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
graphic_view()