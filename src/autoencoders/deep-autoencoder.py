# conhecido também como stack autoencoder

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras import backend as K
from rich import print

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

# normalizando o dado entre 0 e 1, mas poderiamos usar o MinMaxScaler
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

# adicionado dimensões na base
# 60000 registros, por 784 pixels
# cada uma dessas linhas nós temos uma imagem
previsores_treinamento = previsores_treinamento.reshape(
    (len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))
)
# adicionado dimensões na base
# 10000 registros, por 784 pixels
# cada uma dessas linhas nós temos uma imagem
previsores_teste = previsores_teste.reshape(
    (len(previsores_teste), np.prod(previsores_teste.shape[1:]))
)

# Entrada1 : 784
# Encode1 : 128
# Encode2 : 64
# Neuronio da camada oculta: 32
# Fator de compactação: 784 / 32 = 24.5
# Decode1 : 64
# Decode2 : 128
# Saida3 : 784
# Total Neuronios: 784 - 128 - 64 - 32 - 64 - 128 - 784
# criando o deep autoencoder
autoencoder = Sequential()

# Encode - Camada de entrada e Primeira camada de encode
autoencoder.add(
    Dense(
        units=128,
        activation='relu',
        input_dim=784
    )
)
# Encode - Segunda camada de encode
autoencoder.add(
    Dense(
        units=64,
        activation='relu'
    )
)
# Encode - Camada de compressão/oculta
autoencoder.add(
    Dense(
        units=32,
        activation='relu'
    )
)

# Decode - Primeira camada de decode
autoencoder.add(
    Dense(
        units=64,
        activation='relu'
    )
)
# Decode - Segunda camada de decode
autoencoder.add(
    Dense(
        units=128,
        activation='relu'
    )
)
# Decode - Camada de sáida
autoencoder.add(
    Dense(
        units=784,
        activation='sigmoid'
    )
)

# visualizando os detalkhes da rede
print(autoencoder.summary())

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
    epochs=50,
    batch_size=256,
    validation_data=(previsores_teste, previsores_teste)
)

# a apredizagem da rede (codificador)
dimensao_original = Input(
    shape=(784,)
)
camada_encoder1 = autoencoder.layers[0]
camada_encoder2 = autoencoder.layers[1]
camada_encoder3 = autoencoder.layers[2]

encoder = Model(
    dimensao_original,
    camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original)))
)

print(encoder.summary())

imagens_codificadas = encoder.predict(previsores_teste)

# decodificado

imagens_decodificadas =  autoencoder.predict(previsores_teste)

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
    plt.imshow(imagens_codificadas[idx_img].reshape(8,4)) # 8 * 4 = 32
    plt.xticks(())
    plt.yticks(())
    
    # imagem decodifica/reconstruida
    eixo = plt.subplot(10, 10, i +1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[idx_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
graphic_view()