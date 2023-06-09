import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.utils import np_utils

#utils
def graphic_view():
    plt.axis('off')
    plt.show()

# carregando as bases de dados
(previsores_treinamento, classe_treinamento),\
(previsores_teste, classe_teste) = mnist.load_data()
# dim treinamento = (60000, 28, 28) = 28 x 28 = 784 Pixels
# dim teste = (10000, 28, 28) = 28 x 28 = 784 Pixels

# normalizando o dado entre 0 e 1, mas poderiamos usar o MinMaxScaler
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

# processo de onehotencoder
classe_dummy_treinamento = np_utils.to_categorical(classe_treinamento)
classe_dummy_teste = np_utils.to_categorical(classe_teste)

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

# Entrada : 784
# Neuronio da camada oculta: 32
# Fator de compactação: 784 / 32 = 24.5
# Saida : 784
# criando o autoencoder

autoencoder = Sequential()

# por padrão a relu da bons resultados
# camada de entrada com a camada oculta / encode
autoencoder.add(
    Dense(
        units=32,
        activation='relu',
        input_dim= 784
    )
)

# podemos usar sigmoid nesse caso, pois deixamos os dados normalizados
# muito comum usar a tangente hiperbolica
# camada de saida / reconstrução / decode
autoencoder.add(
    Dense(
        units=784,
        activation='sigmoid'
    )
)

# mostrando a estrutura da rede neural
print(autoencoder.summary())

# compilando o modelo
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# como a saida tem que ser igual a entrada o terinamento será com o mesmo previsor
# treinando o modelo
autoencoder.fit(
    previsores_treinamento,
    previsores_treinamento,
    epochs=50,
    batch_size=256,
    validation_data = (previsores_teste, previsores_teste)
)

# codificando e decodificando para gerar a imagem
dimensao_original = Input(
    shape=(784,)
)

# primeira camada que nós contruimos acima
camada_encoder = autoencoder.layers[0]
encoder = Model(
    dimensao_original,
    camada_encoder(dimensao_original)
)
encoder.summary()

# encodificando
previsores_treinamento_codificados = encoder.predict(previsores_treinamento)
previsores_teste_codificados = encoder.predict(previsores_teste)

# criando duas redes neurais
# Neoronios: 784 entradas + 10 saidas / 2 = 397
# primeira : sem redução de dimensionalidade
c1 = Sequential()
c1.add(
    Dense(
        units=397,
        activation='relu',
        input_dim = 784
    )
)

## segunda camada
c1.add(
    Dense(
        units=397,
        activation='relu'
    )
)

# camada de saída
c1.add(
    Dense(
        units=10,
        activation='softmax'
    )
)

#compilando a rede neural
c1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# treinando o modelo
c1.fit(
    previsores_treinamento,
    classe_dummy_treinamento,
    batch_size=256,
    epochs=100,
    validation_data=(previsores_teste, classe_dummy_teste)
)

# Neoronios: 32 entradas + 10 saidas / 2 = 21
# segundo : sem redução de dimensionalidade
c2 = Sequential()
c2.add(
    Dense(
        units=21,
        activation='relu',
        input_dim = 32
    )
)

## segunda camada
c2.add(
    Dense(
        units=21,
        activation='relu'
    )
)

# camada de saída
c2.add(
    Dense(
        units=10,
        activation='softmax'
    )
)

#compilando a rede neural
c2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# treinando o modelo
c2.fit(
    previsores_treinamento_codificados,
    classe_dummy_treinamento,
    batch_size=256,
    epochs=100,
    validation_data=(previsores_teste_codificados, classe_dummy_teste)
)

# ficou 3% a menos que o resultado sem encoder, porém rodou muito mais ráido
# é bom avaliar a performnace, custo computacional e o resultado.