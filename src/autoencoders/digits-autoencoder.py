import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model, Sequential

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

# Entrada : 784
# Neuronio da camada oculta: 32
# Fator de compactação: 784 / 32 = 24.5
# Saida : 784
# criando o autoenconder

autoenconder = Sequential()

# por padrão a relu da bons resultados
# camada de entrada com a camada oculta
autoenconder.add(
    Dense(
        units=32,
        activation='relu',
        input_dim= 784
    )
)

# podemos usar sigmoid nesse caso, pois deixamos os dados normalizados
# muito comum usar a tangente hiperbolica
# camada de saida / reconstrução
autoenconder.add(
    Dense(
        units=784,
        activation='sigmoid'
    )
)

# mostrando a estrutura da rede neural
print(autoenconder.summary())

# compilando o modelo
autoenconder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# como a saida tem que ser igual a entrada o terinamento será com o mesmo previsor
# treinando o modelo
autoenconder.fit(
    previsores_treinamento,
    previsores_treinamento,
    epochs=50,
    batch_size=256,
    validation_data = (previsores_teste, previsores_teste)
)